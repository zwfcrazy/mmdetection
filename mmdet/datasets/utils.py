# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
from functools import partial
from typing import Sequence
from mmcv.transforms import LoadImageFromFile

from mmdet.datasets.transforms import LoadAnnotations, LoadPanopticAnnotations
from mmdet.registry import TRANSFORMS
from mmengine.dataset import COLLATE_FUNCTIONS


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = TRANSFORMS.get(cfg['type'])
        # TODO：use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations,
                                               LoadPanopticAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg


def _get_collate_fn(collate_fn_cfg: dict):
    collate_fn_cfg = copy.deepcopy(collate_fn_cfg)
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
    collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    return collate_fn


@COLLATE_FUNCTIONS.register_module()
def semi_collate(data_batch: Sequence,
                teacher_collate_fn: dict,
                student_collate_fn: dict) -> dict:
    """apply different collate functions for student and teacher separately
    """
    
    teacher_collate_fn = _get_collate_fn(teacher_collate_fn)
    student_collate_fn = _get_collate_fn(student_collate_fn)
    
    teacher_data_batch = copy.deepcopy(data_batch)
    student_data_batch = copy.deepcopy(data_batch)
    
    for i in range(len(data_batch)):
        teacher_data_batch[i]['data_samples'] = data_batch[i]['data_samples']['teacher']
        teacher_data_batch[i]['inputs'] = data_batch[i]['inputs']['teacher']
        student_data_batch[i]['data_samples'] = data_batch[i]['data_samples']['student']
        student_data_batch[i]['inputs'] = data_batch[i]['inputs']['student']
        
    teacher_collated_results = teacher_collate_fn(teacher_data_batch)
    student_collated_results = student_collate_fn(student_data_batch)
    
    collated_results = {
        'data_samples': {
            'teacher': teacher_collated_results['data_samples'],
            'student': student_collated_results['data_samples']
        },
        'inputs': {
            'teacher': teacher_collated_results['inputs'],
            'student': student_collated_results['inputs']
        }
    }
    
    return collated_results


@COLLATE_FUNCTIONS.register_module()
def semi_yolov5_collate(data_batch: Sequence,
                   use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    batch_data_samples = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)
        batch_data_samples.append(datasamples)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to_tensor(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0),
            'original_samples': batch_data_samples
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)
    return collated_results