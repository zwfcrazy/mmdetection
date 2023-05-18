import copy
from typing import List, Optional, Tuple, Dict

import torch
from mmengine.structures import InstanceData
from torch import Tensor
from mmcv.ops import batched_nms

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_project
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from .semi_base import SemiBaseDetector


@MODELS.register_module()
class MultiDatasetDetector(SemiBaseDetector):
    r"""
    """
    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        
        x = self.student.extract_feat(multi_batch_inputs['student'])
        
        cls_data_samples, bbox_data_samples = self.get_pseudo_instances(
            multi_batch_inputs['teacher'], multi_batch_data_samples['teacher'])
        
        student_head_outs = self.student.bbox_head(x)
        
        cls_pseduo_samples = self._process_pseudo_samples(multi_batch_data_samples['student'], cls_data_samples)
        cls_loss = self.student.bbox_head.loss_by_feat(*student_head_outs, \
            batch_gt_instances=cls_pseduo_samples['bboxes_labels'], batch_img_metas=cls_pseduo_samples['img_metas'])
        bbox_pseduo_samples = self._process_pseudo_samples(multi_batch_data_samples['student'], bbox_data_samples)
        bbox_loss = self.student.bbox_head.loss_by_feat(*student_head_outs, \
            batch_gt_instances=bbox_pseduo_samples['bboxes_labels'], batch_img_metas=bbox_pseduo_samples['img_metas'])
        
        losses = {
            'loss_cls': cls_loss['loss_cls'],
            'loss_bbox': bbox_loss['loss_bbox'],
            'loss_dfl': bbox_loss['loss_dfl'],
        }
        # losses = self.student.bbox_head.loss(x, multi_batch_data_samples['student'])
        
        return losses
    
    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        self.teacher.eval()
        pred_data_samples = self.teacher.predict(batch_inputs, batch_data_samples)
        cls_data_samples = copy.deepcopy(pred_data_samples)
        bbox_data_samples = copy.deepcopy(pred_data_samples)
        for i, data_sample in enumerate(pred_data_samples):
            cls_pseudo_instances = data_sample.pred_instances[
                data_sample.pred_instances.scores > 
                self.semi_train_cfg.cls_pseudo_thr
            ]
            cls_data_samples[i].pseudo_instances = cls_pseudo_instances
            bbox_pseudo_instances = data_sample.pred_instances[
                data_sample.pred_instances.scores > 
                self.semi_train_cfg.bbox_pseudo_thr
            ]
            bbox_data_samples[i].pseudo_instances = bbox_pseudo_instances
            
        return cls_data_samples, bbox_data_samples
    
    def _process_pseudo_samples(self, student_samples, teacher_pred_samples):
        device = student_samples['bboxes_labels'].device
        _student_samples = copy.deepcopy(student_samples['original_samples'])
        for student_sample, teacher_sample in zip(_student_samples, teacher_pred_samples):
            # transform teacher bboxes to student image space
            student_matrix = torch.tensor(student_sample.homography_matrix, device=device)
            teacher_matrix = torch.tensor(teacher_sample.homography_matrix, device=device)
            pseudo_bboxes = teacher_sample.pseudo_instances.bboxes
            homography_matrix = teacher_matrix @ student_matrix.inverse()
            projected_bboxes = bbox_project(pseudo_bboxes, homography_matrix,
                                            teacher_sample.img_shape)
            student_sample.pseudo_instances = copy.deepcopy(teacher_sample.pseudo_instances)
            student_sample.pseudo_instances.bboxes = projected_bboxes
            
            # merge gt and prediction
            student_sample.gt_instances.scores = torch.ones_like(student_sample.gt_instances.labels, dtype=torch.float32, device=device)
            student_sample.gt_instances.bboxes = student_sample.gt_instances.bboxes.tensor
            # print('gt %d pseudo %d' % (len(student_sample.gt_instances), len(student_sample.pseudo_instances)))
            student_sample.pseudo_instances = InstanceData.cat([student_sample.pseudo_instances, student_sample.gt_instances])
            student_sample.pseudo_instances = student_sample.gt_instances
            # use nms to filter out predictions that overlap gt
            if student_sample.pseudo_instances.bboxes.shape[0] > 1:
                det_bboxes, keep_idxs = batched_nms(student_sample.pseudo_instances.bboxes, student_sample.pseudo_instances.scores,
                                                    student_sample.pseudo_instances.labels, self.semi_train_cfg.nms)
                student_sample.pseudo_instances = student_sample.pseudo_instances[keep_idxs]
        
        batch_bboxes_labels = []
        for i, student_sample in enumerate(_student_samples):
            pseudo_bboxes = student_sample.pseudo_instances.bboxes
            pseudo_labels = student_sample.pseudo_instances.labels
            batch_idx = pseudo_labels.new_full((len(pseudo_labels), 1), i)
            bboxes_labels = torch.cat((batch_idx, pseudo_labels[:, None], pseudo_bboxes), dim=1)
            batch_bboxes_labels.append(bboxes_labels)
        
        bboxes_labels = torch.cat(batch_bboxes_labels, 0)
        pseduo_samples = {
            'bboxes_labels': bboxes_labels,
            'img_metas': student_samples['img_metas']
        }
        return pseduo_samples