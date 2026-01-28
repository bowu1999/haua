import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from ..utils.bbox import cxcywh2xyxy, boxIou, generalizedBoxIou


__all__ = ['RTDETRCriterionv2']


class RTDETRCriterionv2(nn.Module):
    """
    该类用于计算 DETR (RT-DETR) 的损失。
    过程分为两步：
        1) 计算真实框 (ground truth) 与模型输出之间的匈牙利匹配 (Hungarian assignment)
        2) 对每一对匹配的 真实框/预测结果 进行监督训练 (监督类别和边界框)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False):
        """
        创建损失计算准则。
        参数:
            matcher: 用于计算目标和预测框之间匹配关系的模块
            num_classes: 目标类别的数量，不包含特殊的“无物体”类别
            weight_dict: 字典，键为损失名称，值为其相对权重
            losses: 需要计算的所有损失列表。可用损失列表请参考 get_loss 方法
            alpha: Focal Loss 的 alpha 参数
            gamma: Focal Loss 的 gamma 参数
            boxes_weight_format: 边界框权重的格式 (如 'iou', 'giou' 等)，用于加权 Loss
            share_matched_indices: 是否在辅助头 (Auxiliary Heads) 之间共享匹配索引
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """
        计算 Focal Loss (用于分类)。
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        # 获取匹配后的预测索引
        idx = self._get_src_permutation_idx(indices)
        
        # 获取匹配后的目标类别
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 初始化目标类别张量，默认填充为背景类 (self.num_classes)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 将匹配到的位置填入真实类别
        target_classes[idx] = target_classes_o
        
        # 转为 One-hot 编码，并去掉背景类的一列 (Focal Loss 通常处理前景分类)
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        
        # 计算 Sigmoid Focal Loss
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction='none')
        
        # 归一化 Loss
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        """
        计算 Varifocal Loss (VFL)，通常用于结合 IoU 质量分数的分类 Loss。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # 如果没有传入 values (通常是 IoU)，则现场计算预测框和真实框的 IoU
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = boxIou(cxcywh2xyxy(src_boxes), cxcywh2xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # 构建 VFL 的目标分数：正样本位置为 IoU 值，负样本为 0
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        
        # VFL 权重计算公式
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """
        计算边界框相关的损失：L1 回归损失和 GIoU 损失。
        targets 字典必须包含键 "boxes"，其值为维度 [nb_target_boxes, 4] 的张量。
        目标框格式应为 (center_x, center_y, w, h)，并按图像尺寸归一化。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        
        # 计算 L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 计算 GIoU Loss
        loss_giou = 1 - torch.diag(
            generalizedBoxIou(cxcywh2xyxy(src_boxes), cxcywh2xyxy(target_boxes)))
        
        # 如果有额外的框权重 (如基于 IoU 加权)，则应用权重
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # 根据匹配索引重新排列预测结果 (Batch 维度和 Query 维度)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 根据匹配索引重新排列目标结果
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'你确定要计算 {loss} 损失吗？'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, num_boxes=None, **kwargs):
        """
        参数:
            num_boxes (float): 全局平均目标框数量 (Global Average Number of Boxes)。
                               由上层模块计算好并传入。
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # 1. 如果外部没传 num_boxes，就计算局部的 (兼容单卡调试)
        if num_boxes is None:
            # 这是一个 fallback，仅用于调试或非标准调用
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = max(num_boxes, 1.0)
        
        # 获取最后一层输出与目标之间的匹配关系
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # 如果存在辅助损失 (Auxiliary Losses)，对每个中间层的输出重复上述过程
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 如果不共享匹配索引，则对每层重新进行匹配
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] \
                            for k in l_dict if k in self.weight_dict}
                    # 为辅助损失添加后缀，如 _aux_0, _aux_1
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 如果存在 CDN (Contrastive Denoising) 辅助损失 (RT-DETR 特有)
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            # CDN 的框数量需要乘以组数
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 如果存在 Encoder 辅助损失 (RT-DETR v2 特有)
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            # 如果是类别无关的 (Class Agnostic)，将所有目标类别设为 0 (前景)
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            # 恢复原始类别数
            if class_agnostic:
                self.num_classes = orig_num_classes # type: ignore

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        """
        获取计算 Loss 所需的元信息 (如 IoU 值用于加权)。
        """
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = boxIou(cxcywh2xyxy(src_boxes.detach()), cxcywh2xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalizedBoxIou(
                cxcywh2xyxy(src_boxes.detach()), cxcywh2xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """
        获取 CDN (Contrastive Denoising) 的匹配索引。
        CDN 产生的去噪 Query 是有固定对应关系的，不需要匈牙利匹配。
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                # 复制多组 (CDN Group)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices