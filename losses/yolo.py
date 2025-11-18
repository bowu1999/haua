import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


from .iou import ciou_loss
from .assigner import tal_assign
from ..models.utils import make_grid, decode_dfl, bbox_iou, bbox_area


class YOLOv8Loss(nn.Module):
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        num_classes: int = 80,
        dfl_bins: int = 16,
        loss_cls_weight: float = 0.5,
        loss_iou_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        tal_topk: int = 10,
    ):
        super().__init__()
        self.strides = strides
        self.num_classes = num_classes
        self.dfl_bins = dfl_bins
        self.loss_cls_weight = loss_cls_weight
        self.loss_iou_weight = loss_iou_weight
        self.loss_dfl_weight = loss_dfl_weight
        self.tal_topk = tal_topk

    def _make_dfl_targets(
            self,
            gt_boxes: torch.Tensor,
            matched_gt_inds: torch.Tensor,
            grid: torch.Tensor,
            stride: int
        ) -> torch.Tensor:
        """
        Build DFL soft targets for each positive anchor.
        gt_boxes: (B, G, 4)
        matched_gt_inds: (B, N) values in -1 or [0,G-1]
        grid: (N,2)
        returns: target_dist: (B, N, 4, bins) where negatives get zeros
        """
        B, N = matched_gt_inds.shape
        device = matched_gt_inds.device
        bins = self.dfl_bins
        target_dist = torch.zeros((B, N, 4, bins), device=device)

        for b in range(B):
            matched = matched_gt_inds[b]  # (N,)
            pos_idx = (matched >= 0).nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            # get gt for each positive anchor
            assigned = matched[pos_idx]
            boxes = gt_boxes[b][assigned]  # (P,4)
            # centers
            cx = grid[:, 0][pos_idx]
            cy = grid[:, 1][pos_idx]
            # distances in pixels
            l = (cx - boxes[:, 0]).clamp(min=0)
            t = (cy - boxes[:, 1]).clamp(min=0)
            r = (boxes[:, 2] - cx).clamp(min=0)
            b_ = (boxes[:, 3] - cy).clamp(min=0)
            dists = torch.stack([l, t, r, b_], dim=1)  # (P,4)
            # convert to bin units
            t_bins = dists / stride
            # clamp within [0, bins-1]
            t_bins = t_bins.clamp(0, bins - 1 - 1e-6)
            lower = t_bins.floor().long()
            upper = lower + 1
            upper = upper.clamp(max=bins - 1)
            alpha = (t_bins - lower.float())  # (P,4)
            # fill target_dist
            for i_coord in range(4):
                li = lower[:, i_coord]
                ui = upper[:, i_coord]
                a = alpha[:, i_coord]
                # scatter
                target_dist[b, pos_idx, i_coord].scatter_add_(
                    1, li.unsqueeze(1), (1 - a).unsqueeze(1))
                target_dist[b, pos_idx, i_coord].scatter_add_(1, ui.unsqueeze(1), a.unsqueeze(1))

        return target_dist

    def forward(self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: dict) -> dict:
        """
        preds: tuple of 3 feature map outputs, each [B, C=144, H, W]
        batch: dict containing at least:
           'gt_bboxes': tensor (B, G, 4) xyxy (padded with zeros for missing)
           'gt_labels': tensor (B, G) long
        returns losses dict
        """
        device = preds[0].device
        B = preds[0].shape[0]

        per_scale_scores = []  # list of (B, N, C)
        per_scale_dfl = []     # list of (B, N, 4, bins)
        per_scale_bboxes = []             # list of grid (N,2)
        Ns = []
        grids = []  

        # 解析所有尺度的输出
        for i, p in enumerate(preds):
            stride = self.strides[i]
            Bp, Cc, H, W = p.shape
            N = H * W
            Ns.append(N)
            # reshape
            dfl, cls_logits = torch.split(p, [64, self.num_classes], dim=1)
            per_bboxes, _ = decode_dfl(dfl, stride=stride)
            per_scale_bboxes.append(per_bboxes)
            cls_logits = cls_logits.permute(0, 2, 3, 1).view(Bp, N, self.num_classes)
            per_scale_scores.append(cls_logits)
            dfl = dfl.permute(0, 2, 3, 1).view(Bp, N, 4, self.dfl_bins)
            per_scale_dfl.append(dfl)
            grid = make_grid((H, W), stride, device)
            grid = grid.view(N, 2)
            grids.append(grid)
        # 拼接所有尺度的输出
        all_scores = torch.cat(per_scale_scores, dim=1)  # (B, N_total, C)
        all_dfl = torch.cat(per_scale_dfl, dim=1)        # (B, N_total, 4, bins)
        pred_boxes = torch.cat(per_scale_bboxes, dim=1)   # (B, N_total, 4)
        all_grids = torch.cat(grids, dim=0)
        # run TAL assigner
        gt_bboxes = batch['gt_bboxes']
        gt_labels = batch['gt_labels']
        # expect shapes: (B, G, 4), (B, G)
        target_scores, target_bboxes, fg_mask, matched_gt_inds = tal_assign(
            all_scores, pred_boxes, gt_bboxes, gt_labels, topk=self.tal_topk)
        
        # 1. 分类损失（BCEwithLogits，软目标）
        loss_cls = F.binary_cross_entropy_with_logits(all_scores, target_scores, reduction='none')  # (B,N,C)
        # 对类和锚点求和
        loss_cls = loss_cls.sum() / max(1, int(fg_mask.sum().item()))

        # 2. 正样本 CIoU Loss
        B, N_total, _ = all_scores.shape
        pos_mask = fg_mask
        pos_idx = pos_mask.view(B, N_total)
        num_pos = pos_idx.sum().item()
        if num_pos > 0:
            pred_pos = pred_boxes[pos_idx]
            tgt_pos = target_bboxes[pos_idx]
            loss_iou = ciou_loss(pred_pos, tgt_pos)
        else:
            loss_iou = torch.tensor(0.0, device=device)

        # 3. 正样本 DFL loss
        target_dfl = self._make_dfl_targets(gt_bboxes, matched_gt_inds, all_grids, stride=1)
        if num_pos > 0:
            pred_dfl = all_dfl[pos_idx]  # (P,4,bins)
            targ = target_dfl[pos_idx]
            # log softmax over bins
            lsm = F.log_softmax(pred_dfl, dim=-1)
            loss_dfl = -(targ * lsm).sum() / max(1, num_pos)
        else:
            loss_dfl = torch.tensor(0.0, device=device)

        return {
            'loss_iou': self.loss_iou_weight * loss_iou,
            'loss_cls': self.loss_cls_weight * loss_cls,
            'loss_dfl': self.loss_dfl_weight * loss_dfl}


class YOLOv10Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one2one_loss = YOLOv8Loss(tal_topk=1)
        self.one2many_loss = YOLOv8Loss(tal_topk=10)
    
    def forward(self, preds, targs):
        one2one_loss = self.one2one_loss(preds['one2one'], targs)
        one2many_loss = self.one2many_loss(preds['one2many'], targs)
    
        return {'loss_one2one': one2one_loss, 'loss_one2many': one2many_loss}
