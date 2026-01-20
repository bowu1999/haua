import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from .iou import ciou_loss
from .assigner import tal_assign
from ..models.utils import make_grid, decode_dfl, bbox_iou, bbox_area
from .seg import InstanceSegLoss


class YOLOv8Loss(nn.Module):
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        num_classes: int = 80,
        dfl_bins: int = 16,
        loss_cls_weight: float = 0.5,
        loss_iou_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        cls_loss_type: str = "bce",      # 目前仍使用 BCE/Focal，只对正样本+对应类
        label_smoothing: float = 0.0,
        use_focal: bool = True,          # 建议默认 True
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tal_topk: int = 10,
        debug: bool = False,
        pos_cls_weight: float = 1.0,
        neg_cls_weight: float = 0.1,     # 降低负样本权重，避免负样本主导
        return_midvars: bool = False,
    ):
        """
        更接近 YOLOv8 风格的损失实现：
        - DFL + IoU 只在正样本上
        - 分类只在正样本 + 对应类上主学，负样本弱权重
        """
        super().__init__()
        assert cls_loss_type in ("bce", "ce")
        self.strides = list(strides)
        self.num_classes = num_classes
        self.dfl_bins = dfl_bins
        self.loss_cls_weight = loss_cls_weight
        self.loss_iou_weight = loss_iou_weight
        self.loss_dfl_weight = loss_dfl_weight
        self.cls_loss_type = cls_loss_type
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tal_topk = tal_topk
        self.debug = debug
        self.return_midvars = return_midvars
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight

    @staticmethod
    def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        logits: (N, C)
        targets: (N, C) in {0,1}
        """
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        mod = (1.0 - p_t).pow(gamma)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        loss = alpha_t * mod * ce
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

    def _make_dfl_targets(
        self,
        gt_boxes: Union[List[torch.Tensor], torch.Tensor],
        matched_gt_inds: torch.Tensor,
        grid: torch.Tensor,
        stride_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Make DFL targets.

        Args:
            gt_boxes: either
                - list of length B, each Tensor (Gi,4) in xyxy; or
                - Tensor (B,G,4)
            matched_gt_inds: (B,N)  正样本对应 gt 的索引, 负样本=-1
            grid: (N,2)  anchor 中心像素坐标
            stride_map: (N,)
        Returns:
            target_dist: (B, N, 4, bins)
        """
        B, N = matched_gt_inds.shape
        device = matched_gt_inds.device
        bins = self.dfl_bins

        target_dist = torch.zeros((B, N, 4, bins), device=device)

        if stride_map is None:
            stride_map = torch.ones((N,), device=device, dtype=grid.dtype)
        else:
            assert stride_map.shape[0] == N

        is_list = isinstance(gt_boxes, (list, tuple))

        for b_idx in range(B):
            m = matched_gt_inds[b_idx]        # (N,)
            pos_idx = (m >= 0).nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            assigned = m[pos_idx]             # (P,)

            # per-image gt boxes
            if is_list:
                boxes_b = gt_boxes[b_idx]
                if not isinstance(boxes_b, torch.Tensor):
                    boxes_b = torch.as_tensor(boxes_b, device=device)
                else:
                    boxes_b = boxes_b.to(device)
                if boxes_b.numel() == 0:
                    continue
                boxes = boxes_b[assigned.long()]  # (P,4)
            else:
                boxes = gt_boxes[b_idx].to(device)[assigned.long()]

            cx = grid[:, 0][pos_idx]
            cy = grid[:, 1][pos_idx]

            # l, t, r, b
            l = (cx - boxes[:, 0]).clamp(min=0)
            t = (cy - boxes[:, 1]).clamp(min=0)
            r = (boxes[:, 2] - cx).clamp(min=0)
            b_ = (boxes[:, 3] - cy).clamp(min=0)
            dists = torch.stack([l, t, r, b_], dim=1)   # (P,4)

            s = stride_map[pos_idx].unsqueeze(1)
            t_bins = (dists / s).clamp(0, bins - 1 - 1e-6)

            lower = t_bins.floor().long()
            upper = (lower + 1).clamp(max=bins - 1)
            alpha = t_bins - lower.float()

            P_ = pos_idx.numel()
            for i_coord in range(4):
                li = lower[:, i_coord]
                ui = upper[:, i_coord]
                a = alpha[:, i_coord]
                temp = torch.zeros((P_, bins), device=device)
                temp.scatter_add_(1, li.unsqueeze(1), (1 - a).unsqueeze(1))
                temp.scatter_add_(1, ui.unsqueeze(1), a.unsqueeze(1))
                target_dist[b_idx, pos_idx, i_coord] = temp

        return target_dist

    def forward(self, preds: Tuple[torch.Tensor, ...], batch: dict) -> dict:
        """
        preds: tuple of pyramid predictions, each: (B, 4*dfl_bins + num_classes, H, W)
        batch: {
            "gt_bboxes": list[Tensor(Gi,4)] or Tensor(B,G,4) in xyxy,
            "gt_labels": list[Tensor(Gi,)]  or Tensor(B,G)
        }
        """
        device = preds[0].device
        B = preds[0].shape[0]

        per_scale_cls = []
        per_scale_dfl = []
        per_scale_boxes = []
        Ns = []
        grids = []

        # 解析各尺度预测
        for i, p in enumerate(preds):
            stride = self.strides[i]
            Bp, Cc, H, W = p.shape
            N = H * W
            Ns.append(N)

            dfl, cls_logits = torch.split(p, [4 * self.dfl_bins, self.num_classes], dim=1)

            # bbox 解码 (B, N, 4)  —— decode_dfl 返回 xyxy
            per_boxes, _ = decode_dfl(dfl, stride=stride)
            per_scale_boxes.append(per_boxes)

            # 分类 logits (B, N, C)
            cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(Bp, N, self.num_classes)
            per_scale_cls.append(cls_logits)

            # DFL logits (B, N, 4, bins)
            dfl = dfl.permute(0, 2, 3, 1).reshape(Bp, N, 4, self.dfl_bins)
            per_scale_dfl.append(dfl)

            # 网格中心 (N, 2)
            grid = make_grid((H, W), stride, device)  # (H*W, 2)
            grids.append(grid.view(N, 2))

        # 拼接所有尺度
        all_cls_logits = torch.cat(per_scale_cls, dim=1)   # (B, N_tot, C)
        all_dfl = torch.cat(per_scale_dfl, dim=1)          # (B, N_tot, 4, bins)
        pred_bboxes = torch.cat(per_scale_boxes, dim=1)    # (B, N_tot, 4) xyxy
        all_grids = torch.cat(grids, dim=0)                # (N_tot, 2)

        # stride map
        stride_list = [
            torch.full((N,), self.strides[i], device=device, dtype=all_grids.dtype)
            for i, N in enumerate(Ns)]
        stride_map = torch.cat(stride_list, dim=0)         # (N_tot,)

        gt_bboxes = batch["gt_bboxes"]
        gt_labels = batch["gt_labels"]

        # ============ 匹配（Task-Aligned Assigner） ============
        with torch.no_grad():
            # 使用 pred_cls.sigmoid 作为分类置信度
            cls_prob = all_cls_logits.sigmoid()    # (B, N_tot, C)
            # target_scores: (B, N_tot, C)
            # target_bboxes: (B, N_tot, 4)
            # fg_mask:       (B, N_tot) bool/0-1
            # matched_gt_inds: (B, N_tot) long，正样本对应 gt 索引，负样本=-1
            target_scores, target_bboxes, fg_mask, matched_gt_inds = tal_assign(
                cls_prob, pred_bboxes, gt_bboxes, gt_labels, topk=self.tal_topk)

        pos_mask = fg_mask.bool()      # (B, N_tot)
        num_pos = pos_mask.sum()       # 标量
        num_pos_den = num_pos.clamp(min=1).float()

        # ==================== IoU Loss（box）====================
        if num_pos > 0:
            pred_pos = pred_bboxes[pos_mask]   # (P,4)
            tgt_pos = target_bboxes[pos_mask]  # (P,4)
            loss_iou = ciou_loss(pred_pos, tgt_pos).mean()
        else:
            loss_iou = torch.tensor(0.0, device=device)

        # ==================== DFL Loss =========================
        if num_pos > 0:
            # 生成 DFL 目标 (B, N_tot, 4, bins)，只在正样本处非零
            target_dfl_all = self._make_dfl_targets(
                gt_bboxes,
                matched_gt_inds,  # (B,N_tot)
                all_grids,        # (N_tot,2)
                stride_map)        # (N_tot,)
            pred_dfl_pos = all_dfl[pos_mask]          # (P,4,bins)
            targ_dfl_pos = target_dfl_all[pos_mask]   # (P,4,bins)
            # 对每个方向上的 (bins) 做 softmax + 交叉熵
            log_probs = F.log_softmax(pred_dfl_pos, dim=-1)
            # P*4*bins 上求和，然后按正样本数归一化
            loss_dfl = -(targ_dfl_pos * log_probs).sum() / num_pos_den
        else:
            loss_dfl = torch.tensor(0.0, device=device)

        # 注意：官方直接使用 target_scores 作为 BCE/Focal 的 target
        # target_scores: (B, N_tot, C)，正样本对应类别 > 0，其他为 0
        # 负样本在所有类别上都为 0
        if self.label_smoothing > 0:
            # 如需保留，可对正样本的非零 scores 做缩放，下面是一个温和版本：
            # target' = target * (1 - ε) + ε/C
            eps = self.label_smoothing
            # 只对非零的 target_scores 做 smoothing
            non_zero_mask = target_scores > 0
            target_scores = target_scores * (1.0 - eps)
            # 均匀分布加到所有类上，严格模仿 one-hot smoothing 的话是加到全体；
            target_scores = target_scores + eps / self.num_classes

        pred_cls = all_cls_logits.reshape(-1, self.num_classes)       # (B*N_tot, C)
        target_cls = target_scores.reshape(-1, self.num_classes)      # (B*N_tot, C)

        if self.use_focal:
            # 使用 sigmoid_focal_loss
            loss_cls_raw = self.sigmoid_focal_loss(
                pred_cls, target_cls,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction='sum')
            loss_cls = loss_cls_raw / num_pos_den
        else:
            # 与官方类似：BCEWithLogitsLoss + 以正样本数归一化
            loss_cls_raw = F.binary_cross_entropy_with_logits(
                pred_cls, target_cls, reduction='sum')
            loss_cls = loss_cls_raw / num_pos_den
        
        result = {
            "box": self.loss_iou_weight * loss_iou,
            "dfl": self.loss_dfl_weight * loss_dfl,
            "cls": self.loss_cls_weight * loss_cls}

        if self.return_midvars:
            result['fg_mask'] = fg_mask
            result['matched_gt_inds'] = matched_gt_inds
            result['all_grids'] = all_grids

        return result


class YOLOv10Loss(nn.Module):
    def __init__(
        self,
        strides: list = [8, 16, 32],
        num_classes: int = 80,
        dfl_bins: int = 16,
        loss_cls_weight: float = 0.5,
        loss_iou_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        cls_loss_type: str = "bce",
        label_smoothing: float = 0.0,
        use_focal: bool = True,
        focal_alpha = 0.75,
        focal_gamma = 2.0,
        debug = False,
        o2m_weight: float = 0.8,
        pos_cls_weight: float = 1.0,
        neg_cls_weight: float = 0.1,
    ):
        super().__init__()
        self.o2m_weight = o2m_weight

        self.one2one_loss = YOLOv8Loss(
            strides = strides,
            num_classes = num_classes,
            dfl_bins = dfl_bins,
            loss_cls_weight = loss_cls_weight,
            loss_iou_weight = loss_iou_weight,
            loss_dfl_weight = loss_dfl_weight,
            cls_loss_type = cls_loss_type,
            label_smoothing = label_smoothing,
            use_focal = use_focal,
            focal_alpha = focal_alpha,
            focal_gamma = focal_gamma,
            debug = debug,
            pos_cls_weight = pos_cls_weight,
            neg_cls_weight = neg_cls_weight,
            tal_topk= 1)
        self.one2many_loss = YOLOv8Loss(
            strides = strides,
            num_classes = num_classes,
            dfl_bins = dfl_bins,
            loss_cls_weight = loss_cls_weight,
            loss_iou_weight = loss_iou_weight,
            loss_dfl_weight = loss_dfl_weight,
            cls_loss_type = cls_loss_type,
            label_smoothing = label_smoothing,
            use_focal = use_focal,
            focal_alpha = focal_alpha,
            focal_gamma = focal_gamma,
            debug = debug,
            pos_cls_weight = pos_cls_weight,
            neg_cls_weight = neg_cls_weight,
            tal_topk=10)

    def forward(self, preds, targs):
        """
        preds: {'one2one': [...], 'one2many': [...]}
        """
        one2one_loss = self.one2one_loss(preds['one2one'], targs)
        one2many_loss = self.one2many_loss(preds['one2many'], targs)
        loss = {}
        for k, v in one2one_loss.items():
            loss[f'loss_o2o_{k}'] = v * (1 - self.o2m_weight)
        for k, v in one2many_loss.items():
            loss[f'loss_o2m_{k}'] = v * self.o2m_weight

        return loss


class YOLOv11SegLoss(YOLOv10Loss):
    def __init__(
        self,
        strides: list = [8, 16, 32],
        num_classes: int = 80,
        dfl_bins: int = 16,
        loss_cls_weight: float = 0.5,
        loss_iou_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        cls_loss_type: str = "bce",
        label_smoothing: float = 0.0,
        use_focal: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        debug: bool = False,
        o2m_weight: float = 0.8,
        pos_cls_weight: float = 1.0,
        neg_cls_weight: float = 0.1,
        loss_seg_weight: float = 1.0,
        seg_debug: bool = False,
    ):
        super().__init__(
            strides, num_classes, dfl_bins, loss_cls_weight, loss_iou_weight, 
            loss_dfl_weight, cls_loss_type, label_smoothing, use_focal, 
            focal_alpha, focal_gamma, debug, o2m_weight, pos_cls_weight, neg_cls_weight)
        
        self.one2one_loss = YOLOv8Loss(
            strides = strides,
            num_classes = num_classes,
            dfl_bins = dfl_bins,
            loss_cls_weight = loss_cls_weight,
            loss_iou_weight = loss_iou_weight,
            loss_dfl_weight = loss_dfl_weight,
            cls_loss_type = cls_loss_type,
            label_smoothing = label_smoothing,
            use_focal = use_focal,
            focal_alpha = focal_alpha,
            focal_gamma = focal_gamma,
            debug = debug,
            pos_cls_weight = pos_cls_weight,
            neg_cls_weight = neg_cls_weight,
            tal_topk = 1,
            return_midvars = True)
        
        self.seg_loss = InstanceSegLoss(
            num_protos = 32,
            proto_size = (160, 160),
            loss_seg_weight = loss_seg_weight,
            bce_weight = 1.0,
            dice_weight = 0.2,
            eps = 1e-6)
    
    def forward(self, preds, targs):
        img_h, img_w = targs["img"].shape[-2:]
        # 检测损失
        o2o_out = self.one2one_loss(preds['one2one'], targs)
        o2m_out = self.one2many_loss(preds['one2many'], targs)
        loss = {}
        for k in ["box", "dfl", "cls"]:
            loss[f'loss_o2o_{k}'] = o2o_out[k] * (1 - self.o2m_weight)
        for k in ["box", "dfl", "cls"]:
            loss[f'loss_o2m_{k}'] = o2m_out[k] * self.o2m_weight
        # 分割损失
        fg_mask = o2o_out["fg_mask"]
        matched_gt_inds = o2o_out["matched_gt_inds"]
        seg_loss = self.seg_loss(
            seg_out = preds["seg_out"],
            prototype_mask = preds["prototype_mask"],
            fg_mask = fg_mask,
            matched_gt_inds = matched_gt_inds,
            gt_bboxes = targs["gt_bboxes"],
            gt_masks = targs["gt_masks"],
            img_shape = (img_h, img_w))
        loss["loss_seg"] = seg_loss
        
        return loss