from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceSegLoss(nn.Module):
    """
    YOLO-style Instance Segmentation Loss
    (Recommended version: crop loss, not logits)
    """
    def __init__(
        self,
        num_protos: int = 32,
        proto_size: Tuple[int, int] = (160, 160),
        loss_seg_weight: float = 10.0,
        bce_weight: float = 1.0,
        dice_weight: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_protos = num_protos
        self.proto_h, self.proto_w = proto_size
        self.loss_seg_weight = loss_seg_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    @staticmethod
    def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Apply bbox mask (NOT modifying logits semantics)
        Args:
            masks: (N, H, W)
            boxes: (N, 4) in xyxy (same scale as masks)
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = boxes.T

        r = torch.arange(w, device=masks.device)[None, None, :]
        c = torch.arange(h, device=masks.device)[None, :, None]

        return masks * (
            (r >= x1[:, None, None]) &
            (r <  x2[:, None, None]) &
            (c >= y1[:, None, None]) &
            (c <  y2[:, None, None]))

    def forward(
        self,
        seg_out: Tuple[torch.Tensor, ...],
        prototype_mask: torch.Tensor,
        fg_mask: torch.Tensor,
        matched_gt_inds: torch.Tensor,
        gt_bboxes: Union[List[torch.Tensor], torch.Tensor],
        gt_masks: Union[List[torch.Tensor], torch.Tensor],
        img_shape: Tuple[int, int],
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            seg_out: tuple of mask coefficients from different FPN levels
                     each shape: (B, C, H_i, W_i)
            prototype_mask: (B, num_protos, proto_h, proto_w)
            fg_mask: (B, num_preds)
            matched_gt_inds: (B, num_preds)
            gt_bboxes: list or tensor, (B, num_gt, 4) in image scale
            gt_masks: list or tensor, (B, num_gt, H, W)
            img_shape: (img_h, img_w)
        """
        device = prototype_mask.device
        img_h, img_w = img_shape

        # (B, num_protos, num_preds)
        mask_coef = torch.cat([s.flatten(2) for s in seg_out], dim=2)

        total_loss = torch.tensor(0.0, device=device)
        total_inst = 0

        for b in range(prototype_mask.size(0)):
            # 获取非零索引
            pos_idx = fg_mask[b].nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            # ---- coefficients ----
            coef = mask_coef[b, :, pos_idx].T
            coef = coef.clamp(-10, 10)  # numerical stability

            gt_idx = matched_gt_inds[b][pos_idx].long()

            gt_mask = gt_masks[b][gt_idx].float()
            gt_box = gt_bboxes[b][gt_idx].clone()

            # ---- proto -> mask logits ----
            proto = prototype_mask[b].view(self.num_protos, -1)
            pred = (coef @ proto).view(-1, self.proto_h, self.proto_w)

            # ---- resize GT mask ----
            gt_mask = F.interpolate(
                gt_mask.unsqueeze(1),
                size=(self.proto_h, self.proto_w),
                mode="bilinear",
                align_corners=False).squeeze(1)
            gt_mask = (gt_mask > 0.5).float()

            # ---- scale bbox to proto space ----
            scale_x = self.proto_w / img_w
            scale_y = self.proto_h / img_h
            gt_box[:, [0, 2]] *= scale_x
            gt_box[:, [1, 3]] *= scale_y

            # ---- bbox mask (loss only) ----
            box_mask = self.crop_mask(torch.ones_like(pred), gt_box)

            # ---- BCE (logit space) ----
            bce = F.binary_cross_entropy_with_logits(pred, gt_mask, reduction="none")
            bce = (bce * box_mask).sum(dim=(1, 2))

            # instance-wise normalization
            area = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
            bce = bce / (area + self.eps)

            # ---- Dice (probability space, optional) ----
            if self.dice_weight > 0:
                prob = torch.sigmoid(pred)
                prob = prob * box_mask

                inter = (prob * gt_mask).sum(dim=(1, 2))
                union = prob.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
                dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)
            else:
                dice = torch.zeros_like(bce)

            total_loss += (self.bce_weight * bce.mean() + self.dice_weight * dice.mean())
            total_inst += 1

        if total_inst > 0:
            total_loss = total_loss / total_inst
        else:
            total_loss = prototype_mask.sum() * 0.0

        return self.loss_seg_weight * total_loss