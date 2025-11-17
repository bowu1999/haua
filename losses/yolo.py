import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


from .cls import ClassificationLoss
from .dfl import dfl_loss
from .iou import ciou_loss
from .assigner import tal_assign


class YoloV10Loss(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 15,
        cls_loss_type: str = 'bce',
        cls_loss_weight: float = 1.0,
        dfl_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            num_classes: number of classes (default 80)
            reg_max: reg_max for DFL (default 15 -> 16 bins)
            cls_loss_type: 'bce' or 'ce'
            *_weight: scalar weights for loss terms
        """
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_loss = ClassificationLoss(loss_type=cls_loss_type, num_classes=num_classes)
        self.cls_w = cls_loss_weight
        self.dfl_w = dfl_loss_weight
        self.iou_w = iou_loss_weight
        self.device = device

    def _ensure_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept tensors shaped (B,N,C) or (B,C,H,W) -> return (B,N,C)
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()  # (B, N, C)
        elif x.dim() == 3:
            # assume already (B,N,C)
            pass
        else:
            raise ValueError(f"Unsupported pred tensor shape {x.shape}")
        return x

    def _decode_preds_from_dfl(
        self,
        pred_dist: torch.Tensor,
        anchors_centers: torch.Tensor,
        stride: int
    ) -> torch.Tensor:
        """
        Decode boxes from DFL distributions.
        Args:
            pred_dist: (B, N, 4*(reg_max+1)) logits
            anchors_centers: (N,2) centers in pixel coords (cx,cy)
            stride: scalar stride
        Returns:
            boxes_xyxy: (B, N, 4) in pixel coords (x1,y1,x2,y2)
        """
        B, N, D = pred_dist.shape
        reg_len = self.reg_max + 1
        assert D == 4 * reg_len, "pred_dist channel mismatch"

        # softmax over bins then expectation -> offset in units of "pixels/stride"
        p = pred_dist.view(B * N * 4, reg_len)  # [B*N*4, reg_len]
        prob = F.softmax(p, dim=-1)
        values = torch.arange(reg_len, device=prob.device, dtype=prob.dtype).unsqueeze(0)  # [1, reg_len]
        exp = (prob * values).sum(dim=1)  # [B*N*4]
        exp = exp.view(B, N, 4)  # offsets in [0, reg_max] (in units of stride)
        # Convert to pixel offsets: multiply by stride
        offsets_px = exp * float(stride)

        # offsets ordering assumption: [l, t, r, b] (common)
        l = offsets_px[..., 0]
        t = offsets_px[..., 1]
        r = offsets_px[..., 2]
        b = offsets_px[..., 3]

        # anchors_centers: (N,2) -> (1,N,2) broadcast to B
        centers = anchors_centers.to(pred_dist.device).unsqueeze(0).expand(B, -1, -1)
        cx = centers[..., 0]
        cy = centers[..., 1]

        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B,N,4)
        return boxes

    def forward(
        self,
        preds: dict,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        anchors_centers: Optional[List[torch.Tensor]] = None,
        strides: Optional[List[int]] = None,
        decoded_pred_boxes: Optional[torch.Tensor] = None,
        topk: int = 10,
    ) -> dict:
        """
        Args:
            preds: dict with keys 'one2one' and 'one2many', each is list of 3 tensors (per-scale)
                   each tensor shape either (B,N,144) or (B,144,H,W)
            gt_bboxes: (B, G, 4) in pixel coords (x1,y1,x2,y2)
            gt_labels: (B, G) long
            anchors_centers: list of 3 tensors [(N1,2),(N2,2),(N3,2)] centers in pixel coords
            strides: list of 3 ints e.g. [8,16,32]
            decoded_pred_boxes: optional pre-decoded (B, total_N, 4); if provided, anchors/strides not needed
            topk: TAL topk
        Returns:
            dict with losses {'cls':..., 'dfl':..., 'ciou':..., 'total': ...} and extras
        """
        device = gt_bboxes.device if self.device is None else self.device

        # 1) collect predictions from both 'one2one' and 'one2many' and 3 scales each
        all_levels = []
        if 'one2one' not in preds or 'one2many' not in preds:
            raise ValueError("preds must contain 'one2one' and 'one2many' keys with 3-scale lists each.")

        for key in ['one2one', 'one2many']:
            plist = preds[key]
            if len(plist) != 3:
                raise ValueError(f"{key} must contain 3 scale tensors.")
            for lvl_pred in plist:
                lvl_pred = self._ensure_flat(lvl_pred)  # (B,N_lvl,144)
                all_levels.append(lvl_pred)
        # now all_levels is list of 6 tensors
        # concatenate along N dim
        # verify batch-consistent
        B = all_levels[0].shape[0]
        device = all_levels[0].device

        preds_cat = torch.cat(all_levels, dim=1)  # (B, total_N, 144)
        B, total_N, C = preds_cat.shape
        assert C == 144, "expected 144 channels per position"

        # split dist and cls
        reg_len = self.reg_max + 1
        dfl_ch = 4 * reg_len  # 64
        pred_dist = preds_cat[:, :, :dfl_ch].contiguous()  # (B, total_N, 4*(reg_max+1))
        pred_cls_logits = preds_cat[:, :, dfl_ch:].contiguous()  # (B, total_N, num_classes)
        # if channels for classes < self.num_classes, handle gracefully
        if pred_cls_logits.shape[-1] != self.num_classes:
            # allow case where model produced different class dims (but warn)
            # we will expand/truncate accordingly
            if pred_cls_logits.shape[-1] < self.num_classes:
                # pad zeros
                pad = torch.zeros(B, total_N, self.num_classes - pred_cls_logits.shape[-1], device=device)
                pred_cls_logits = torch.cat([pred_cls_logits, pad], dim=-1)
            else:
                pred_cls_logits = pred_cls_logits[:, :, :self.num_classes]

        # 2) obtain pred_bboxes either from decoded_pred_boxes or decode from DFL + anchors/strides per level
        if decoded_pred_boxes is not None:
            if decoded_pred_boxes.shape[1] != total_N:
                raise ValueError("decoded_pred_boxes has mismatched total_N")
            pred_bboxes = decoded_pred_boxes.to(device)
        else:
            # require anchors_centers and strides
            if anchors_centers is None or strides is None:
                raise ValueError("Either decoded_pred_boxes or anchors_centers+strides must be provided for bbox decoding.")
            # anchors_centers: list of 6? -> because we concatenated 6 levels
            # Input anchors_centers should match the order above: one2one scale1, scale2, scale3, one2many scale1, scale2, scale3
            # We'll flatten them
            flat_centers = []
            for ac in anchors_centers:
                # ac is (N_i,2)
                if ac.dim() != 2 or ac.shape[1] != 2:
                    raise ValueError("anchors_centers elements must be (N_i,2)")
                flat_centers.append(ac.to(device))
            # If user passed 3 anchors for each key (length 3), but we concatenated 6 levels (two keys * 3),
            # we expect anchors_centers to be length 6. If they provided only 3, we assume it's shared for one2one/one2many,
            # so we duplicate.
            if len(flat_centers) == 3:
                flat_centers = flat_centers + flat_centers  # duplicate for both keys
            if len(flat_centers) != 6:
                raise ValueError("anchors_centers must be length 6 (or 3 to be duplicated).")
            # same for strides
            if len(strides) == 3:
                strides = strides + strides
            if len(strides) != 6:
                raise ValueError("strides must be length 6 (or 3 to be duplicated).")

            # concat centers into (total_N,2)
            centers_cat = torch.cat(flat_centers, dim=0)  # (total_N,2)
            # We will decode using per-level stride: need strides per position -> expand
            # Build per-position stride array
            stride_list_per_pos = []
            for s, ac in zip(strides, flat_centers):
                stride_list_per_pos.append(torch.full((ac.shape[0],), float(s), device=device))
            stride_per_pos = torch.cat(stride_list_per_pos, dim=0)  # (total_N,)
            # WARNING: our decode function expects a single scalar stride; DFL offsets are in units of stride.
            # But stride differs per position; easiest is to decode per-level and then concat.
            dl = []
            idx = 0
            for i, ac in enumerate(flat_centers):
                Ni = ac.shape[0]
                pd_lvl = pred_dist[:, idx:idx+Ni, :]  # (B, Ni, D)
                boxes_lvl = self._decode_preds_from_dfl(pd_lvl, ac, stride_per_pos[idx].item())
                dl.append(boxes_lvl)
                idx += Ni
            pred_bboxes = torch.cat(dl, dim=1)  # (B, total_N, 4)

        # 3) TAL assignment
        # tal_assign expects pred_scores (B,N,C) and pred_bboxes (B,N,4)
        target_scores, target_bboxes, fg_mask, matched_gt_inds = tal_assign(
            pred_scores = pred_cls_logits,
            pred_bboxes = pred_bboxes,
            gt_bboxes = gt_bboxes,
            gt_labels = gt_labels,
            topk = topk
        )
        # target_scores: (B,N,C) soft one-hot (0 for negatives)
        # target_bboxes: (B,N,4)
        # fg_mask: (B,N) bool mask for positives

        # 4) Classification loss: BCEWithLogits over all positions using target_scores
        pred_cls_flat = pred_cls_logits.view(-1, self.num_classes)
        targ_cls_flat = target_scores.view(-1, self.num_classes)
        cls_loss_val = self.cls_loss(pred_cls_flat, targ_cls_flat)  # already averages internally
        # multiply by weight
        cls_loss_val = cls_loss_val * self.cls_w

        # 5) Regression losses: DFL on positive anchors & CIoU on boxes
        # gather positive indices
        fg_mask_flat = fg_mask.view(B * total_N)
        num_pos = int(fg_mask_flat.sum().item())
        # prepare outputs
        if num_pos > 0:
            # Gather pred_dist for positives: shape (P, 4*(reg_max+1))
            pred_dist_pos = pred_dist.view(B * total_N, -1)[fg_mask_flat]  # (P, D)
            # Need target offsets in units of [0, reg_max]
            # For each positive anchor we need distances l,t,r,b relative to center normalized by stride
            # So we must retrieve anchor centers and strides per position
            # Build anchors centers per pos like before:
            # If decoded_pred_boxes provided, we may not have anchors => attempt to compute offsets from pred_bboxes vs target_bboxes
            # Here we'll compute offsets relative to anchor centers if available; else compute continuous distances and divide by stride_per_pos.
            # Build centers_cat used earlier (if present)
            if decoded_pred_boxes is None:
                centers_cat = centers_cat.to(device)  # (total_N,2)
                # expand centers per batch
                centers_exp = centers_cat.unsqueeze(0).expand(B, -1, -1)  # (B, total_N,2)
                centers_flat = centers_exp.reshape(B * total_N, 2)
            else:
                # If no anchors available, approximate center as pred_box center
                pred_centers = ((pred_bboxes[...,0] + pred_bboxes[...,2]) / 2,
                                (pred_bboxes[...,1] + pred_bboxes[...,3]) / 2)
                centers_flat = torch.stack(pred_centers, dim=-1).reshape(B * total_N, 2)

            # stride per pos built earlier if available; else assume uniform stride = 1
            if 'stride_per_pos' in locals():
                stride_per_pos_flat = stride_per_pos.repeat(B) if stride_per_pos.dim() == 1 else stride_per_pos
                # stride_per_pos currently (total_N,), convert to (B*total_N,)
                stride_per_pos_flat = stride_per_pos_flat.to(device).repeat(B)
            else:
                stride_per_pos_flat = torch.ones(B * total_N, device=device)

            # target boxes flatten
            targ_boxes_flat = target_bboxes.view(B * total_N, 4)
            pos_targ_boxes = targ_boxes_flat[fg_mask_flat]  # (P,4)
            # centers for positives
            pos_centers = centers_flat[fg_mask_flat]  # (P,2)
            pos_strides = stride_per_pos_flat[fg_mask_flat].to(device)  # (P,)

            # compute l,t,r,b in pixels
            tx1 = pos_targ_boxes[:, 0]
            ty1 = pos_targ_boxes[:, 1]
            tx2 = pos_targ_boxes[:, 2]
            ty2 = pos_targ_boxes[:, 3]
            cx = pos_centers[:, 0]
            cy = pos_centers[:, 1]
            l = (cx - tx1).clamp(min=0)
            t = (cy - ty1).clamp(min=0)
            r = (tx2 - cx).clamp(min=0)
            b = (ty2 - cy).clamp(min=0)
            # normalize by stride to get in units corresponding to reg_max
            # avoid division by zero
            pos_strides = pos_strides.clamp(min=1e-6)
            l_u = l / pos_strides
            t_u = t / pos_strides
            r_u = r / pos_strides
            b_u = b / pos_strides
            target_offsets = torch.stack([l_u, t_u, r_u, b_u], dim=1)  # (P,4)
            # clamp to [0, reg_max]
            target_offsets = target_offsets.clamp(min=0.0, max=float(self.reg_max))

            # DFL
            dfl_loss_val = dfl_loss(pred_dist_pos, target_offsets, reg_max=self.reg_max)
            dfl_loss_val = dfl_loss_val * self.dfl_w

            # CIoU: compute on predicted boxes and gt boxes for positives
            pred_bboxes_flat = pred_bboxes.view(B * total_N, 4)
            pos_pred_boxes = pred_bboxes_flat[fg_mask_flat]
            ciou_loss_val = ciou_loss(pos_pred_boxes, pos_targ_boxes)
            ciou_loss_val = ciou_loss_val * self.iou_w
        else:
            dfl_loss_val = torch.tensor(0.0, device=device, requires_grad=True)
            ciou_loss_val = torch.tensor(0.0, device=device, requires_grad=True)

        # 6) total
        total_loss = cls_loss_val + dfl_loss_val + ciou_loss_val

        # prepare return dict
        ret = {
            'total': total_loss,
            'cls': cls_loss_val,
            'dfl': dfl_loss_val,
            'ciou': ciou_loss_val,
            'num_pos': num_pos,
            'matched_gt_inds': matched_gt_inds}

        return ret
