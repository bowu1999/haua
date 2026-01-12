import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class InstanceSegLoss(nn.Module):
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        num_protos: int = 32,
        proto_size: Tuple[int, int] = (160, 160),
        loss_seg_weight: float = 1.0,
        use_dice: bool = True,
    ):
        super().__init__()
        self.strides = strides
        self.num_protos = num_protos
        self.proto_h, self.proto_w = proto_size
        self.loss_seg_weight = loss_seg_weight
        self.use_dice = use_dice

    def dice_loss(self, pred, target, eps=1e-6):
        """
        pred: (P, H, W) sigmoid 后
        target: (P, H, W) in {0,1}
        """
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + eps) / (union + eps)

        return 1.0 - dice.mean()

    def forward(
        self,
        seg_out: Tuple[torch.Tensor, ...],       # tuple len=3, each (B, C, Hs, Ws)
        prototype_mask: torch.Tensor,           # (B, C, Hp, Wp)
        fg_mask: torch.Tensor,                  # (B, N_tot) bool
        matched_gt_inds: torch.Tensor,          # (B, N_tot) long, -1 for neg
        gt_bboxes: Union[List[torch.Tensor], torch.Tensor],
        gt_masks: List[torch.Tensor],           # list of length B, each (Gi, H_img, W_img)
        img_shape: Tuple[int, int],             # (img_h, img_w)
        grids: List[torch.Tensor],              # 与检测中 make_grid 对应的各尺度 (N_i, 2)
    ):
        """
        返回一个标量 seg_loss
        """
        device = prototype_mask.device
        B = prototype_mask.shape[0]
        img_h, img_w = img_shape
        # 1. 把 seg_out 按尺度拉平，得到 (B, N_tot, C)
        per_scale_coeffs = []
        Ns = []
        for i, p in enumerate(seg_out):
            Bp, Cc, N = p.shape
            Ns.append(N)
            coeff = p.permute(0, 2, 1).reshape(Bp, N, Cc)  # (B, N, C)
            per_scale_coeffs.append(coeff)
        all_coeffs = torch.cat(per_scale_coeffs, dim=1)        # (B, N_tot, C)
        # 2. 取正样本 anchor 的 coeff
        pos_mask = fg_mask.bool()  # (B, N_tot)
        num_pos = pos_mask.sum()
        if num_pos == 0:
            return prototype_mask.new_tensor(0.0)
        # (P, C)
        pos_coeffs = all_coeffs[pos_mask]
        # 3. 准备对应的 GT mask （P, Hp, Wp）
        #    - 对于每个 batch 中的每个正样本，根据 matched_gt_inds 找到 gt index
        #    - 将原图尺寸的 gt_mask resize 到 proto_size
        Hp, Wp = self.proto_h, self.proto_w
        all_gt_instance_masks = []

        is_list_boxes = isinstance(gt_bboxes, (list, tuple))

        # 为了能按 pos_mask 展平，我们先按 B 遍历，把每个 batch 内正样本的 gt mask 收集
        for b in range(B):
            m = matched_gt_inds[b]          # (N_tot,)
            pos_idx = (m >= 0) & pos_mask[b]  # 这个 batch 的正样本位置
            if pos_idx.sum() == 0:
                continue
            assigned = m[pos_idx].long()    # (P_b,)
            # 当前 batch 的 gt masks: (Gi, H_img, W_img)
            gt_masks_b = gt_masks[b].to(device)  # 可能是 cpu tensor，放到 device 上
            # 如果没有 mask，跳过
            if gt_masks_b.numel() == 0:
                continue

            # 逐个 gt inst 取出对应 mask，resize 到 proto_size
            for gi in assigned:
                mask_i = gt_masks_b[gi]  # (H_img, W_img)
                # [1,1,H,W] -> [1,1,Hp,Wp] 再 squeeze
                mask_i = mask_i.unsqueeze(0).unsqueeze(0).float()
                mask_i = F.interpolate(mask_i, size=(Hp, Wp), mode="bilinear", align_corners=False)
                mask_i = mask_i.squeeze(0).squeeze(0)  # (Hp,Wp)
                # 二值化（如果需要）
                mask_i = (mask_i > 0.5).float()
                all_gt_instance_masks.append(mask_i)

        if len(all_gt_instance_masks) == 0:
            return prototype_mask.new_tensor(0.0)

        gt_inst_masks = torch.stack(all_gt_instance_masks, dim=0)  # (P, Hp, Wp)

        # 注意：上面收集的顺序与 pos_coeffs 的顺序要一致，
        # 简单的写法是只在一个地方 flatten，确保遍历顺序与 pos_mask 的展开方式相同。
        # 为严谨起见，可以直接用下面这种更对齐的写法（略复杂），
        # 这里给的是逻辑示例，你可根据自己 batch 结构微调索引。

        # 4. 利用 prototype + coeff 生成预测实例 mask
        # prototype_mask: (B, C, Hp, Wp)
        # 对应正样本所在 batch 的 proto，需要按 pos_mask 拿出来
        # 做法：先把 proto 按 batch 展平到一个列表，再按正样本所属 batch 取
        # 简化方案：假设 proto 在一个 batch 内共享（即同一 batch 所有实例用同一个 prototype_mask[b]）

        # 先为每个正样本找它所属的 batch index
        batch_inds = []
        for b in range(B):
            nb = pos_mask[b].sum().item()
            if nb > 0:
                batch_inds.extend([b] * nb) # type: ignore
        batch_inds = torch.as_tensor(batch_inds, device=device, dtype=torch.long)  # (P,)

        # (P, C, Hp, Wp)
        proto_per_pos = prototype_mask[batch_inds]  # 按 batch 取 proto

        # pos_coeffs: (P, C)
        # proto_per_pos: (P, C, Hp, Wp)
        # 预测 mask: sum_k coeff_k * proto_k
        P, C = pos_coeffs.shape
        coeffs = pos_coeffs.view(P, C, 1, 1)        # (P, C, 1, 1)
        pred_masks = (coeffs * proto_per_pos).sum(dim=1)  # (P, Hp, Wp)
        pred_masks = pred_masks.sigmoid()

        # 5. 计算 BCE + Dice (可选)
        bce = F.binary_cross_entropy(pred_masks, gt_inst_masks, reduction='mean')
        if self.use_dice:
            dice = self.dice_loss(pred_masks, gt_inst_masks)
            seg_loss = bce + dice
        else:
            seg_loss = bce

        return seg_loss * self.loss_seg_weight