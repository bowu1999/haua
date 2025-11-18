import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..models.utils import centers_of_boxes, pairwise_iou, bbox_iou


def atss_assign(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    num_candidates: int = 9,
    topk: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ATSS label assignment (per image).
    Args:
        anchors: (N,4) anchor boxes / grid boxes in x1,y1,x2,y2
        gt_boxes: (M,4) ground-truth boxes in same coords
        num_candidates: number of nearest anchors per GT to consider (default 9)
        topk: alias for num_candidates (kept for clarity)
    Returns:
        assigned_gt_inds: (N,) long tensor, -1 means negative/background, >=0 is index of assigned gt in gt_boxes
        assigned_ious: (N,) float tensor, IoU of assigned gt (0 if negative)
    Notes:
        This implements the core ATSS idea:
         - For each GT, select K nearest anchors (by center distance)
         - Compute IoUs of those K anchors to this GT, take mean+std as threshold
         - Anchors whose IoU >= threshold and whose center is inside GT are positives for that GT.
         - If multiple GTs assign same anchor, choose GT with highest IoU.
    """
    if gt_boxes.numel() == 0:
        N = anchors.size(0)
        return (
            torch.full((N,),-1, dtype=torch.long, device=anchors.device),
            torch.zeros(N, device=anchors.device))

    N = anchors.size(0)
    M = gt_boxes.size(0)
    # centers
    anc_centers = centers_of_boxes(anchors)  # (N,2)
    gt_centers = centers_of_boxes(gt_boxes)  # (M,2)

    # pairwise center distance
    dist = torch.cdist(gt_centers, anc_centers)  # (M,N)
    # for each gt, choose k nearest anchors
    k = num_candidates if topk is None else topk
    k = min(k, N)
    topk_ids = dist.topk(k, largest=False, dim=1).indices  # (M,k)

    # compute IoU matrix (N,M) or (M,N) -> keep (N,M) for convenience
    ious = pairwise_iou(anchors, gt_boxes)  # (N,M)
    ious_t = ious.T  # (M,N)

    # threshold per GT: mean + std of top-k ious
    candidate_ious = torch.gather(ious_t, 1, topk_ids)  # (M,k)
    mean_per_gt = candidate_ious.mean(1)
    std_per_gt = candidate_ious.std(1)
    thr = mean_per_gt + std_per_gt  # (M,)

    # for each GT mark anchors whose IoU >= thr and whose center is inside GT box
    # center in gt:
    anc_x = anc_centers[:,0]
    anc_y = anc_centers[:,1]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]  # (M,)

    # expand to (M,N)
    anc_x_expand = anc_x[None, :].repeat(M,1)
    anc_y_expand = anc_y[None, :].repeat(M,1)
    inside_gt = (anc_x_expand >= gt_x1[:,None]) & (anc_x_expand <= gt_x2[:,None]) & \
                (anc_y_expand >= gt_y1[:,None]) & (anc_y_expand <= gt_y2[:,None])

    ious_mask = (ious_t >= thr[:,None]) & inside_gt  # (M,N)
    # assign: for each anchor, pick gt with highest IoU among candidates, else -1
    # convert back to (N,M) for easier gather
    ious_NM = ious  # (N,M)
    # for anchors that have any True in ious_mask[:,n], select the gt with max iou
    matched_gt_inds = torch.full((N,), -1, dtype=torch.long, device=anchors.device)
    matched_ious = torch.zeros((N,), dtype=anchors.dtype, device=anchors.device)

    for m in range(M):
        mask_m = ious_mask[m]  # (N,)
        if mask_m.any():
            # candidate anchors for this gt
            candidate_idxs = mask_m.nonzero(as_tuple=False).squeeze(1)
            # for those anchors, try to set gt index if IoU larger than previous assigned
            cand_ious = ious_NM[candidate_idxs, m]  # IoUs
            # compare with existing assigned IoUs
            prev_ious = matched_ious[candidate_idxs]
            update_mask = cand_ious > prev_ious
            if update_mask.any():
                to_update = candidate_idxs[update_mask]
                matched_gt_inds[to_update] = m
                matched_ious[to_update] = cand_ious[update_mask]

    return matched_gt_inds, matched_ious


def simota_assign(
    anchors: torch.Tensor,
    pred_cls_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: Optional[torch.Tensor] = None,
    center_radius: float = 2.5,
    topk: int = 10,
    cls_weight: float = 1.0,
    iou_weight: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SimOTA-like assigner (per image).
    Args:
        anchors: (N,4) candidate boxes (x1,y1,x2,y2) or grid centers but boxes expected.
        pred_cls_logits: (N,C) predicted class logits for each anchor (before sigmoid)
        gt_boxes: (M,4) gt boxes
        gt_labels: (M,) long labels (optional, used for computing classification cost); if None classification cost ignored
        center_radius: radius (in units of anchor stride) to select candidate anchors per GT. If anchors are arbitrary, code uses center distance normalized by average stride; if uncertain, keep it large.
        topk: number of top candidate anchors by center distance to consider per GT
        cls_weight, iou_weight: weights for cost terms
    Returns:
        matched_gt_inds: (N,) long tensor, -1 background, >=0 gt index
        matched_ious: (N,) float tensor of IoU with matched gt or 0
    Notes:
        This is a simplified implementation of SimOTA/OTA dynamic-k matching.
    """
    device = anchors.device
    N = anchors.size(0)
    M = gt_boxes.size(0)
    if M == 0:
        return torch.full((N,), -1, dtype=torch.long, device=device), torch.zeros((N,), device=device)

    # 1) compute pairwise iou (N,M)
    ious = pairwise_iou(anchors, gt_boxes)  # (N,M)

    # 2) compute center distances and pre-select candidates per GT
    anc_centers = centers_of_boxes(anchors)  # (N,2)
    gt_centers = centers_of_boxes(gt_boxes)  # (M,2)
    center_dist = torch.cdist(gt_centers, anc_centers)  # (M,N)
    k = min(topk, N)
    candidate_idxs = center_dist.topk(k, largest=False, dim=1).indices  # (M,k)

    # 3) compute cost matrix for candidates: classification cost + iou cost
    # classification cost: if gt_labels provided, use BCE between pred prob and one-hot, else ignore cls cost
    # convert logits -> prob (sigmoid)
    cls_cost = torch.zeros((N, M), device=device)
    if gt_labels is not None and pred_cls_logits is not None:
        # pred_cls_logits: (N,C)
        pred_prob = pred_cls_logits.sigmoid()  # (N,C)
        # for each gt m, classification cost = -pred_prob[:,gt_label] (higher prob smaller cost)
        for m in range(M):
            label = int(gt_labels[m].item())
            # negative log-likelihood like cost: use -log(p) ~ but we can use -p to keep simple and consistent
            cls_cost[candidate_idxs[m], m] = -pred_prob[candidate_idxs[m], label]
    # iou cost: -log(iou)
    iou_cost = -torch.log(ious.clamp(min=1e-7))  # (N,M)

    # combine cost only for candidate positions; others stay large
    INF = 1e9
    cost = torch.full((N, M), INF, device=device)
    for m in range(M):
        idxs = candidate_idxs[m]  # (k,)
        cost[idxs, m] = cls_weight * cls_cost[idxs, m] + iou_weight * iou_cost[idxs, m]

    # 4) dynamic K matching (greedy)
    matched_gt_inds = torch.full((N,), -1, dtype=torch.long, device=device)
    matched_ious = torch.zeros((N,), dtype=anchors.dtype, device=device)

    # for each gt, select dynamic top anchors
    # step: for each gt, determine dynamic_k = max(1, int(sum(top_ious)))
    topk_ious, _ = ious[candidate_idxs, torch.arange(M)[:,None]].topk(k=min(10, k), dim=1, largest=True) if k>0 else (torch.zeros((M,0), device=device), None)
    dynamic_ks = (topk_ious.sum(1).int().clamp(min=1)).tolist()  # list of M ints

    # now for each gt, select dynamic_k anchors with smallest cost
    # but need to resolve conflicts: anchors assigned multiple times -> keep lowest cost (or highest iou)
    for m in range(M):
        idxs = candidate_idxs[m]  # (k,)
        if idxs.numel() == 0:
            continue
        c = cost[idxs, m]  # (k,)
        k_m = dynamic_ks[m]
        k_m = min(k_m, idxs.numel())
        _, topk_idx_in_c = torch.topk(-c, k=k_m, largest=True)  # smallest cost -> largest -cost
        chosen = idxs[topk_idx_in_c]  # anchor indices chosen for this gt

        # assign: if anchor already assigned, compare cost and update if this gt gives lower cost
        for a in chosen:
            a = int(a.item())
            prev = matched_gt_inds[a].item()
            if prev == -1:
                matched_gt_inds[a] = m
                matched_ious[a] = ious[a, m]
            else:
                # conflict: keep gt giving higher IoU (or lower cost)
                if ious[a, m] > matched_ious[a]:
                    matched_gt_inds[a] = m
                    matched_ious[a] = ious[a, m]

    return matched_gt_inds, matched_ious


def tal_assign(
    pred_scores: torch.Tensor,
    pred_bboxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    topk: int = 10,
    cls_power: float = 1.0,
    iou_power: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    检测器的任务批处理对齐标签分配 (TAL)

    Args:
        pred_scores: (B, N, C) 预测类别logits或sigmoid前得分
        pred_bboxes: (B, N, 4) 预测框位于 xyxy 坐标系中（与 gt_ 框的坐标系相同）
        gt_bboxes:   (B, G, 4) 真实框（xyxy）
        gt_labels:   (B, G) 真实类别索引（long）
        topk:        根据对齐指标，每个 GT 选择候选锚点的数量
        cls_power:   应用于对齐度量中分类得分的指数（默认值为 1.0）
        iou_power:   应用于 align 指标中的 IoU 的指数（默认值为 2.0，与 TAL 论文实践相符）

    Returns:
        target_scores: (B, N, C) 软独热目标分数（0 或按对齐分数加权）
        target_bboxes: (B, N, 4) 为正样本锚点分配 GT bbox（为负样本锚点分配零）
        fg_mask:       (B, N) 布尔掩码指示正样本
        matched_gt_inds:(B, N) long tensor，-1 表示背景，否则表示分配的 GT 在 0..G-1 范围内的索引

    Notes:
        - This implementation follows the typical TAL flow:
            1. compute IoU(pred_bbox, gt_bbox) -> (N, G)
            2. compute cls_score (N, G) by picking predicted prob for each gt label
            3. align_metric = (cls_score ** cls_power) * (iou ** iou_power)
            4. for each GT, select top-k anchors by align_metric (these are candidates)
            5. anchors that are selected by any GT become positives (if conflict, we choose the GT
               that gives maximum align_metric / IoU)
        - We return targets suitable to be plugged into your loss computation.
    """
    device = pred_scores.device
    B, N, C = pred_scores.shape
    _, _, _ = pred_bboxes.shape  # (B,N,4)
    _, G = gt_bboxes.shape[0], gt_bboxes.shape[1] if gt_bboxes.dim() == 3 else (0,0)

    # outputs
    target_scores = torch.zeros_like(pred_scores, device=device)  # (B,N,C)
    target_bboxes = torch.zeros_like(pred_bboxes, device=device)  # (B,N,4)
    fg_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    matched_gt_inds = torch.full((B, N), -1, dtype=torch.long, device=device)

    # process batch element-wise
    for b in range(B):
        gt_b = gt_bboxes[b]        # (G,4)
        gt_l = gt_labels[b]        # (G,)
        if gt_b.numel() == 0:
            continue

        pb = pred_bboxes[b]        # (N,4)
        ps = pred_scores[b]        # (N,C)
        # 1) IoU matrix (N, G) -- use bbox_iou (broadcast-friendly)
        ious = bbox_iou(pb, gt_b)  # (N, G)

        # 2) classification score per (N, G): take predicted probability for each GT label
        #    apply sigmoid to logits to get probability in [0,1]
        prob = ps.sigmoid()        # (N, C)
        # 构建 (G, C) 独热编码，用于 gt 标签，因此概率 @ one_hot.T -> (N,G) 选择标签概率
        gt_one_hot = F.one_hot(gt_l.long(), num_classes=C).float()  # (G, C)
        cls_score = prob @ gt_one_hot.T                            # (N, G)

        # 3) 度量矩阵
        align_metric = (cls_score.clamp(min=1e-8) ** cls_power) * (ious.clamp(min=1e-8) ** iou_power)  # (N, G)

        # 4) 对于每个 GT，根据 align_metric 选择前 k 个锚框
        k = min(topk, N)
        # topk over dim=0 gives topk anchors for each GT -> returns (k, G) values/indices
        topk_vals, topk_idx = align_metric.topk(k=k, dim=0, largest=True)  # topk_idx: (k, G)
        # 构建 bool 掩码 (N, G)，标记每个 GT 的候选锚框。
        candidate_mask = torch.zeros_like(align_metric, dtype=torch.bool)  # (N,G)
        # [None, :] 是增加维度操作，将原本的 1D 张量 (K,) 变成 2D 张量 (1, K)，方便与 topk_idx 的形状 (N, K) 广播匹配。
        candidate_mask[topk_idx, torch.arange(topk_idx.size(1), device=device)[None, :]] = True

        # 5) determine final positive anchors:
        # anchors that are candidate for any GT are positives (but may conflict)
        # 主要用来标记正样本，即与任何 GT 有重叠的锚点
        pos_mask_any = candidate_mask.any(dim=1)  # (N,)
        fg_mask[b] = pos_mask_any

        if pos_mask_any.sum() == 0:
            continue

        # 解决冲突：对于每个锚点，选择 align_metric 值最大的 GT
        # 将非候选位置的 align_metric 值设置为 -inf，以便 argmax 仅在候选位置中选择
        am = align_metric.clone()
        am[~candidate_mask] = -1e9  # ignore non-candidates
        # matched_gt for each anchor: argmax over G -> value in [0,G-1]
        matched = am.argmax(dim=1)  # (N,)
        matched_scores = am.max(dim=1)[0]  # chosen align metric per anchor

        # 那些并非候选锚点的匹配分数将为 -1e9；我们必须将它们屏蔽掉。
        valid_pos = matched_scores > -1e9

        # set outputs for valid positives
        pos_idxs = valid_pos.nonzero(as_tuple=False).squeeze(1)
        if pos_idxs.numel() > 0:
            # 填充匹配的 gt 索引
            matched_gt_inds[b, pos_idxs] = matched[pos_idxs].long()
            # 分配边界框和类别目标
            assigned_gt_idx = matched[pos_idxs].long()  # GT 下标
            target_bboxes[b, pos_idxs] = gt_b[assigned_gt_idx]  # (P,4)
            # 独热编码分类（硬独热编码，按对齐指标加权）
            # 构建独热编码并乘以对齐分数作为软目标（可与 BCE 损失一起使用）
            assigned_labels = gt_l[assigned_gt_idx]  # (P,)
            # 将指定标签的独热码设置为 1.0
            target_scores[b, pos_idxs, assigned_labels] = 1.0
            # 每个锚点的权重（可选择对每个锚点的重量进行标准化）
            weights = matched_scores[pos_idxs].clamp(min=0)
            # Normalize weights to [0,1] by dividing by max (avoid division by zero)
            if weights.numel() > 0:
                wmax = weights.max()
                if wmax > 0:
                    weights = weights / (wmax + 1e-12)
            # apply weights to target_scores for these anchors
            target_scores[b, pos_idxs] = target_scores[b, pos_idxs] * weights.unsqueeze(-1)

    return target_scores, target_bboxes, fg_mask, matched_gt_inds