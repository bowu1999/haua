import torch
import torch.nn.functional as F


def make_grid(h: int, w: int, stride: int = 1, device=None, dtype=torch.float32):
    """
    创建 feature map 的格点中心坐标 grid
    Args:
        h, w (int):
            特征图的高度与宽度

        stride (int):
            特征图相对于原图的下采样倍数（一个格子对应多少像素）

        device, dtype:
            返回 Tensor 的设备与数据类型

    Returns:
        grid (Tensor):
            形状 (H, W, 2) 的网格，每个位置为 (cx, cy)
    """
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1)

    return grid



def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points and stride tensors. (anchor-free style)
    feats: list of feature maps [(B, C, H, W), ...]
    strides: list of stride values
    grid_cell_offset: usually 0.5 for center offset
    """
    anchor_points = []
    stride_tensors = []

    for i, (feat, stride) in enumerate(zip(feats, strides)):
        _, _, h, w = feat.shape

        # grid_x shape: (h*w,)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=feat.device),
            torch.arange(w, device=feat.device),
            indexing="ij"
        )

        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        grid = (grid + grid_cell_offset)  # add 0.5 (center of cell)

        anchor_points.append(grid)  # in feature map scale
        stride_tensors.append(torch.full((h * w, 1), stride, device=feat.device))

    # concat multilevel
    anchor_points = torch.cat(anchor_points, dim=0).float()
    stride_tensors = torch.cat(stride_tensors, dim=0).float()

    return anchor_points, stride_tensors


def dfl2dist(pred_reg: torch.Tensor, reg_max: int = 16, apply_softmax: bool = True):
    """
    将 DFL（Distribution Focal Loss）风格的回归输出转换为连续距离值（l, t, r, b）
    Args:
        pred_reg (Tensor):回归预测值，其中 4 对应 l, t, r, b 四个边距离的分布 logits，形状可为:
                - (B, 4*reg_max, H, W)
                - (B, 4, reg_max, H, W)
        reg_max (int):DFL 的 bins 数（每个边的离散分布长度）。默认 16
        apply_softmax (bool):
            是否对 logits 做 softmax 以获得概率分布，一般为 True，除非你的 logits 已提前 softmax

    Returns:
        dist (Tensor):
            连续的边界偏移量，形状 (B, 4, H, W)，对应 l,t,r,b
            单位仍是“bins 单位”，后续需要乘 stride 得到像素量
    """
    if pred_reg.dim() == 4 and pred_reg.size(1) == 4 * reg_max:
        B, C, H, W = pred_reg.shape
        pred = pred_reg.view(B, 4, reg_max, H, W)
    elif pred_reg.dim() == 5 and pred_reg.size(1) == 4 and pred_reg.size(2) == reg_max:
        pred = pred_reg
    else:
        raise ValueError("预测的形状必须为 (B,4*reg_max,H,W) 或 (B,4,reg_max,H,W)")
    if apply_softmax:
        prob = F.softmax(pred, dim=2)
    else:
        prob = F.softmax(pred, dim=2)
    device = pred.device
    project = torch.arange(reg_max, dtype=prob.dtype, device=device).view(1, 1, reg_max, 1, 1)
    dist = (prob * project).sum(dim=2)  # (B,4,H,W)

    return dist


def decode_dfl(
    pred_reg: torch.Tensor,
    reg_max: int = 16,
    stride: int = 1,
    apply_softmax: bool = True
):
    """
    YOLOv8/YOLO10/YOLO11 风格的 DFL 边界框解码函数
    Args:
        pred_reg (Tensor):边界框回归预测，形状可为：
            - (B, 4*reg_max, H, W)
            - (B, 4, reg_max, H, W)
        reg_max (int):DFL bins 数（每条边的离散概率长度）
        stride (int):当前特征图的 stride，用于将 bins 距离转换为像素单位
        apply_softmax (bool):是否对 logits 应用 softmax。一般为 True

    Returns:
        boxes (Tensor): 解码后的边界框位置，形状 (B, H*W, 4)，格式为 (x1, y1, x2, y2)
        dist_pixels (Tensor): 每个位置的像素级别的偏移量 (l, t, r, b)，形状为 (B,4,H,W)
            可用于 loss 分支等其他用途。
    """
    if pred_reg.dim() == 4:
        _, C, H, W = pred_reg.shape
    else:
        _, _, _, H, W = pred_reg.shape
    dist_bins = dfl2dist(pred_reg, reg_max=reg_max, apply_softmax=apply_softmax)  # (B,4,H,W)
    dist_pixels = dist_bins * float(stride)
    grid = make_grid(H, W, stride=stride, device=pred_reg.device, dtype=dist_pixels.dtype)  # (H,W,2)
    cx = grid[..., 0].unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
    cy = grid[..., 1].unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
    l = dist_pixels[:, 0:1, :, :]
    t = dist_pixels[:, 1:2, :, :]
    r = dist_pixels[:, 2:3, :, :]
    b = dist_pixels[:, 3:4, :, :]
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    boxes = torch.cat([x1, y1, x2, y2], dim=1)  # (B,4,H,W)
    boxes = boxes.permute(0, 2, 3, 1).reshape(pred_reg.shape[0], H * W, 4)

    return boxes, dist_pixels


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    将预测的距离编码（ltrb, left-top-right-bottom）转换为边界框坐标
    YOLOv8/YOLOv10 等模型使用 DFL（Distribution Focal Loss）时，
    回归分支预测的是每个点到边界的距离，而不是直接预测 bbox
    本函数用于把该“距离格式”解码为标准 bbox（xyxy 或 xywh）

    Args:
        distance (Tensor): 预测的距离值，格式为 (..., 4)，在 dim 维上排列为 [l, t, r, b]。
            - l = anchor_point_x - x1
            - t = anchor_point_y - y1
            - r = x2 - anchor_point_x
            - b = y2 - anchor_point_y
        anchor_points (Tensor): 锚点坐标 (..., 2)，通常来自特征图中心点。
        xywh (bool): 是否将最终结果转换为 (x_center, y_center, w, h) 格式。
            - True  -> 输出 xywh
            - False -> 输出 xyxy
        dim (int): 表示 bbox 维度所在的维度（默认最后一维）。

    Returns:
        Tensor: 解码后的 bbox。
            - xywh=True  -> (..., 4) 格式为 [cx, cy, w, h]
            - xywh=False -> (..., 4) 格式为 [x1, y1, x2, y2]
    """
    # 确保距离向量 dim 维长度为 4（必须是 l, t, r, b）
    assert distance.shape[dim] == 4
    # 将预测的 distance 分割成左上 (lt) 和右下 (rb) 两部分
    # lt = (l, t)，rb = (r, b)
    lt, rb = distance.split([2, 2], dim)
    # 根据锚点位置解码出左上角坐标：x1 = anchor_x - l, y1 = anchor_y - t
    x1y1 = anchor_points - lt
    # 解码出右下角坐标：x2 = anchor_x + r, y2 = anchor_y + b
    x2y2 = anchor_points + rb
    if xywh:
        # xywh 模式：计算中心坐标和宽高
        c_xy = (x1y1 + x2y2) / 2  # (cx, cy)
        wh = x2y2 - x1y1          # (w, h)
        return torch.cat((c_xy, wh), dim)  # 最终输出 [cx, cy, w, h]

    # 否则输出 xyxy
    return torch.cat((x1y1, x2y2), dim)  # 输出 [x1, y1, x2, y2]


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    将边界框从 (cx, cy, w, h) 转换为 (x1, y1, x2, y2) 格式。

    这是检测任务中常见的 bbox 转换方式：
        - (cx, cy) 表示 bbox 中心点坐标
        - (w, h)   表示 bbox 宽与高
        - (x1, y1) 表示左上角坐标
        - (x2, y2) 表示右下角坐标

    Args:
        boxes (Tensor): 输入张量，形状为 (..., 4)，格式为 (cx, cy, w, h)。

    Returns:
        Tensor: 输出张量，形状为 (..., 4)，格式为 (x1, y1, x2, y2)。
    """
    # 从最后一维拆分出 cx, cy, w, h 四个分量
    cx, cy, w, h = boxes.unbind(-1)

    # 左上角 (x1, y1) = (cx - w/2, cy - h/2)
    x1 = cx - w / 2
    y1 = cy - h / 2

    # 右下角 (x2, y2) = (cx + w/2, cy + h/2)
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 合并成 (..., 4)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2dist(
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    reg_max: int,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    将目标边界框 (xyxy) 转换为到锚点的左/上/右/下连续距离

    Args:
        anchor_points: (N,2) float (x,y)
        target_bboxes: (N,4) float (x1,y1,x2,y2)
        reg_max: 最大离散区间 (classes = reg_max + 1)。返回的目标区域被裁剪到 [0, reg_max - eps]。
    Returns:
        dist: (N,4) floats [l, t, r, b]
    """
    assert anchor_points.ndim == 2 and anchor_points.shape[-1] == 2
    assert target_bboxes.ndim == 2 and target_bboxes.shape[-1] == 4
    assert anchor_points.shape[0] == target_bboxes.shape[0], "anchor_points and target_bboxes must have same first dim"
    px = anchor_points[:, 0]
    py = anchor_points[:, 1]
    x1 = target_bboxes[:, 0]
    y1 = target_bboxes[:, 1]
    x2 = target_bboxes[:, 2]
    y2 = target_bboxes[:, 3]
    l = (px - x1).clamp(min=0)
    t = (py - y1).clamp(min=0)
    r = (x2 - px).clamp(min=0)
    b = (y2 - py).clamp(min=0)
    dist = torch.stack([l, t, r, b], dim=-1)
    # Clip to avoid hitting reg_max exactly (so that tl = floor(target) always < reg_max)
    max_val = float(reg_max) - eps
    dist = dist.clamp(min=0.0, max=max_val)

    return dist


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = False, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU or CIoU between box1 and box2.
    - box1: (N,4)
    - box2: (M,4) or (N,4)
    - if shapes match (N==M) -> returns (N,) (elementwise)
    - else -> returns (N,M) IoU matrix (CIoU not computed in pairwise case)
    - xywh: if True, inputs are (cx,cy,w,h)
    - CIoU: if True and shapes match, compute CIoU (per-element)
    """
    if xywh:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    # ensure last dim is 4
    assert box1.shape[-1] == 4 and box2.shape[-1] == 4

    N = box1.shape[0]
    M = box2.shape[0]

    # elementwise case (N == M) -> produce (N,)
    if N == M:
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h

        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0)) * ((box1[:, 3] - box1[:, 1]).clamp(min=0))
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0)) * ((box2[:, 3] - box2[:, 1]).clamp(min=0))
        union = area1 + area2 - inter
        iou = inter / union.clamp(min=eps)

        if not CIoU:
            return iou

        # --- CIoU terms (per-element) ---
        # center distance
        c_x1 = (box1[:, 0] + box1[:, 2]) / 2
        c_y1 = (box1[:, 1] + box1[:, 3]) / 2
        c_x2 = (box2[:, 0] + box2[:, 2]) / 2
        c_y2 = (box2[:, 1] + box2[:, 3]) / 2
        center_dist2 = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2

        # smallest enclosing box
        enc_x1 = torch.min(box1[:, 0], box2[:, 0])
        enc_y1 = torch.min(box1[:, 1], box2[:, 1])
        enc_x2 = torch.max(box1[:, 2], box2[:, 2])
        enc_y2 = torch.max(box1[:, 3], box2[:, 3])
        enc_w = (enc_x2 - enc_x1).clamp(min=0)
        enc_h = (enc_y2 - enc_y1).clamp(min=0)
        c2 = enc_w ** 2 + enc_h ** 2 + eps

        # aspect ratio term v and weighting factor alpha
        w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
        h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
        w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
        h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)

        # v measure
        atan1 = torch.atan(w1 / h1)
        atan2 = torch.atan(w2 / h2)
        v = (4 / (torch.pi ** 2)) * (atan1 - atan2) ** 2
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        ciou = iou - (center_dist2 / c2) - alpha * v
        return ciou

    # pairwise IoU matrix case (N,M)
    # compute pairwise intersections
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])  # (N,M,2)
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0))[:, None]
    area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0))[None, :]
    union = area1 + area2 - inter
    iou_matrix = inter / union.clamp(min=eps)
    if CIoU:
        # CIoU for pairwise would be expensive and is rarely expected; fall back to IoU matrix
        return iou_matrix
    return iou_matrix


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N,4) x1,y1,x2,y2
    boxes2: (M,4) x1,y1,x2,y2
    returns: (N,M) IoU matrix
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    # areas
    area1 = (boxes1[:,2] - boxes1[:,0]).clamp(min=0) * (boxes1[:,3] - boxes1[:,1]).clamp(min=0) # (N,)
    area2 = (boxes2[:,2] - boxes2[:,0]).clamp(min=0) * (boxes2[:,3] - boxes2[:,1]).clamp(min=0) # (M,)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)   # (N,M,2)
    inter = wh[:,:,0] * wh[:,:,1]  # (N,M)
    union = area1[:,None] + area2[None,:] - inter
    iou = inter / (union + 1e-6)

    return iou


def centers_of_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (N,4) x1,y1,x2,y2
    returns: (N,2) center (cx,cy)
    """
    cx = (boxes[:,0] + boxes[:,2]) * 0.5
    cy = (boxes[:,1] + boxes[:,3]) * 0.5

    return torch.stack([cx, cy], dim=1)