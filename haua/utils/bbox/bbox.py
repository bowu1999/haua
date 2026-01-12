from typing import List, Tuple, Optional, Literal, Union, Sequence
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch


BoxFormat = Literal[
    'xyxy',
    'xyxyn',
    'xywh',
    'xywhn',
    'cxcywh',
    'cxcywhn']

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]]


def convertBbox(
    box: Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray],
    fmt: BoxFormat,
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int]:
    # 兼容 Tensor 或 Numpy 输入，统一转为 list/tuple 处理
    if isinstance(box, (torch.Tensor, np.ndarray)):
        box = box.tolist()

    if fmt == 'xyxy':
        x1, y1, x2, y2 = box
    elif fmt == 'xywh':
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif fmt == 'xyxyn':
        x1n, y1n, x2n, y2n = box
        x1, y1 = x1n * img_w, y1n * img_h
        x2, y2 = x2n * img_w, y2n * img_h
    elif fmt == 'xywhn':
        xn, yn, wn, hn = box
        x1, y1 = xn * img_w, yn * img_h
        x2, y2 = (xn + wn) * img_w, (yn + hn) * img_h
    elif fmt == 'cxcywh':
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    elif fmt == 'cxcywhn':
        cxn, cyn, wn, hn = box
        cx = cxn * img_w
        cy = cyn * img_h
        w = wn * img_w
        h = hn * img_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    # 坐标限制在图像范围内
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w - 1))
    y2 = max(0, min(int(round(y2)), img_h - 1))

    return x1, y1, x2, y2


def _toTensor(boxes: ArrayLike) -> Tuple[torch.Tensor, str]:
    """将输入统一转换为 torch.Tensor，并返回原始类型信息。"""
    if isinstance(boxes, torch.Tensor):
        return boxes, "torch"
    elif isinstance(boxes, np.ndarray):
        return torch.from_numpy(boxes), "numpy"
    elif isinstance(boxes, Sequence):
        # list / tuple 等，转成 tensor
        return torch.tensor(boxes, dtype=torch.float32), "list"
    else:
        raise TypeError(f"Unsupported boxes type: {type(boxes)}")


def _fromTensor(t: torch.Tensor, org_type: str) -> Union[torch.Tensor, np.ndarray, list]:
    """将 torch.Tensor 转回原始类型。"""
    if org_type == "torch":
        return t
    elif org_type == "numpy":
        return t.numpy()
    elif org_type == "list":
        return t.tolist()
    else:
        raise ValueError(f"Unknown original type: {org_type}")


def bboxArea(
    boxes: ArrayLike,
    fmt: BoxFormat = 'xyxy',
    has_class: bool | None = None,
) -> Union[torch.Tensor, np.ndarray, list]:
    """
    计算 bbox 面积，支持:
      - 输入类型: torch.Tensor, np.ndarray, list
      - 格式: 'xyxy', 'xyxyn', 'xywh', 'xywhn', 'cxcywh', 'cxcywhn'
      - 自动识别 [cls + 4 coords] 或 [4 coords]

    Args:
        boxes:
            形状可以是 [..., 4] 或 [..., 5]，当为 5 时视为 [cls, x1, y1, x2, y2] 等。
        fmt:
            BoxFormat 中的一种。
        has_class:
            - None: 自动判断，如果最后一维是 5 则视为含类别，否则视为不含类别；
            - True: 强制视为 [cls + 4 coords]；
            - False: 强制视为纯 4 coords。

    Returns:
        areas: 与输入类型一致的面积数组/张量/列表，形状为 [...,]
    """
    t, org_type = _toTensor(boxes)

    if t.ndim == 1:
        # [4] or [5] -> [1, 4] / [1, 5]
        t = t.unsqueeze(0)

    if has_class is None:
        if t.size(-1) == 5:
            has_class = True
        elif t.size(-1) == 4:
            has_class = False
        else:
            raise ValueError(
                f"Cannot infer has_class from last dim={t.size(-1)}, "
                "expect 4 (bbox) or 5 (cls + bbox). Please set has_class explicitly.")
    # 取出 bbox 部分
    if has_class:
        if t.size(-1) != 5:
            raise ValueError(f"has_class=True but last dim is {t.size(-1)} (expected 5)")
        coords = t[..., 1:]  # [cls, x1, y1, x2, y2] -> [x1, y1, x2, y2]
    else:
        if t.size(-1) != 4:
            raise ValueError(f"has_class=False but last dim is {t.size(-1)} (expected 4)")
        coords = t
    # 计算宽高
    if fmt in ("xyxy", "xyxyn"):
        x1 = coords[..., 0]
        y1 = coords[..., 1]
        x2 = coords[..., 2]
        y2 = coords[..., 3]
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
    elif fmt in ("xywh", "xywhn"):
        w = coords[..., 2].clamp(min=0)
        h = coords[..., 3].clamp(min=0)
    elif fmt in ("cxcywh", "cxcywhn"):
        w = coords[..., 2].clamp(min=0)
        h = coords[..., 3].clamp(min=0)
    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    areas = w * h  # 对于 *n 格式，这是相对面积；否则是像素面积

    return _fromTensor(areas, org_type)