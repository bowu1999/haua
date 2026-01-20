from typing import List, Union, Tuple, Optional, Dict

import cv2
import numpy as np

from ..image import processImage2opencvNumpy


def rle2mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """
    将 RLE 格式转为 binary mask (H, W)
    rle: {"counts": [...], "size": [h, w]}

    Args:
        rle (Dict): RLE 字典
        height (int): mask 图像高度
        width (int): mask 图像宽度

    Returns:
        np.ndarray: mask 图像
    """
    counts = rle["counts"]
    size = rle["size"]
    h, w = size
    assert h == height and w == width, "RLE size 与图像尺寸不一致"
    mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if idx + c > mask.size:
            break
        if val == 1:
            mask[idx: idx + c] = 1
        idx += c
        val ^= 1

    return mask.reshape((h, w), order="F")


def mask2polygon(mask: np.ndarray) -> List[np.ndarray]:
    """
    将 binary mask 转为 polygon (contours)
    Args:
        mask: (H, W) uint8, 0/1
    Returns:
        list of numpy arrays, 每个 array 是一个轮廓点集 (N, 1, 2)
    """
    # 确保是 uint8 二值图
    mask = (mask > 0.5).astype(np.uint8) * 255
    # findContours 寻找轮廓
    # cv2.RETR_EXTERNAL 只检测外轮廓，如果需要空洞则用 cv2.RETR_TREE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return list(contours)


def polygon2mask(
    polygons: Union[List[float], List[List[float]], np.ndarray],
    height: int,
    width: int
) -> np.ndarray:
    """
    将多边形坐标转换为二值 Mask (H, W)。
    
    Args:
        polygons: 多边形数据，支持以下格式：
            1. COCO 标准格式 (List of lists): [[x1, y1, x2, y2, ...], [x1, y1, ...]] (支持带孔或多段物体)
            2. 单个扁平列表 (List): [x1, y1, x2, y2, ...]
            3. Numpy 数组: (N, 2) 或 (N, 1, 2)
        height (int): 输出 Mask 的高度
        width (int): 输出 Mask 的宽度

    Returns:
        mask (np.ndarray): (H, W) uint8 数组，前景为 1，背景为 0。
    """
    # 初始化全黑 mask
    mask = np.zeros((height, width), dtype=np.uint8)

    if not polygons:
        return mask

    # 统一转换为 cv2.fillPoly 需要的格式: List[np.ndarray(N, 2)]
    poly_list = []

    # 情况 1: 输入是 Numpy 数组
    if isinstance(polygons, np.ndarray):
        # 确保形状是 (N, 2)
        if polygons.ndim == 3:
            polygons = polygons.squeeze(1) # (N, 1, 2) -> (N, 2)
        poly_list = [polygons]

    # 情况 2: 输入是 List
    elif isinstance(polygons, list):
        # 判断是 [x, y, ...] 还是 [[x, y, ...], ...]
        if len(polygons) > 0 and isinstance(polygons[0], (int, float)):
            # 单个扁平列表 -> reshape 为 (N, 2) 并放入 list
            poly_list = [np.array(polygons).reshape(-1, 2)]
        else:
            # 列表的列表 (COCO 格式) -> 遍历每个 poly 并 reshape
            poly_list = [np.array(p).reshape(-1, 2) for p in polygons]

    # 绘制多边形
    # cv2.fillPoly 需要坐标点为 int 类型
    int_polys = [p.astype(np.int32) for p in poly_list]
    
    # color=1 表示前景像素值为 1
    cv2.fillPoly(mask, int_polys, 1)

    return mask


def mask2rle(mask: np.ndarray) -> Dict:
    """
    将 binary mask 转为 COCO RLE 格式（不依赖 pycocotools）

    Args:
        mask: np.ndarray, shape (H, W), 0/1 或 bool

    Returns:
        rle: dict
            {
              "counts": [int, int, ...],
              "size": [H, W]
            }
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be np.ndarray, got {type(mask)}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    # 保证是 0/1
    mask = mask.astype(np.uint8)

    h, w = mask.shape

    pixels = mask.flatten(order="F")

    counts = []
    prev = 0
    cnt = 0

    for p in pixels:
        if p == prev:
            cnt += 1
        else:
            counts.append(cnt)
            cnt = 1
            prev = p

    counts.append(cnt)

    return {"counts": counts, "size": [h, w]}


def drawMasksOnImage(
    img_input,
    masks: List[Union[np.ndarray, Dict]],
    *,
    alpha: float = 0.5,
    draw_contour: bool = True,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    在图像上绘制多个 mask（支持 binary mask 或 RLE）

    Args:
       img_input(numpy): 输入图像，BGR 格式
       masks(List[Union[np.ndarray, Dict]]): 需要绘制的 mask，每个元素可以是
       alpha(float): mask 的透明度
       draw_contour(bool): 是否绘制轮廓
       colors(Optional[List[Tuple[int, int, int]]]): 每个 mask 的 BGR 颜色，不给则随机

    Returns:
        numpy: 绘制好的图像，BGR 格式
    """
    img, _ = processImage2opencvNumpy(img_input)
    h, w = img.shape[:2]

    if colors is None:
        rng = np.random.default_rng(42)
        colors = [
            tuple(int(c) for c in rng.integers(0, 255, size=3)) for _ in range(len(masks))]

    overlay = img.copy()

    for idx, mask_item in enumerate(masks):
        # ---------- 统一转 binary mask ----------
        if isinstance(mask_item, dict):  # RLE
            mask = rle2mask(mask_item, h, w)
        elif isinstance(mask_item, np.ndarray):
            mask = mask_item
        else:
            raise TypeError(f"Unsupported mask type: {type(mask_item)}")

        if mask.shape != (h, w):
            raise ValueError(f"Mask shape {mask.shape} != image shape {(h, w)}")

        mask = mask.astype(bool)
        color = colors[idx]

        # ---------- 填充 mask ----------
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

        # ---------- 画轮廓 ----------
        if draw_contour:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, thickness=2)

    return overlay
