from typing import List, Tuple, Optional, Literal, Union

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

from ..bbox import BoxFormat, convertBbox
from .unified_fmt import ImageInput, processImage2opencvNumpy


def drawBoxes(
    img_input: ImageInput,
    boxes: Union[List, np.ndarray, torch.Tensor],
    fmt: BoxFormat = 'xyxy',
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show: bool = True,
    backend: Literal['cv2', 'plt'] = 'plt',
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    在图像上绘制目标框。

    Args:
        img_input: 图像路径(str)、OpenCV图像(np.ndarray) 或 PIL图像(Image.Image)。
        boxes: 边界框列表。
        fmt: 边界框格式。
        labels: 标签列表。
        colors: 颜色列表。
        line_thickness: 线条粗细。
        font_scale: 字体大小。
        show: 是否显示图像。
        backend: 显示后端 ('cv2' 或 'plt')。
        save_path: 保存路径。如果为 None 且输入是路径，会自动生成后缀；如果输入是对象且为 None，则不保存。

    Returns:
        np.ndarray: 绘制了框的图像 (OpenCV BGR 格式)。
    """
    img, source_path = processImage2opencvNumpy(img_input)
    img_h, img_w = img.shape[:2]
    if isinstance(boxes, (np.ndarray, torch.Tensor)):
        boxes = boxes.tolist()
    if labels is not None and len(labels) != len(boxes):
        print(
            f"Warning: labels length ({len(labels)}) != boxes length ({len(boxes)})."
            f"Labels may be truncated or misaligned.")
    if colors is None:
        colors = []
        for i in range(len(boxes)):
            color = (int(37 * i) % 255, int(17 * i) % 255, int(29 * i) % 255)
            colors.append(color)
    for i, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = convertBbox(box, fmt, img_w, img_h)
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness)
            if labels is not None and i < len(labels):
                label = str(labels[i])
                ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                # 标签背景框 (处理边界溢出)
                text_y2 = y1
                text_y1 = y1 - th - 4
                if text_y1 < 0: # 如果标签跑出上边界，移到框内部
                    text_y1 = y1
                    text_y2 = y1 + th + 4
                    text_pos = (x1 + 1, y1 + th + 2)
                else:
                    text_pos = (x1 + 1, y1 - 2)

                cv2.rectangle(img, (x1, text_y1), (x1 + tw + 2, text_y2), color, -1)
                cv2.putText(
                    img, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        except Exception as e:
            print(f"Error drawing box {i}: {e}")
            continue
    if save_path is None and source_path is not None:
        base, ext = os.path.splitext(source_path)
        save_path = base + "_boxed" + ext
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"Saved to: {save_path}")
    if show:
        if backend == 'cv2':
            cv2.imshow("Image with boxes", img)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif backend == 'plt':
            # matplotlib 显示（BGR -> RGB）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    return img


def tensor2imgShow(tensor_img, title="Tensor Image", figsize=(6, 6), is_rgb=True):
    """
    将PyTorch的Tensor张量绘制为图像的通用函数

    Args:
        tensor_img (torch.Tensor): 形状为[C, H, W]或[H, W]的PyTorch张量
        title (str): 窗口标题
        figsize (tuple): 窗口大小，默认为(6, 6)
        is_rgb (bool): 是否将通道作为颜色通道绘制，默认为True
    """
    # -------------------------- 张量预处理：通用兼容逻辑 --------------------------
    if tensor_img.is_cuda:
        tensor_img = tensor_img.cpu()
    img_tensor = tensor_img.clone().detach()
    if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if len(img_tensor.shape) == 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
        is_rgb = False
    # -------------------------- 数据归一化：适配图像像素范围 --------------------------
    # 核心：PyTorch的图像张量一般是 float类型(0-1 或 0-255) / uint8类型(0-255)，统一转成0-255的uint8
    if img_tensor.dtype in [torch.float16, torch.float32, torch.float64]:
        # 如果张量数值在0~1之间 → 乘以255放大到像素范围
        if torch.max(img_tensor) <= 1.0:
            img_tensor = img_tensor * 255.0
    # 转为numpy数组 + 转为uint8像素格式（图像标准格式）
    img_np = img_tensor.numpy().astype(np.uint8)
    # -------------------------- 维度转换：CHW → HWC --------------------------
    # PyTorch 默认存储格式：C(通道) H(高) W(宽) 【CHW】
    # Matplotlib/pillow 绘图格式：H(高) W(宽) C(通道) 【HWC】
    if len(img_np.shape) == 3 and is_rgb:
        img_np = np.transpose(img_np, (1, 2, 0))
    # -------------------------- 绘制图像 --------------------------
    plt.figure(figsize=figsize)
    plt.imshow(img_np, cmap='gray' if not is_rgb else None) # 灰度图用cmap='gray'，彩色图直接绘制
    plt.title(title, fontsize=12)
    plt.axis('off')  # 关闭坐标轴，美观
    plt.show()