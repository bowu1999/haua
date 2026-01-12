from typing import Tuple, Optional, Union

import os
import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, np.ndarray, Image.Image]


def processImage2opencvNumpy(img_input: ImageInput) -> Tuple[np.ndarray, Optional[str]]:
    """
    将不同类型的输入统一转换为 OpenCV BGR numpy 数组。
    返回: (image_numpy_bgr, original_filename_if_exists)
    """
    img = None
    filename = None
    # 处理路径字符串
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Image not found: {img_input}")
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Failed to read image: {img_input}")
        filename = img_input
    # 处理 PIL Image
    elif isinstance(img_input, Image.Image):
        # PIL 是 RGB，OpenCV 需要 BGR
        img = np.asarray(img_input)
        # 处理灰度图或RGBA
        if img.ndim == 2:  # 灰度
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:  # RGB -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:  # RGBA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # 处理 Numpy 数组 (假设已经是 BGR 或 灰度)
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy() # 避免修改原图
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 假设输入已经是 BGR，不做转换
    else:
        raise TypeError(f"Unsupported image type: {type(img_input)}")

    return img, filename