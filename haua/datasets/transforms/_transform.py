from typing import Tuple, List, Dict, Optional, Union, Callable

import math
import random
import numpy as np
from PIL import Image, ImageOps

import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor:
    """
    核心步骤：
    1. 将 PIL Image 转为 Tensor (C, H, W)
    2. 自动将像素值从 [0, 255] 除以 255 归一化到 [0.0, 1.0]
    """
    def __call__(self, image, target=None):
        image = F.to_tensor(image)

        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        
        return image, target


class SanityCheck:
    """
    数据清洗与合法性检查类 (Sanity Check)。

    作用：
        在经过一系列几何变换（如 Resize, RandomCrop, Flip）后，过滤掉那些变得无效的
        标注框（BBox）。
    
    具体逻辑：
        1. 计算每个 BBox 的宽和高。
        2. 剔除 宽 < 1 或 高 < 1 的“退化”框（degenerate boxes）。
        3. 同步过滤对应的 labels, masks, area, iscrowd 等信息，确保数据对齐。

    为什么需要它：
        - 防止计算 IoU Loss 时出现除以零 (Divide by zero)。
        - 防止出现 NaN (Not a Number) 导致训练中断。
        - 提高模型训练的稳定性。
    """
    def __call__(self, image, target):
        if target is None: 
            return image, target
            
        if 'boxes' in target:
            boxes = target['boxes']
            
            # --- 新增防御性代码 ---
            # 如果 boxes 为空，直接返回，避免后续切片报错
            if boxes.numel() == 0:
                return image, target
            
            # 如果 boxes 是 1维的 (例如 shape(4,))，强制转为 2维 (1, 4)
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
                target['boxes'] = boxes
            # --------------------

            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws > 1) & (hs > 1)
            
            target['boxes'] = boxes[keep]
            target['labels'] = target['labels'][keep]
            if 'area' in target: target['area'] = target['area'][keep]
            if 'iscrowd' in target: target['iscrowd'] = target['iscrowd'][keep]
            if 'masks' in target and len(target['masks']) > 0:
                target['masks'] = target['masks'][keep]

        return image, target


class UnifiedResize:
    """
    统一图像缩放类，支持三种主流缩放策略。

    Args:
        mode (str): 缩放模式，可选值：
            - 'stretch': 强制拉伸到 target_size，不保持宽高比 (Simple Resizing)。
            - 'letterbox': 保持宽高比缩放，并填充灰边至 target_size (YOLO style)。
            - 'keep_ratio': 保持宽高比，短边缩放到 min_size，长边不超过 max_size (R-CNN style)。
        
        target_size (int | Tuple[int, int]): 目标尺寸 (h, w)。
            - 用于 'stretch' 和 'letterbox' 模式。
            - 如果是 int，则表示 (size, size)。
        
        min_size (int | List[int]): 短边尺寸范围。
            - 仅用于 'keep_ratio' 模式。
            - 如果是 list，训练时会从中随机选择一个作为短边尺寸（多尺度训练）。
        
        max_size (int): 长边最大尺寸限制。
            - 仅用于 'keep_ratio' 模式。
        
        stride (int): 对齐步长 (通常为 32)。
            - 仅用于 'letterbox' 模式，用于确保填充后的尺寸能被 stride 整除。
        
        pad_color (int | Tuple): 填充颜色。
            - 仅用于 'letterbox' 模式，默认 114。
    """

    def __init__(
        self, 
        mode: str = 'stretch',
        target_size: Union[int, Tuple[int, int]] = 640,
        min_size: Union[int, List[int]] = 800,
        max_size: int = 1333,
        stride: int = 32,
        pad_color: Union[int, Tuple[int, int, int]] = 114
    ):
        assert mode in ['stretch', 'letterbox', 'keep_ratio'], \
            f"mode must be one of ['stretch', 'letterbox', 'keep_ratio'], got {mode}"
        
        self.mode = mode
        
        # 处理 target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
            
        # 处理 min_size
        if isinstance(min_size, int):
            self.min_size = (min_size,)
        else:
            self.min_size = tuple(min_size)
            
        self.max_size = max_size
        self.stride = stride
        self.pad_color = pad_color

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Tensor]]=None):
        if self.mode == 'stretch':
            return self._stretch(image, target)
        elif self.mode == 'letterbox':
            return self._letterbox(image, target)
        elif self.mode == 'keep_ratio':
            return self._keep_ratio(image, target)
        else:
            return image, target

    def _stretch(self, image: Image.Image, target: Optional[Dict] = None):
        """模式 1: 强制拉伸"""
        orig_w, orig_h = image.size
        target_h, target_w = self.target_size
        
        # 1. Resize Image
        image = F.resize(image, self.target_size, interpolation=F.InterpolationMode.BILINEAR) # type: ignore
        
        if target is None: return image, target

        scale_w = target_w / orig_w
        scale_h = target_h / orig_h

        # 2. Resize Boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, target_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, target_h)
            target['boxes'] = boxes
            if 'area' in target: target['area'] *= (scale_w * scale_h)

        # 3. Resize Masks
        if 'masks' in target and len(target['masks']) > 0:
            target['masks'] = F.resize(
                target['masks'], self.target_size, interpolation=F.InterpolationMode.NEAREST # type: ignore
            )
            
        return image, target

    def _letterbox(self, image: Image.Image, target: Optional[Dict] = None):
        """模式 2: LetterBox (YOLO Style)"""
        orig_w, orig_h = image.size
        target_h, target_w = self.target_size
        
        # 计算缩放比例 (保持宽高比)
        r = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))
        
        # Resize
        if (orig_w, orig_h) != (new_w, new_h):
            image = image.resize((new_w, new_h), resample=Image.BILINEAR) # type: ignore
            
        # 计算 Padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        # Auto stride (可选，这里为了保持输出尺寸固定，通常训练时设为 False，推理时设为 True)
        # 这里为了简化，严格按照 target_size 填充
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            image = ImageOps.expand(
                image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.pad_color)
            
        if target is None: return image, target

        # Update Boxes (Scale + Shift)
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            boxes = boxes * r
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, target_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, target_h)
            target['boxes'] = boxes
            if 'area' in target: target['area'] *= (r * r)

        # Update Masks (Interpolate + Pad)
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks'].unsqueeze(1) # (N, 1, H, W)
            masks = torch.nn.functional.interpolate(masks, size=(new_h, new_w), mode="nearest")
            masks = torch.nn.functional.pad(masks, (pad_left, pad_right, pad_top, pad_bottom))
            target['masks'] = masks.squeeze(1)
            
        return image, target

    def _keep_ratio(self, image: Image.Image, target: Optional[Dict] = None):
        """模式 3: Short-side Resize (R-CNN Style)"""
        orig_w, orig_h = image.size
        
        # 随机选择短边尺寸
        min_size = random.choice(self.min_size)
        
        # 计算比例
        min_orig = float(min((orig_w, orig_h)))
        max_orig = float(max((orig_w, orig_h)))
        scale = min_size / min_orig
        
        if max_orig * scale > self.max_size:
            scale = self.max_size / max_orig
            
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        
        # Resize Image
        image = image.resize((new_w, new_h), resample=Image.BILINEAR) # type: ignore
        
        if target is None: return image, target
        
        # Resize Boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            boxes = boxes * scale
            # 注意：这里不需要 clamp 到固定尺寸，因为图像尺寸变了
            target['boxes'] = boxes
            if 'area' in target: target['area'] *= (scale * scale)
            
        # Resize Masks
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks'].unsqueeze(1)
            masks = torch.nn.functional.interpolate(masks, scale_factor=scale, mode="nearest")
            target['masks'] = masks.squeeze(1)
            
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image: Image.Image, target: Optional[Dict]=None):
        if random.random() < self.prob:
            image = F.hflip(image) # type: ignore
            if target is None: return image, target
            
            w, h = image.size
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
            if 'masks' in target and target['masks'].shape[0] > 0:
                target['masks'] = target['masks'].flip(-1)

        return image, target


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image.Image, target: Optional[Dict]=None):
        image = self.transform(image)

        return image, target


def get_train_transforms(img_size: int=640, mode='stretch'):
    if isinstance(img_size, int):
        img_size = (img_size, img_size) # type: ignore
    else:
        img_size = img_size

    return Compose([
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        RandomHorizontalFlip(prob=0.5),
        UnifiedResize(mode=mode, target_size=img_size),
        ToTensor(),
        SanityCheck()])


def get_val_transforms(img_size: int=640, mode='stretch'):
    if isinstance(img_size, int):
        img_size = (img_size, img_size) # type: ignore
    else:
        img_size = img_size

    return Compose([
        UnifiedResize(mode=mode, target_size=img_size),
        ToTensor()])


get_infer_transforms = get_val_transforms