from pathlib import Path
from typing import Callable, List, Tuple, Optional, Dict

import json
import math
import random

from PIL import Image, ImageOps
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
import torchvision.transforms.functional as F
try:
    from pycocotools.coco import COCO
except Exception as e:
    COCO = None
    # 如果没有 pycocotools，Dataset 的初始化会报错，提示安装 pycocotools


# 在文件顶部（imports 之后）添加 COCO id -> 0..79 的映射
# 原始 COCO 的 80 类在 annotation 中对应的 category_id（不连续）
COCO80_ORIG_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90]
# map original id -> contiguous 0..79
COCO_ID_TO_80 = {orig_id: i for i, orig_id in enumerate(COCO80_ORIG_IDS)}
# optionally inverse mapping (0..79 -> orig id)
COCO_80_TO_ORIG = {i: orig for i, orig in enumerate(COCO80_ORIG_IDS)}

# 标准 COCO 80 类名，索引 0..79 对应上面的 COCO80_ORIG_IDS 映射顺序
coco80_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
    "teddy bear", "hair drier", "toothbrush"]


def coco_segmentation_to_mask(segmentation, height: int, width: int) -> Optional[np.ndarray]:
    """
    将 COCO 的 segmentation (polygon / RLE) 转为 binary mask (H, W)
    返回 np.ndarray[uint8]，无效 segmentation 返回 None
    """
    if segmentation is None:
        return None

    # Polygon 格式 (List)
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return None
        # 将 polygon 转为 RLE 列表
        rles = mask_utils.frPyObjects(segmentation, height, width)
        # 将多个 polygon 片段合并为一个 RLE
        rle = mask_utils.merge(rles)

    # RLE 格式 (Dict)
    elif isinstance(segmentation, dict):
        # 关键修复：
        # COCO JSON 中的 RLE counts 可能是 list (未压缩)。
        # decode() 需要 bytes (压缩)。
        # frPyObjects 能自动处理 list -> bytes 的转换。
        rle = mask_utils.frPyObjects(segmentation, height, width)

    else:
        return None

    # Decode RLE -> Binary Mask
    mask = mask_utils.decode(rle)  # (H, W) or (H, W, 1)
    
    # 处理维度，确保返回 (H, W)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    return mask.astype(np.uint8)


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO 的 bbox 是 [x, y, w, h] -> 转为 [x1, y1, x2, y2]"""
    boxes = boxes.copy().astype(np.float32)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return boxes


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        return_masks: bool = False
    ):
        if COCO is None:
            raise RuntimeError("pycocotools is required. Install with `pip install pycocotools`.")

        self.root = Path(root)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.return_masks = return_masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root / img_info['file_name']

        image = Image.open(img_path).convert('RGB')
        W, H = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        masks = []

        for ann in anns:
            if 'bbox' not in ann:
                continue

            x, y, w, h = ann['bbox']
            # 过滤掉无效的小框
            if w <= 1 or h <= 1:
                continue

            orig_cat = ann['category_id']
            if orig_cat not in COCO_ID_TO_80:
                continue

            # --- 修复核心开始 ---
            # 1. 如果需要 mask，先尝试解析 mask
            current_mask = None
            if self.return_masks:
                seg = ann.get('segmentation', None)
                current_mask = coco_segmentation_to_mask(seg, H, W)
                
                # 关键修复：如果 mask 无效（None 或 全黑），直接跳过这个物体
                # 这样就不会出现有 box 没 mask 的情况了
                if current_mask is None or current_mask.sum() <= 0:
                    continue
            
            # 2. 只有 mask 有效（或不需要 mask）时，才添加 box 和 label
            boxes.append([x, y, x + w, y + h])
            labels.append(COCO_ID_TO_80[orig_cat])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

            if self.return_masks:
                masks.append(current_mask)
            # --- 修复核心结束 ---

        # 处理空数据的情况 (防止之前的 IndexError: too many indices)
        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.uint8)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.uint8)

        target = {
            'boxes': boxes_t,
            'labels': labels_t,
            'image_id': torch.tensor([img_id]),
            'area': areas_t,
            'iscrowd': iscrowd_t}

        if self.return_masks:
            if len(masks) > 0:
                target['masks'] = torch.from_numpy(np.stack(masks, axis=0).astype(np.float32))
            else:
                target['masks'] = torch.zeros((0, H, W), dtype=torch.float32)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def coco_collate(batch):
    images, targets = zip(*batch)
    # Stack images -> [B, C, H, W]
    inputs = torch.stack(images, dim=0)
    # 计算当前 batch 中最大 bbox 数量
    max_num_boxes = max([len(t['boxes']) for t in targets])
    # 初始化用于存储 padded bboxes 和 labels 的 tensor
    batch_size = len(targets)
    gt_bboxes = torch.zeros((batch_size, max_num_boxes, 4), dtype=torch.float32)
    gt_labels = -torch.ones((batch_size, max_num_boxes), dtype=torch.int64)  # 使用-1填充
    for i, target in enumerate(targets):
        num_boxes = len(target['boxes'])
        if num_boxes > 0:
            gt_bboxes[i, :num_boxes] = target['boxes']
            gt_labels[i, :num_boxes] = target['labels']
    data_samples = {
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels}

    return inputs, data_samples


def coco_seg_collate(batch):
    images, targets = zip(*batch)
    inputs = torch.stack(images, dim=0)  # [B,C,H,W]
    batch_size, _, H, W = inputs.shape

    num_boxes_per_img = [len(t['boxes']) for t in targets]
    max_num_boxes = max(num_boxes_per_img) if batch_size > 0 else 0

    gt_bboxes = torch.zeros((batch_size, max_num_boxes, 4), dtype=torch.float32)
    gt_labels = -torch.ones((batch_size, max_num_boxes), dtype=torch.int64)
    gt_masks  = torch.zeros((batch_size, max_num_boxes, H, W), dtype=torch.float32)
    num_gts   = torch.zeros((batch_size,), dtype=torch.long)

    for i, target in enumerate(targets):
        boxes_i: torch.Tensor = target['boxes']    # (Gi, 4)
        labels_i: torch.Tensor = target['labels']  # (Gi,)
        masks_i: torch.Tensor = target.get('masks', None)  # (Gi, H, W)

        if masks_i is None or len(boxes_i) == 0:
            num_gts[i] = 0
            continue

        assert masks_i.ndim == 3, "target['masks'] 必须是 (N,H,W) bitmask"

        filtered_boxes = []
        filtered_labels = []
        filtered_masks = []

        for box, label, mask in zip(boxes_i, labels_i, masks_i):
            # 如果 mask 全 0，认为是无效实例
            if mask.sum() <= 0:
                continue
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_masks.append(mask)

        Gi_keep = len(filtered_masks)
        num_gts[i] = Gi_keep

        if Gi_keep == 0:
            continue

        Gi_keep = min(Gi_keep, max_num_boxes)
        gt_bboxes[i, :Gi_keep] = torch.stack(filtered_boxes[:Gi_keep], dim=0)
        gt_labels[i, :Gi_keep] = torch.stack(filtered_labels[:Gi_keep], dim=0)
        gt_masks[i,  :Gi_keep] = torch.stack(filtered_masks[:Gi_keep], dim=0)

    data_samples = {
        'img': inputs,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_masks': gt_masks,
        'num_gts': num_gts}

    return inputs, data_samples

