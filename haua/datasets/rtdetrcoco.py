import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class RTDETRInstanceDataset(Dataset):
    """
    完全适配 RT-DETR 的数据集类，支持目标检测和实例分割。
    
    特性：
    1. 输出格式完全符合 RT-DETR 要求 (Box: cxcywh norm, Metadata: orig_size, size)。
    2. 内置 Mask 生成逻辑 (基于 pycocotools)。
    3. 自动过滤无效框和无效 Mask。
    4. 支持类别重映射 (remap_mscoco_category)。
    """
    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        return_masks: bool = False,
        remap_mscoco_category: bool = False
    ):
        self.root = Path(root)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        # 构建类别映射 (如果需要)
        if remap_mscoco_category:
            categories = self.coco.dataset['categories']
            self.category2label = {cat['id']: i for i, cat in enumerate(categories)}
        else:
            self.category2label = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict]:
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root / img_info['file_name']

        # 1. 读取图像
        image = Image.open(img_path).convert('RGB')
        w_orig, h_orig = image.size

        # 2. 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 过滤掉 iscrowd 的标注 (RT-DETR 通常不训练 crowd)
        anns = [obj for obj in anns if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 3. 解析标注数据
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for obj in anns:
            # --- Box 处理 ---
            # COCO 原始格式: [x, y, w, h]
            x, y, w, h = obj['bbox']
            
            # 过滤无效框 (面积极小或宽高为负)
            if w < 1 or h < 1:
                continue

            # 转换 Box 格式: [x, y, w, h] -> [x1, y1, x2, y2] (绝对坐标)
            # 这一步是为了方便后续 transforms 处理 (如 Resize, Crop)
            # 注意：RT-DETR 的 ConvertCocoPolysToMask 里做了 clamp，这里我们也做
            x1 = np.clip(x, 0, w_orig)
            y1 = np.clip(y, 0, h_orig)
            x2 = np.clip(x + w, 0, w_orig)
            y2 = np.clip(y + h, 0, h_orig)
            
            # 再次检查有效性
            if (x2 <= x1) or (y2 <= y1):
                continue

            # --- Label 处理 ---
            cat_id = obj['category_id']
            if self.remap_mscoco_category and self.category2label:
                label = self.category2label.get(cat_id, -1)
            else:
                label = cat_id
            
            if label == -1: # 跳过未定义类别
                continue

            # --- Mask 处理 (如果需要) ---
            if self.return_masks:
                seg = obj.get('segmentation', None)
                if seg:
                    # 使用 pycocotools 生成 mask
                    if isinstance(seg, list):
                        # Polygon 格式
                        rles = coco_mask.frPyObjects(seg, h_orig, w_orig) # type: ignore
                        mask = coco_mask.decode(rles)
                    elif isinstance(seg, dict):
                        # RLE 格式
                        mask = coco_mask.decode(seg) # type: ignore
                    else:
                        mask = None

                    if mask is not None:
                        if len(mask.shape) < 3:
                            mask = mask[..., None]
                        mask = torch.as_tensor(mask, dtype=torch.uint8)
                        mask = mask.any(dim=2) # 合并多边形
                        masks.append(mask)
                    else:
                        # 如果有 box 但没 mask，且要求 return_masks，通常跳过该物体
                        continue 
                else:
                    continue

            # 所有检查通过，添加数据
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            areas.append(obj.get('area', (x2-x1)*(y2-y1)))
            iscrowd.append(obj.get('iscrowd', 0))

        # 4. 转换为 Tensor
        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)
            if self.return_masks:
                masks_t = torch.stack(masks, dim=0)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            if self.return_masks:
                masks_t = torch.zeros((0, h_orig, w_orig), dtype=torch.uint8)

        # 5. 构建初始 Target 字典
        target = {
            'boxes': boxes_t,
            'labels': labels_t,
            'image_id': torch.tensor([img_id]),
            'area': areas_t,
            'iscrowd': iscrowd_t,
            'orig_size': torch.tensor([h_orig, w_orig]), # RT-DETR 必需
            'size': torch.tensor([h_orig, w_orig]),      # 初始 size 等于 orig_size
            'idx': torch.tensor([index])                 # RT-DETR 必需
        }

        if self.return_masks:
            target['masks'] = masks_t # type: ignore

        # 6. 应用 Transforms (Resize, Augmentation 等)
        # 注意：transforms 必须能同时处理 image, target['boxes'], target['masks']
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # 7. 核心适配：将 Box 转换为 RT-DETR 要求的 CXCYWH 归一化格式
        # 注意：此时 image 已经被 resize 了，target['size'] 应该在 transforms 里被更新
        # 如果你的 transforms 没有更新 target['size']，我们需要用 image.shape 更新
        
        # 获取当前图像尺寸 (Transform 之后)
        if isinstance(image, torch.Tensor):
            h_curr, w_curr = image.shape[-2:]
        else:
            w_curr, h_curr = image.size
            
        target['size'] = torch.tensor([h_curr, w_curr]) # 更新当前尺寸

        boxes = target['boxes']
        if len(boxes) > 0:
            # XYXY (Abs) -> CXCYWH (Norm)
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w_b = x2 - x1
            h_b = y2 - y1

            cx = cx / w_curr
            cy = cy / h_curr
            w_b = w_b / w_curr
            h_b = h_b / h_curr

            target['boxes'] = torch.stack([cx, cy, w_b, h_b], dim=-1).clamp(0.0, 1.0)

        return image, target # type: ignore

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
