from typing import Any, Union, Tuple, Optional, Sequence, List

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# 假设这些工具函数在你的项目中存在
from ...utils.bbox import cxcywh2xyxy
from ...utils.mask import mask2polygon # 假设你有这个函数，如果没有，可以使用 cv2.findContours

# 简单的颜色列表 (复用之前的)
FIXED_COLOR_LIST = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
] + [(int(255 * np.random.rand()), int(255 * np.random.rand()), int(255 * np.random.rand())) \
     for _ in range(100)]

__all__ = [
    "RTDETRDecoder",
    "RTDETRResult",
    "RTDETRSegDecoder",
    "RTDETRSegResult"
]


class RTDETRDecoder:
    """
    RT-DETR 解码器
    负责将模型输出的归一化坐标转换为原图坐标，并进行置信度筛选。
    """
    def __init__(
        self,
        threshold: float = 0.5,
        img_size: Union[int, Tuple[int, int]] = 640,
        num_classes: int = 80,
        padding: bool = False, # 是否使用了 Letterbox padding
    ):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
            
        self.threshold = threshold
        self.num_classes = num_classes
        self.padding = padding

    @torch.no_grad()
    def __call__(self, outputs, original_img_size=None):
        """
        Args:
            outputs: 模型的原始输出，通常是一个字典或元组
                     期望包含:
                     - 'pred_logits': (B, N, num_classes)
                     - 'pred_boxes': (B, N, 4) [cx, cy, w, h] 归一化坐标
            original_img_size: (w, h) 原图尺寸
        Returns:
            cls_res: (M,) 类别索引
            score_res: (M,) 置信度
            bbox_res: (M, 4) 原图坐标 [x1, y1, x2, y2]
        """
        # 解析输出
        # RT-DETR 输出通常是字典，或者 tuple (feats, fused, head_out)
        if isinstance(outputs, (list, tuple)):
            # 尝试取最后一个作为 head output
            outputs = outputs[-1]
        
        if isinstance(outputs, dict):
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
        else:
            # 假设是 Tensor 列表，按顺序 [logits, boxes]
            # 这取决于你的模型具体返回什么，这里做个兼容性假设
            pred_logits = outputs[0]
            pred_boxes = outputs[1]

        # 假设 batch_size = 1，取第一张图
        # 如果是批量推理，需要外部循环调用
        if pred_logits.ndim == 3:
            pred_logits = pred_logits[0]
            pred_boxes = pred_boxes[0]

        # 计算分数
        # RT-DETR 通常使用 Sigmoid
        scores = pred_logits.sigmoid()
        
        # 筛选 (Top-K 或 Threshold)
        # 获取每个 Query 的最大分数和对应类别
        max_scores, labels = scores.max(dim=-1)
        
        # 阈值筛选
        keep = max_scores > self.threshold
        
        score_res = max_scores[keep]
        cls_res = labels[keep]
        box_res_norm = pred_boxes[keep] # [cx, cy, w, h] 归一化

        # 坐标转换: cxcywh (norm) -> xyxy (norm)
        box_res_norm = cxcywh2xyxy(box_res_norm)

        # 坐标还原到原图
        bbox_res = self._rescale_bboxes(box_res_norm, original_img_size)

        return cls_res, score_res, bbox_res

    def _rescale_bboxes(self, box_res_norm, original_img_size):
        """
        将归一化坐标还原到原图坐标
        """
        if original_img_size is None:
            # 如果没给原图尺寸，默认还原到输入尺寸
            h, w = self.img_size
            scale_tensor = torch.tensor([w, h, w, h], device=box_res_norm.device)
            return box_res_norm * scale_tensor

        orig_w, orig_h = original_img_size
        input_w, input_h = self.img_size

        if self.padding:
            # === 模式 A: Letterbox (保持长宽比 + 填充) ===
            # 1. 先还原到 input_size (640x640)
            box_res = box_res_norm * torch.tensor([input_w, input_h, input_w, input_h],
                                                  device=box_res_norm.device)
            
            # 2. 去除 padding 并缩放回原图
            scale = min(input_w / orig_w, input_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_left = (input_w - new_w) // 2
            pad_top = (input_h - new_h) // 2
            
            box_res[:, [0, 2]] -= pad_left
            box_res[:, [1, 3]] -= pad_top
            box_res[:, [0, 2]] /= scale
            box_res[:, [1, 3]] /= scale
        else:
            # === 模式 B: Direct Resize (直接拉伸) ===
            # 直接乘以原图尺寸即可，因为 box_res_norm 是归一化的
            scale_tensor = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                                        device=box_res_norm.device)
            box_res = box_res_norm * scale_tensor

        # 限制坐标在原图范围内
        box_res[:, [0, 2]] = torch.clamp(box_res[:, [0, 2]], 0, orig_w)
        box_res[:, [1, 3]] = torch.clamp(box_res[:, [1, 3]], 0, orig_h)

        return box_res


class RTDETRResult:
    """
    RT-DETR 结果可视化类
    逻辑与 YoloResult 基本一致，复用了大部分代码结构。
    """
    def __init__(
        self,
        image: Union[str, Path, np.ndarray],
        cls_res: Union[torch.Tensor, np.ndarray],
        score_res: Union[torch.Tensor, np.ndarray],
        bbox_res: Union[torch.Tensor, np.ndarray],
        class_names: Sequence[str],
        conf_threshold: float = 0.0, # 如果 Decoder 已经筛选过，这里可以设为 0
        target_classes: Optional[Sequence[int]] = None,
        color_list: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        # 1. 处理图像
        if isinstance(image, (str, Path)):
            self.image = cv2.imread(str(image))
            if self.image is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            self.image = image.copy()
        else:
            raise TypeError("image 必须是文件路径或 NumPy 数组")

        # 2. 数据转 Numpy
        self.cls_res = self._to_numpy(cls_res)
        self.score_res = self._to_numpy(score_res)
        self.bbox_res = self._to_numpy(bbox_res)

        assert len(self.cls_res) == len(self.score_res) == len(self.bbox_res)

        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None

        # 3. 颜色配置
        if color_list is None:
            self.color_list = FIXED_COLOR_LIST
        else:
            self.color_list = list(color_list)
        self.color_map = self._get_color_map()

        # 4. 二次过滤 (可选)
        self._filter_detections()

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        elif isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def _filter_detections(self):
        """根据置信度和目标类别进行过滤"""
        keep = self.score_res >= self.conf_threshold
        if self.target_classes is not None:
            in_target = np.array([c in self.target_classes for c in self.cls_res])
            keep = keep & in_target
        
        # 如果 keep 是全 True，可以跳过索引操作以提速
        if not np.all(keep):
            idx = np.where(keep)[0]
            self.cls_res = self.cls_res[idx]
            self.score_res = self.score_res[idx]
            self.bbox_res = self.bbox_res[idx]

    def _get_color_map(self):
        color_map = {}
        num_colors = len(self.color_list)
        for i, name in enumerate(self.class_names):
            color_map[i] = self.color_list[i % num_colors]
        return color_map

    def _cls_color(self, cls_id):
        return self.color_map.get(int(cls_id), (0, 255, 0))

    @property
    def show(self):
        def _show(figsize=(12, 8), title="RT-DETR Detection"):
            img_draw = self.image.copy()

            for cls_id, score, (x1, y1, x2, y2) in zip(
                self.cls_res, self.score_res, self.bbox_res
            ):
                cls_id = int(cls_id)
                # 类别名称处理
                if 0 <= cls_id < len(self.class_names):
                    label = self.class_names[cls_id]
                else:
                    label = f"class{cls_id}"
                
                text = f"{label} {score:.2f}"
                color = self._cls_color(cls_id)

                # 绘制边框
                cv2.rectangle(
                    img_draw,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,)

                # 绘制标签背景
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(
                    img_draw,
                    (int(x1), int(y1) - 20),
                    (int(x1) + w, int(y1)),
                    color,
                    -1,)

                # 绘制文本
                cv2.putText(
                    img_draw,
                    text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0), # 黑色字体
                    1,
                    cv2.LINE_AA,)

            # OpenCV BGR -> RGB for Matplotlib
            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return _show

    def __call__(self):
        self.show()


class RTDETRSegDecoder(RTDETRDecoder):
    """
    RT-DETR 实例分割解码器
    继承自 RTDETRDecoder，增加了 Mask 生成和处理逻辑。
    """
    def __init__(
        self,
        threshold: float = 0.5,
        img_size: Union[int, Tuple[int, int]] = 640,
        num_classes: int = 80,
        padding: bool = False,
        mask_channels: int = 32,
        proto_size: Tuple[int, int] = (160, 160), # 原型 Mask 的尺寸
    ):
        # 初始化父类
        super().__init__(
            threshold=threshold,
            img_size=img_size,
            num_classes=num_classes,
            padding=padding
        )
        
        self.mask_channels = mask_channels
        self.proto_h, self.proto_w = proto_size

    @staticmethod
    def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        使用预测框裁剪 Mask，去除框外的噪声。
        Args:
            masks: (N, H, W)
            boxes: (N, 4) xyxy 格式
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        
        # 创建网格
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
        
        # 生成裁剪掩码
        keep = (r >= x1) * (r < x2) * (c >= y1) * (c < y2)
        
        return masks * keep

    @torch.no_grad()
    def __call__(
        self,
        outputs,           
        original_img_size, 
        batch_idx: int = 0,
    ):
        """
        Args:
            outputs: 包含检测和分割输出的字典/元组
                     期望包含: 'pred_logits', 'pred_boxes', 'pred_mask_coeffs', 'proto_masks'
            original_img_size: (w, h)
            batch_idx: 处理 batch 中的第几张图
        """
        orig_w, orig_h = original_img_size

        # 1. 解析输出 (兼容性处理)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
        
        # 提取检测部分
        pred_logits = outputs['pred_logits'][batch_idx] # (300, 80)
        pred_boxes = outputs['pred_boxes'][batch_idx]   # (300, 4)
        
        # 提取分割部分
        # pred_mask_coeffs: (B, 300, 32)
        # proto_masks: (B, 32, 160, 160)
        mask_coeffs = outputs['pred_mask_coeffs'][batch_idx] 
        proto_masks = outputs['proto_masks'][batch_idx]

        # 2. 检测解码与筛选
        # 复用父类逻辑，但需要手动实现筛选以获取 indices
        scores = pred_logits.sigmoid()
        max_scores, labels = scores.max(dim=-1)
        keep = max_scores > self.threshold
        
        score_res = max_scores[keep]
        cls_res = labels[keep]
        box_res_norm = pred_boxes[keep]
        
        # 关键：获取保留下来的 Query 的索引
        coeffs = mask_coeffs[keep] # (N_det, 32)

        # 如果没有检测到目标
        if len(score_res) == 0:
            return (
                cls_res,
                score_res,
                torch.zeros((0, 4)),
                np.empty((0, orig_h, orig_w), dtype=np.float32))

        # 3. 坐标还原 (复用父类方法)
        box_res_norm = cxcywh2xyxy(box_res_norm)
        bbox_res = self._rescale_bboxes(box_res_norm, original_img_size)

        # 4. 生成 Mask (Matrix Multiplication)
        # (N, 32) @ (32, 160*160) -> (N, 160*160)
        proto = proto_masks.view(self.mask_channels, -1)
        masks = (coeffs @ proto).sigmoid().view(-1, self.proto_h, self.proto_w)

        # 5. 上采样到 Input Size (640x640)
        masks_in = F.interpolate(
            masks.unsqueeze(1),
            size=self.img_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        # 6. 还原到原图尺寸 (处理 Padding)
        if self.padding:
            in_w, in_h = self.img_size
            scale = min(in_w / orig_w, in_h / orig_h)
            pad_w = (in_w - orig_w * scale) / 2
            pad_h = (in_h - orig_h * scale) / 2
            
            x_start = int(round(pad_w))
            y_start = int(round(pad_h))
            x_end = int(round(in_w - pad_w))
            y_end = int(round(in_h - pad_h))
            
            masks_cropped = masks_in[:, y_start:y_end, x_start:x_end]
            
            masks_img = F.interpolate(
                masks_cropped.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
        else:
            masks_img = F.interpolate(
                masks_in.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

        # 7. Crop Mask (使用预测框裁剪)
        masks_img = self.crop_mask(masks_img, bbox_res)

        # 8. 转 Numpy
        masks_np = masks_img.cpu().numpy().astype(np.float32)

        return cls_res, score_res, bbox_res, masks_np


class RTDETRSegResult(RTDETRResult):
    """
    RT-DETR 实例分割结果封装
    继承自 RTDETRResult，增加了 Mask 的可视化功能。
    """
    def __init__(
        self,
        image,
        cls_res,
        score_res,
        bbox_res,
        class_names,
        masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        conf_threshold: float = 0.0,
        target_classes: Optional[Sequence[int]] = None,
        show_bbox: bool = True,
        show_masks: bool = True,
        show_polygon: bool = False,
        mask_alpha: float = 0.4,
        color_list=None
    ):
        # 1. 预处理 Masks
        self.raw_masks = self._to_numpy(masks) if masks is not None else None
        
        # 2. 初始化父类 (会执行 _filter_detections)
        super().__init__(
            image, cls_res, score_res, bbox_res, class_names, 
            conf_threshold, target_classes, color_list
        )
        
        # 3. 保存可视化配置
        self.show_bbox_flag = show_bbox
        self.show_masks_flag = show_masks
        self.show_polygon_flag = show_polygon
        self.mask_alpha = mask_alpha

    def _filter_detections(self):
        """
        重写父类的过滤逻辑，确保 Mask 与 BBox 同步过滤。
        """
        keep = self.score_res >= self.conf_threshold
        if self.target_classes is not None:
            in_target = np.array([c in self.target_classes for c in self.cls_res])
            keep = keep & in_target
        
        # 过滤基础数据
        self.cls_res = self.cls_res[keep]
        self.score_res = self.score_res[keep]
        self.bbox_res = self.bbox_res[keep]
        
        # 过滤 Mask
        if self.raw_masks is not None:
            if len(self.raw_masks) == len(keep):
                self.masks = self.raw_masks[keep]
            else:
                # 容错处理：如果长度不一致，尝试截断
                min_len = min(len(self.raw_masks), len(keep))
                self.masks = self.raw_masks[:min_len][keep[:min_len]]
        else:
            self.masks = None

    def _draw_masks(self, img_draw):
        """绘制半透明 Mask"""
        if self.masks is None: return img_draw
        
        overlay = img_draw.copy().astype(np.float32)
        
        for cls, mask in zip(self.cls_res, self.masks):
            color = np.array(self._cls_color(int(cls)), dtype=np.float32)
            # 二值化 Mask
            mask_bin = (mask > 0.5)[..., None]
            
            # 混合
            overlay = overlay * (1 - mask_bin * self.mask_alpha) + \
                      color * (mask_bin * self.mask_alpha)
                      
        return overlay.astype(np.uint8)

    def _draw_polygons(self, img_draw):
        """绘制多边形轮廓"""
        if self.masks is None: return img_draw
        
        h, w = img_draw.shape[:2]
        
        for cls, mask in zip(self.cls_res, self.masks):
            # 确保 mask 尺寸正确
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 获取轮廓 (需要 mask2polygon 函数)
            # 如果没有 mask2polygon，可以用 cv2.findContours
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color = self._cls_color(int(cls))
            cv2.drawContours(img_draw, contours, -1, color, 2)
            
        return img_draw

    @property
    def show(self):
        """
        重写 show 方法，增加 Mask 可视化
        """
        def _show(figsize=(12, 8), title="RT-DETR Segmentation"):
            img_draw = self.image.copy()

            # 1. 绘制 Mask
            if self.show_masks_flag:
                img_draw = self._draw_masks(img_draw)

            # 2. 绘制 Polygon
            if self.show_polygon_flag:
                img_draw = self._draw_polygons(img_draw)

            # 3. 绘制 BBox (如果开启)
            if self.show_bbox_flag:
                for cls_id, score, (x1, y1, x2, y2) in zip(
                    self.cls_res, self.score_res, self.bbox_res
                ):
                    cls_id = int(cls_id)
                    color = self._cls_color(cls_id)
                    
                    cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    label = (self.class_names[cls_id] \
                             if cls_id < len(self.class_names) else f"class{cls_id}")
                    text = f"{label} {score:.2f}"
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    cv2.rectangle(
                        img_draw, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), color, -1)
                    cv2.putText(
                        img_draw,
                        text,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA)

            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return _show


if __name__ == "__main__":
    # 1. 模拟模型输出
    # 假设 Batch=1, 300 Queries, 80 Classes, 32 Mask Channels
    outputs = {
        'pred_logits': torch.randn(1, 300, 80),
        'pred_boxes': torch.rand(1, 300, 4),
        'pred_mask_coeffs': torch.randn(1, 300, 32),
        'proto_masks': torch.randn(1, 32, 160, 160)
    }
    
    # 2. 初始化分割解码器
    seg_decoder = RTDETRSegDecoder(
        threshold=0.5,
        img_size=640,
        num_classes=80,
        padding=False
    )
    
    # 3. 解码
    orig_w, orig_h = 1280, 720
    cls_ids, scores, bboxes, masks = seg_decoder(outputs, (orig_w, orig_h))
    
    print(f"Detected: {len(cls_ids)}")
    print(f"Masks shape: {masks.shape}") # 应为 (N, 720, 1280)

    # 4. 可视化
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    class_names = [f"c{i}" for i in range(80)]
    
    result = RTDETRSegResult(
        image=dummy_img,
        cls_res=cls_ids,
        score_res=scores,
        bbox_res=bboxes,
        masks=masks,
        class_names=class_names,
        show_masks=True,
        show_polygon=True
    )
    
    # result.show()