from typing import Any, Union, Tuple, Optional, Sequence

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from ..utils import make_grid
from ...utils import mask2polygon


FIXED_COLOR_LIST = [
    (255, 0, 0),     # 红
    (0, 255, 0),     # 绿
    (0, 0, 255),     # 蓝
    (255, 255, 0),   # 青
    (255, 0, 255),   # 品红
    (0, 255, 255),   # 黄
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (64, 64, 0),
    (64, 0, 64),
    (0, 64, 64),
] + [(int(255 * np.random.rand()), int(255 * np.random.rand()), int(255 * np.random.rand()))
     for _ in range(100)]


class YOLODecoder:
    def __init__(
        self,
        threshold: float = 0.5,
        strides = [8, 16, 32],
        img_size = 640,
        reg_max: int = 16,
        cls_num: int = 80,
        prob_fn: str = "softmax",
        padding: bool = True,
        return_indices: bool = False,   # 新增：是否返回 anchor 索引
    ):
        if isinstance(img_size, tuple):
            self.img_size = img_size
        elif isinstance(img_size, int):
            self.img_size = (img_size,) * 2
        else:
            raise TypeError("img_size must be int or tuple")

        self.strides = list(strides)
        self.scales = [self.img_size[0] // s for s in self.strides]
        self.threshold = threshold
        self.reg_max = reg_max
        self.cls_num = cls_num
        self.channel = 4 * reg_max + cls_num
        self.prob_fn = prob_fn
        self.padding = padding
        self.return_indices = return_indices

    @torch.no_grad()
    def __call__(self, outputs, original_img_size=None):
        """
        outputs:
          - 原始版本支持两种形式：
            1) 直接是 3 个尺度的 tuple: (P3, P4, P5)，每个形状 (B, C, H, W)
            2) model 输出是个 tuple，最后一项才是 (P3,P4,P5)，这里通过
               if isinstance(outputs[-1], tuple): outputs = outputs[-1]
               自动取出最后一项。
        """
        # 保持你原来的兼容逻辑
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0 \
            and isinstance(outputs[-1], (list, tuple)):
            outputs = outputs[-1]

        assert isinstance(outputs, (list, tuple)) and len(outputs) == len(self.strides), \
            (
                f"outputs 必须是长度为 {len(self.strides)} 的 tuple/list，每一项是 (B,C,H,W)"
                f"当前类型 {type(outputs)}")

        # ==== 以下部分逻辑与原始版本保持一致，只是多记录 anchor 下标 ====
        all_results = []
        all_grids = []
        all_strides = []

        for i, p in enumerate(outputs):
            assert isinstance(p, torch.Tensor) and p.dim() == 4, \
                f"第{i}个输出必须是 4D tensor (B,C,H,W)，当前 {type(p)}，dim={p.dim()}"
            B, C, H, W = p.shape # type: ignore
            assert C == self.channel, \
                f"第{i}个输出通道数应为 {self.channel}，当前 {C}"
            N = H * W

            # (B,C,H,W) -> (B*H*W, C)
            all_results.append(p.permute(0, 2, 3, 1).reshape(-1, self.channel))

            # grid 仍按原来的 self.scales 来生成（即 H=W=self.scales[i]）
            grid = make_grid((self.scales[i],) * 2).view(-1, 2)  # (N_i,2)
            all_grids.append(grid)

            # stride map (N_i,)
            all_strides.append(
                torch.full((N,), self.strides[i], dtype=torch.float32, device=p.device))

        # 拼接所有尺度
        all_results = torch.cat(all_results, dim=0)  # (B*N_tot, C)
        all_grids = torch.cat(all_grids, dim=0)      # (N_tot, 2)
        all_strides = torch.cat(all_strides, dim=0)  # (N_tot,)

        # 与原来一样拆分 dfl 与 cls
        dfl_result, cls_result = torch.split(
            all_results, [4 * self.reg_max, self.cls_num], dim=1
        )

        if self.prob_fn == "softmax":
            cls_result = F.softmax(cls_result, dim=1)
        elif self.prob_fn == "sigmoid":
            cls_result = torch.sigmoid(cls_result)

        # 分类得分与标签
        cls_, indices = torch.max(cls_result, dim=1)  # (B*N_tot,)
        resindx = cls_ > self.threshold               # (B*N_tot,)

        # 根据掩码筛选正样本 / 保留目标
        dfl_res = dfl_result[resindx]
        grids_res = all_grids[resindx]        # 这里依旧假定 B=1 的场景与原始实现一致
        strides_res = all_strides[resindx]
        cls_res = indices[resindx]
        score_res = cls_[resindx]

        # 记录 anchor 索引（只相对于全部 anchors 的平面下标）
        anchor_idx_all = torch.arange(all_results.shape[0], device=all_results.device)
        anchor_idx_res = anchor_idx_all[resindx]

        # 解码 bbox
        bbox_res = self._decode_dfl(dfl_res, grids_res, strides_res)

        # 反 letterbox 到原图
        if self.padding and (original_img_size is not None) and (bbox_res.numel() > 0):
            input_w, input_h = self.img_size
            orig_w, orig_h = original_img_size

            scale = min(input_w / orig_w, input_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_left = (input_w - new_w) // 2
            pad_top = (input_h - new_h) // 2

            bbox_res[:, [0, 2]] -= pad_left
            bbox_res[:, [1, 3]] -= pad_top
            bbox_res[:, [0, 2]] /= scale
            bbox_res[:, [1, 3]] /= scale
            bbox_res[:, [0, 2]] = torch.clamp(bbox_res[:, [0, 2]], 0, orig_w)
            bbox_res[:, [1, 3]] = torch.clamp(bbox_res[:, [1, 3]], 0, orig_h)

        # 默认保持旧行为：只返回 cls/score/bbox
        if self.return_indices:
            return cls_res, score_res, bbox_res, anchor_idx_res
        else:
            return cls_res, score_res, bbox_res

    def _decode_dfl(self, dfl_logits, grid, strides):
        assert dfl_logits.shape[1] == 4 * self.reg_max, \
            f"Expected {4 * self.reg_max} dfl channels, got {dfl_logits.shape[1]}"
        assert grid.shape[1] == 2, "Grid must be (N, 2)"
        assert strides.shape[0] == dfl_logits.shape[0], \
            "Strides must have same length as dfl_logits"
        N = dfl_logits.shape[0]

        dfl_reshaped = dfl_logits.view(N, 4, self.reg_max)  # (N,4,reg_max)
        dfl_probs = dfl_reshaped.softmax(dim=-1)
        discrete_values = torch.arange(
            self.reg_max, dtype=dfl_logits.dtype, device=dfl_logits.device
        )
        offsets = (dfl_probs * discrete_values).sum(dim=-1)  # (N,4)

        strides_expanded = strides.unsqueeze(-1).expand_as(offsets)
        offsets_scaled = offsets * strides_expanded
        l, t, r, b = (
            offsets_scaled[:, 0],
            offsets_scaled[:, 1],
            offsets_scaled[:, 2],
            offsets_scaled[:, 3],
        )
        cx = grid[:, 0] * strides
        cy = grid[:, 1] * strides

        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return bboxes


class YoloResult:
    def __init__(
        self,
        image,
        cls_res,
        score_res,
        bbox_res,
        class_names,
        conf_threshold: float = 0.0,
        target_classes: Optional[Sequence[int]] = None,
        color_list: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        # 处理图像输入
        if isinstance(image, (str, Path)):
            self.image = cv2.imread(str(image))
            if self.image is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            self.image = image.copy()
            if not (self.image.ndim == 3 and self.image.shape[2] == 3):
                raise ValueError("图像必须是 HxWx3 的 NumPy 数组")
        else:
            raise TypeError("image 必须是文件路径或 NumPy 数组")

        self.cls_res = self._to_numpy(cls_res)
        self.score_res = self._to_numpy(score_res)
        self.bbox_res = self._to_numpy(bbox_res)

        assert len(self.cls_res) == len(self.score_res) == len(self.bbox_res)

        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None

        # 颜色：如果用户不给，使用固定列表
        if color_list is None:
            self.color_list = FIXED_COLOR_LIST
        else:
            self.color_list = list(color_list)

        self.color_map = self._get_color_map()

        # 过滤结果
        self._filter_detections()

    def _to_numpy(self, x):
        if hasattr(x, "cpu"):
            return x.cpu().detach().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    def _filter_detections(self):
        keep = self.score_res >= self.conf_threshold
        if self.target_classes is not None:
            in_target = np.array([c in self.target_classes for c in self.cls_res])
            keep = keep & in_target
        idx = np.where(keep)[0]
        self.cls_res = self.cls_res[idx]
        self.score_res = self.score_res[idx]
        self.bbox_res = self.bbox_res[idx]

    def _get_color_map(self):
        """为每个类别生成固定 BGR 颜色（从固定列表中取，循环使用）"""
        color_map = {}
        num_colors = len(self.color_list)
        for i, name in enumerate(self.class_names):
            color_map[i] = self.color_list[i % num_colors]
        return color_map

    def _cls_color(self, cls_id):
        """返回某个类别的固定 BGR 颜色"""
        return self.color_map.get(int(cls_id), (0, 255, 0))

    @property
    def show(self):
        def _show(figsize=(12, 8), title="YOLO Detection"):
            img_draw = self.image.copy() # type: ignore

            for cls_id, score, (x1, y1, x2, y2) in zip(
                self.cls_res, self.score_res, self.bbox_res
            ):
                cls_id = int(cls_id)
                label = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f"class{cls_id}"
                )
                text = f"{label} {score:.2f}"

                color = self._cls_color(cls_id)

                # 边框
                cv2.rectangle(
                    img_draw,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )

                # 标签背景
                (w, h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    img_draw,
                    (int(x1), int(y1) - 20),
                    (int(x1) + w, int(y1)),
                    color,
                    -1,
                )

                # 文本
                cv2.putText(
                    img_draw,
                    text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

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


class YoloSegDecoder:
    """
    YOLOv11 实例分割解码器 (优化版)
    包含:
    1. 目标检测解码 (调用 det_decoder)
    2. 实例分割 Mask 生成
    3. Mask 还原与 BBox 裁剪 (关键步骤)
    """

    def __init__(
        self,
        threshold=0.5,
        strides=(8, 16, 32),
        img_size=640,
        reg_max=16,
        cls_num=80,
        prob_fn="sigmoid",
        padding=True,
        mask_channels=32,
        proto_size=(160, 160),
    ):
        # 初始化检测解码器
        self.det_decoder = YOLODecoder(
            threshold=threshold,
            strides=list(strides),
            img_size=img_size,
            reg_max=reg_max,
            cls_num=cls_num,
            prob_fn=prob_fn,
            return_indices=True,  # 必须为 True，用于获取 mask 系数索引
            padding=padding
        )
        
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.mask_channels = mask_channels
        self.proto_h, self.proto_w = proto_size
        self.padding = padding

    @staticmethod
    def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        使用 BBox 裁剪 Mask (矢量化操作)
        Args:
            masks: (N, H, W)
            boxes: (N, 4) xyxy 格式
        Returns:
            cropped_masks: (N, H, W)
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        
        # 创建坐标网格 (复用 device 和 dtype)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows (x)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols (y)
        
        # 生成裁剪掩码: 像素点必须在 bbox 范围内
        # (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
        keep = (r >= x1) * (r < x2) * (c >= y1) * (c < y2)
        
        return masks * keep

    @torch.no_grad()
    def __call__(
        self,
        det_outs,          
        seg_outs,          
        prototype_mask,    
        original_img_size, 
        batch_idx: int = 0,
    ):
        """
        解码单张图片
        """
        device = prototype_mask.device
        orig_w, orig_h = original_img_size

        # 1. 解码检测结果
        # cls_res: (N,), score_res: (N,), bbox_res: (N, 4) [原图坐标], anchor_idx: (N,)
        cls_res, score_res, bbox_res, anchor_idx_res = self.det_decoder(
            det_outs, original_img_size
        )

        # 如果没有检测到目标，直接返回空
        if len(bbox_res) == 0:
            return cls_res, score_res, bbox_res, np.empty((0, orig_h, orig_w), dtype=np.float32)

        # 2. 准备 Mask 系数
        # 将 seg_outs (P3, P4, P5) 拼接 -> (B, 32, N_tot)
        seg_flat = torch.cat([p.flatten(2) for p in seg_outs], dim=2)[batch_idx] # (32, N_tot)
        
        # 根据 anchor 索引提取对应的 mask 系数
        # anchor_idx_res 是检测框对应的 anchor 索引
        coeffs = seg_flat[:, anchor_idx_res].T  # (N_det, 32)

        # 3. 生成原型 Mask (Matrix Multiplication)
        # coeffs: (N, 32) @ proto: (32, 160*160) -> (N, 160*160)
        proto = prototype_mask[batch_idx].view(self.mask_channels, -1) # (32, 25600)
        masks = (coeffs @ proto).sigmoid().view(-1, self.proto_h, self.proto_w) # (N, 160, 160)

        # 4. 上采样到网络输入尺寸 (例如 640x640)
        masks_in = F.interpolate(
            masks.unsqueeze(1),
            size=self.img_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1) # (N, 640, 640)

        # 5. 还原到原图尺寸 (Reverse Letterbox)
        if self.padding:
            in_w, in_h = self.img_size
            scale = min(in_w / orig_w, in_h / orig_h)
            
            # 计算 padding 区域
            pad_w = (in_w - orig_w * scale) / 2
            pad_h = (in_h - orig_h * scale) / 2
            
            # 裁剪掉 padding 部分
            # 注意：这里使用 round 确保和预处理对齐
            x_start = int(round(pad_w))
            y_start = int(round(pad_h))
            x_end = int(round(in_w - pad_w))
            y_end = int(round(in_h - pad_h))
            
            masks_cropped = masks_in[:, y_start:y_end, x_start:x_end]
            
            # Resize 到原图大小
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

        # 6. [关键步骤] 使用预测的 BBox 裁剪 Mask
        # bbox_res 已经是原图坐标，masks_img 也是原图尺寸，直接裁剪
        masks_img = self.crop_mask(masks_img, bbox_res)

        # 7. 转 Numpy 并二值化/截断
        masks_np = masks_img.cpu().numpy().astype(np.float32)
        # 通常这里不需要 clip，因为 sigmoid 已经在 0-1 之间，但为了保险保留
        # 如果需要二值化 mask (0 或 1)，可以在这里做: masks_np = (masks_np > 0.5).astype(np.float32)
        
        return cls_res, score_res, bbox_res, masks_np


class YoloSegResult(YoloResult):
    """
    在 YoloResult 的基础上，增加实例分割结果的可视化：
    - masks: (N, H, W) 与 cls_res/bbox_res 严格对应
    - 支持 mask, polygon, bbox 的显示控制
    """

    def __init__(
        self,
        image,
        cls_res,
        score_res,
        bbox_res,
        class_names,
        masks: Optional[Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]]] = None,
        conf_threshold: float = 0.0,
        target_classes: Optional[Sequence[int]] = None,
        show_bbox: bool = True,     # 新增：是否显示检测框
        show_masks: bool = True,    # 是否显示分割掩码
        show_polygon: bool = False, # 新增：是否显示多边形轮廓
        mask_alpha: float = 0.4,
        color_list: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        # -------------------------------------------------------
        # 1. 数据预处理与同步筛选 (核心修复部分)
        # -------------------------------------------------------
        
        # 统一转为 numpy 以便处理
        if isinstance(cls_res, torch.Tensor): cls_res = cls_res.cpu().numpy()
        if isinstance(score_res, torch.Tensor): score_res = score_res.cpu().numpy()
        if isinstance(bbox_res, torch.Tensor): bbox_res = bbox_res.cpu().numpy()
        
        # 处理 masks 转 numpy
        masks_np = None
        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            elif isinstance(masks, np.ndarray):
                masks_np = masks
            else:
                masks_np = np.stack(masks, axis=0)
            
            # 维度检查
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]
            
            # 归一化
            if masks_np.dtype != np.float32:
                masks_np = masks_np.astype(np.float32)
            if masks_np.max() > 1.0:
                masks_np = masks_np / 255.0

        # --- 生成筛选掩码 (Keep Mask) ---
        # 逻辑：同时满足 置信度阈值 和 目标类别
        keep_idxs = score_res >= conf_threshold
        
        if target_classes is not None:
            # 检查类别是否在目标列表中
            class_keep = np.isin(cls_res, target_classes)
            keep_idxs = keep_idxs & class_keep

        # --- 应用筛选到所有数据 ---
        # 这样 bbox, score, cls, masks 就永远保持一一对应了
        self.cls_res_filtered = cls_res[keep_idxs]
        self.score_res_filtered = score_res[keep_idxs]
        self.bbox_res_filtered = bbox_res[keep_idxs]
        
        if masks_np is not None:
            # 关键修复：使用相同的索引筛选 mask
            if len(masks_np) != len(score_res):
                 # 如果原始输入长度就不对，打印警告或报错
                 print(f"Warning: Input masks length {len(masks_np)} != scores length {len(score_res)}")
                 # 尝试截断或报错，这里假设输入是对应的
                 min_len = min(len(masks_np), len(keep_idxs))
                 self.masks = masks_np[:min_len][keep_idxs[:min_len]]
            else:
                self.masks = masks_np[keep_idxs]
        else:
            self.masks = None

        # -------------------------------------------------------
        # 2. 调用父类初始化
        # -------------------------------------------------------
        # 注意：因为我们已经手动筛选过了，传给父类的 conf_threshold 可以设为 0
        # 或者保持原样（再次筛选也不会有副作用，因为数据已经满足条件）
        super().__init__(
            image=image,
            cls_res=self.cls_res_filtered,
            score_res=self.score_res_filtered,
            bbox_res=self.bbox_res_filtered,
            class_names=class_names,
            conf_threshold=0.0, # 已在外部筛选，这里设为0避免重复逻辑干扰
            target_classes=None, # 已在外部筛选
            color_list=color_list
        )

        # -------------------------------------------------------
        # 3. 保存可视化配置
        # -------------------------------------------------------
        self.show_bbox_flag = show_bbox
        self.show_masks_flag = show_masks
        self.show_polygon_flag = show_polygon
        self.mask_alpha = mask_alpha

    def _draw_masks(self, img_draw):
        """绘制半透明 Mask"""
        if self.masks is None:
            return img_draw

        h_img, w_img = img_draw.shape[:2]
        
        # Resize masks if needed
        if (self.masks.shape[1], self.masks.shape[2]) != (h_img, w_img):
            resized_masks = []
            for m in self.masks:
                m_resized = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                resized_masks.append(m_resized)
            masks_to_draw = np.stack(resized_masks, axis=0)
        else:
            masks_to_draw = self.masks

        overlay = img_draw.copy()
        
        for idx, (cls_id, m) in enumerate(zip(self.cls_res, masks_to_draw)):
            cls_id = int(cls_id)
            color = np.array(self._cls_color(cls_id), dtype=np.float32)

            m3 = np.expand_dims(m, axis=-1)
            mask_bin = (m3 > 0.5).astype(np.float32)

            overlay = overlay.astype(np.float32)
            # 混合公式
            overlay = (
                overlay * (1 - mask_bin * self.mask_alpha) + 
                color * (mask_bin * self.mask_alpha)
            )

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def _draw_polygons(self, img_draw):
        """绘制多边形轮廓"""
        if self.masks is None:
            return img_draw
            
        h_img, w_img = img_draw.shape[:2]
        
        for idx, (cls_id, m) in enumerate(zip(self.cls_res, self.masks)):
            # Resize mask to image size for correct polygon coordinates
            if (m.shape[0], m.shape[1]) != (h_img, w_img):
                m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            
            # 获取轮廓
            contours = mask2polygon(m)
            
            cls_id = int(cls_id)
            color = self._cls_color(cls_id) # tuple (B, G, R)
            
            # 绘制轮廓
            cv2.drawContours(img_draw, contours, -1, color, 2) # thickness=2
            
        return img_draw

    @property
    def show(self):
        """
        可视化入口
        """
        def _show(figsize=(12, 8), title="YOLO Detection + Segmentation"):
            img_draw = self.image.copy() # type: ignore

            # 1. 绘制 Mask (填充)
            if self.show_masks_flag:
                img_draw = self._draw_masks(img_draw)

            # 2. 绘制 Polygon (轮廓)
            if self.show_polygon_flag:
                img_draw = self._draw_polygons(img_draw)

            # 3. 绘制 BBox 和 Label
            if self.show_bbox_flag:
                for cls_id, score, (x1, y1, x2, y2) in zip(
                    self.cls_res, self.score_res, self.bbox_res
                ):
                    cls_id = int(cls_id)
                    color = self._cls_color(cls_id)
                    
                    # 画框
                    cv2.rectangle(
                        img_draw,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,)

                    # 画标签背景和文字
                    label = (
                        self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else f"class{cls_id}")
                    text = f"{label} {score:.2f}"

                    (w, h), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    cv2.rectangle(
                        img_draw,
                        (int(x1), int(y1) - 20),
                        (int(x1) + w, int(y1)),
                        color,
                        -1,)

                    cv2.putText(
                        img_draw,
                        text,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0), # 黑色文字
                        1,
                        cv2.LINE_AA,)

            # 显示
            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return _show