from typing import Any, Union, Tuple, Optional, Sequence

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from ..utils import make_grid


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
    综合解码检测和分割结果：
      输入:
        det_outs: tuple of 3 tensors, each (B, 4*reg_max+cls_num, H, W)
        seg_outs: tuple of 3 tensors, each (B, C_mask=32, N_i)  其中 N_i = H_i*W_i
        prototype_mask: (B, C_mask=32, Hp, Wp)
        original_img_size: (orig_w, orig_h)

      输出:
        cls_res:   (N,) tensor
        score_res: (N,) tensor
        bbox_res:  (N,4) tensor (原图坐标)
        masks_np:  (N, orig_h, orig_w) numpy float32 in [0,1]
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
        # 内部创建一个 YOLODecoder，用于目标检测解码
        self.det_decoder = YOLODecoder(
            threshold=threshold,
            strides=list(strides),
            img_size=img_size,
            reg_max=reg_max,
            cls_num=cls_num,
            prob_fn=prob_fn,
            return_indices=True,  # 要拿到 anchor_idx
            padding=padding)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.mask_channels = mask_channels
        self.proto_h, self.proto_w = proto_size
        self.padding = padding

    @torch.no_grad()
    def __call__(
        self,
        det_outs,          # tuple of 3 tensors, each (B, 4*reg_max+cls_num, H, W)
        seg_outs,          # tuple of 3 tensors, each (B, C_mask, N_i)
        prototype_mask,    # (B, C_mask, Hp, Wp)
        original_img_size, # (orig_w, orig_h)
        batch_idx: int = 0,
    ):
        """
        只解码 batch 中一张图（batch_idx），返回这一张图的检测 + 实例分割结果。
        """
        device = prototype_mask.device
        orig_w, orig_h = original_img_size

        # YOLODecoder 解析检测
        cls_res, score_res, bbox_res, anchor_idx_res = self.det_decoder( # type: ignore
            det_outs, original_img_size)  # cls_res, score_res, bbox_res: 已是原图坐标

        # 如果 batch_size > 1，YOLODecoder 当前实现是把 B,N_tot flatten 在一起，
        # anchor_idx_res 是全局索引。此处假设 B=1 的推理场景，使用起来最简单。
        # 如果将来需要真正 batch 多图推理，建议在 YOLODecoder 中按 batch 分别处理。

        # 将 seg_outs 拼成 (B, C_mask, N_tot)
        # seg_outs: (P3,P4,P5) each (B,C,N_i)
        seg_list = []
        for p in seg_outs:
            assert p.dim() == 3 and p.shape[1] == self.mask_channels, \
                f"seg_out 的形状应为 (B,{self.mask_channels},N_i)，当前为 {p.shape}"
            seg_list.append(p)
        seg_all = torch.cat(seg_list, dim=2)  # (B, C_mask, N_tot)

        # 取出该 batch 的 seg_flat: (C_mask, N_tot)
        seg_flat = seg_all[batch_idx]         # (C_mask, N_tot)
        C_mask, N_tot = seg_flat.shape
        assert C_mask == self.mask_channels

        # 根据 anchor_idx_res 为每个检测框取 coeff: (N_det, C_mask)
        anchor_idx_res = anchor_idx_res.to(device)
        coeffs = seg_flat[:, anchor_idx_res]  # (C_mask, N_det)
        coeffs = coeffs.permute(1, 0)         # (N_det, C_mask)

        # 用 prototype_mask 生成 proto 尺度实例 mask
        proto = prototype_mask[batch_idx]     # (C_mask, Hp, Wp)
        Hp, Wp = proto.shape[1], proto.shape[2]
        N_det = coeffs.shape[0]

        coeffs = coeffs.view(N_det, C_mask, 1, 1)          # (N_det,C,1,1)
        proto_exp = proto.unsqueeze(0).expand(N_det, -1, -1, -1)  # (N_det,C,Hp,Wp)
        masks_proto = (coeffs * proto_exp).sum(dim=1)      # (N_det,Hp,Wp)
        masks_proto = masks_proto.sigmoid()

        # 将 mask 从 proto 尺度 resize 到网络输入尺寸 (例如 640x640)
        masks_in = F.interpolate(
            masks_proto.unsqueeze(1),         # (N_det,1,Hp,Wp)
            size=self.img_size,
            mode="bilinear",
            align_corners=False).squeeze(1)   # (N_det, H_in, W_in)

        # 7. 反 letterbox，映射回原图尺寸
        if self.padding:
            in_w, in_h = self.img_size
            img_w, img_h = orig_w, orig_h
            scale = min(in_w / img_w, in_h / img_h)
            new_w = img_w * scale
            new_h = img_h * scale
            pad_w = in_w - new_w
            pad_h = in_h - new_h
            pad_left = pad_w / 2
            pad_top = pad_h / 2

            x0 = int(round(pad_left))
            y0 = int(round(pad_top))
            x1_pad = int(round(pad_left + new_w))
            y1_pad = int(round(pad_top + new_h))

            masks_cropped = masks_in[:, y0:y1_pad, x0:x1_pad]  # (N_det, new_h, new_w)

            masks_img = F.interpolate(
                masks_cropped.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False).squeeze(1)  # (N_det, orig_h, orig_w)
        else:
            # 如果输入就等于原图尺寸而无 padding，则直接 resize
            masks_img = F.interpolate(
                masks_in.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False).squeeze(1)  # (N_det, orig_h, orig_w)

        masks_np = masks_img.cpu().numpy().astype(np.float32)
        masks_np = np.clip(masks_np, 0.0, 1.0)

        return cls_res, score_res, bbox_res, masks_np


class YoloSegResult(YoloResult):
    """
    在 YoloResult 的基础上，增加实例分割结果的可视化：
    - masks: (N, H, W) 或 list[np.ndarray(H, W)]，与 cls_res/bbox_res 对应
    - show_masks: 是否显示 mask
    - mask_alpha: mask 叠加透明度
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
        show_masks: bool = True,
        mask_alpha: float = 0.4,
        color_list: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        # 先调用父类构造，完成基础检测结果处理与过滤
        super().__init__(
            image=image,
            cls_res=cls_res,
            score_res=score_res,
            bbox_res=bbox_res,
            class_names=class_names,
            conf_threshold=conf_threshold,
            target_classes=target_classes,
            color_list=color_list)

        self.show_masks_flag = show_masks
        self.mask_alpha = mask_alpha

        # 处理 masks
        self.masks = None  # (N, H, W) float32 [0,1] or None

        if masks is not None:
            self._set_masks(masks)

        # 重要：需要和过滤后的检测结果对齐
        # 上面父类在 __init__ 里已经做了过滤，所以这里要根据过滤后的索引再同步 mask
        # 简单起见，在 _set_masks 里根据最终 self.cls_res 长度直接截断/对齐

    def _set_masks(
        self,
        masks: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
    ):
        """
        将外部传入的 masks 转为 (N, H, W) 的 numpy float32 数组，与当前 cls_res/bbox_res 对齐。
        假设：
        - 原 masks 的顺序与传入给 YoloSegResult 的 cls_res/score_res/bbox_res 顺序一致；
        - 过滤后（conf / target_classes），我们只保留前 len(self.cls_res) 个。
        """
        # 转 numpy
        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        elif isinstance(masks, np.ndarray):
            masks_np = masks
        else:
            # list of np.ndarray
            masks_np = np.stack(masks, axis=0)

        # 保证是 (N, H, W)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        assert masks_np.ndim == 3, f"masks shape 应为 (N,H,W)，当前为 {masks_np.shape}"

        # 根据当前过滤后的检测数量裁剪
        N_keep = len(self.cls_res)
        if masks_np.shape[0] < N_keep:
            # 理论上不应出现；出现说明输入不匹配
            raise ValueError(
                f"masks 数量 ({masks_np.shape[0]}) 小于检测数量 ({N_keep})")
        masks_np = masks_np[:N_keep]

        # 归一化到 [0,1]，以防输入为 0/255
        if masks_np.dtype != np.float32:
            masks_np = masks_np.astype(np.float32)
        if masks_np.max() > 1.0:
            masks_np = masks_np / 255.0

        self.masks = masks_np  # (N,H,W)

    def _draw_masks(self, img_draw):
        """
        在 img_draw 上叠加实例 mask（不是必须，需要 show_masks_flag=True）
        """
        if self.masks is None or not self.show_masks_flag:
            return img_draw

        h_img, w_img = img_draw.shape[:2]
        N, Hm, Wm = self.masks.shape

        # 如果 mask 尺寸和图像不一致，resize
        if (Hm, Wm) != (h_img, w_img):
            # 逐个 resize
            resized_masks = []
            for m in self.masks:
                m_resized = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                resized_masks.append(m_resized)
            masks = np.stack(resized_masks, axis=0)
        else:
            masks = self.masks

        # 叠加
        overlay = img_draw.copy()
        for idx, (cls_id, m) in enumerate(zip(self.cls_res, masks)):
            cls_id = int(cls_id)
            color = np.array(self._cls_color(cls_id), dtype=np.float32)

            # 创建 3 通道 mask
            m3 = np.expand_dims(m, axis=-1)  # (H,W,1)
            # 只在 mask > 0.5 的区域着色
            mask_bin = (m3 > 0.5).astype(np.float32)

            overlay = overlay.astype(np.float32)
            overlay = (
                overlay * (1 - mask_bin * self.mask_alpha) + color * (mask_bin * self.mask_alpha))

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

    @property
    def show(self):
        """
        覆盖父类 show：在绘制 bbox 的同时，按需绘制 mask。
        """
        def _show(figsize=(12, 8), title="YOLO Detection + Segmentation"):
            img_draw = self.image.copy() # type: ignore

            # 先画 mask（在原图上），再画 bbox
            img_draw = self._draw_masks(img_draw)

            # 再画 bbox 和 label
            for cls_id, score, (x1, y1, x2, y2) in zip(
                self.cls_res, self.score_res, self.bbox_res
            ):
                cls_id = int(cls_id)
                label = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f"class{cls_id}")
                text = f"{label} {score:.2f}"

                color = self._cls_color(cls_id)

                cv2.rectangle(
                    img_draw,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,)

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
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,)

            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return _show