import random

import torch
import torch.nn.functional as F


__all__ = [
    'BaseCollateFunction', 
    'BatchImageCollateFuncion',
    'batch_image_collate_fn'
]


def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else 0

    def __call__(self, items):
        raise NotImplementedError('')


class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

    def __call__(self, items):
        """
        items: List of (image, target)
        image: Tensor [C, H, W]
        target: Dict
        """
        # 堆叠图像
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # 多尺度训练逻辑
        if self.scales is not None and self.epoch < self.stop_epoch:
            # 随机选择一个尺度
            sz = random.choice(self.scales)
            
            # 对图像进行插值 (Bilinear)
            images = F.interpolate(images, size=sz, mode='bilinear', align_corners=False)
            
            # 对 Mask 进行插值 (Nearest) - 修复了 NotImplementedError
            if 'masks' in targets[0]:
                for tg in targets:
                    if tg['masks'].numel() > 0:
                        # masks: [N, H, W] -> [N, 1, H, W] (插值需要 4D 或 5D)
                        m = tg['masks'].unsqueeze(1).float()
                        # 使用 nearest 保持二值特性
                        m = F.interpolate(m, size=sz, mode='nearest')
                        # 还原: [N, 1, H, W] -> [N, H, W]
                        tg['masks'] = m.squeeze(1).to(tg['masks'].dtype)
                    else:
                        # 处理空 mask 的情况，重新生成对应尺寸的空 tensor
                        tg['masks'] = torch.zeros(
                            (0, sz, sz), dtype=tg['masks'].dtype, device=tg['masks'].device)
            
            # 更新 target 中的 size 信息 (RT-DETR 需要)
            for tg in targets:
                tg['size'] = torch.tensor([sz, sz], device=images.device)

        return images, targets