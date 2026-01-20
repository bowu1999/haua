from typing import List, Optional, Union

import numpy as np
from PIL import Image

import torch


def tensor2image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    undo_std: bool = False,
    undo_norm: bool = True,
    show: bool = False,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    将图像 Tensor 转换为 PIL Image，支持可选的反标准化和反归一化

    Args:
        tensor (torch.Tensor): 输入图像 Tensor，形状为 (C, H, W) 或 (1, C, H, W)
        mean (list): 标准化时使用的均值，默认 ImageNet 均值
        std (list): 标准化时使用的标准差，默认 ImageNet 标准差
        undo_std (bool): 是否进行反标准化 (Reverse Standardization)
                         如果你的 Tensor 经过了 (x-mean)/std 处理，请设为 True
        undo_norm (bool): 是否显式进行反归一化 (Reverse Normalization) 即 x*255
                    - True: 手动乘 255 并转为 uint8
                    - False: 保持 float [0,1]，让 PIL 自动处理 (通常效果一样，但 True 能更精确控制类型)
        show (bool): 是否调用 image.show() 显示图片
        save_path (str, optional): 图片保存路径

    Returns:
        PIL.Image.Image: 转换后的 PIL 图像。
    """
    # 确保 Tensor 在 CPU 上，并不计算梯度，且不改变原 Tensor
    tensor = tensor.detach().cpu().clone()

    # 处理 Batch 维度: 如果是 (1, C, H, W) -> (C, H, W)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # 反标准化 (Reverse Standardization): x = x * std + mean
    if undo_std:
        # 将 mean 和 std 转换为 (3, 1, 1) 以便广播
        dtype = tensor.dtype
        device = tensor.device
        mean_t = torch.as_tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
        std_t = torch.as_tensor(std, dtype=dtype, device=device).view(-1, 1, 1)
        
        tensor.mul_(std_t).add_(mean_t)  # In-place 操作节省内存

    # 反归一化 (Reverse Normalization) 与 类型转换
    if undo_norm:
        # 显式反归一化: [0, 1] -> [0, 255] -> uint8
        tensor = tensor.mul_(255).clamp_(0, 255).to(torch.uint8)
        # (C, H, W) -> (H, W, C) -> Numpy -> PIL
        # 这种方式比 transforms.ToPILImage 更直观可控
        img_np = tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray(img_np)
    else:
        # 保持 float [0, 1] 范围，限制数值防止溢出
        tensor = tensor.clamp_(0, 1)
        # (C, H, W) -> (H, W, C) -> Numpy (float32)
        # PIL.Image.fromarray 支持 float32 输入，但通常需要 uint8 才能正常保存/显示颜色
        # 这里为了通用性，我们还是利用 Numpy 转 uint8，或者直接用 PIL 的模式
        # 最稳妥的方法是让 PIL 处理 float -> int 的量化
        img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(img_np)

    # 展示与保存
    if show:
        image.show()
    if save_path:
        image.save(save_path)

    return image