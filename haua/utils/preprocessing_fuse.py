import torch
import torch.nn as nn
from typing import List, Optional


__all__ = ['fusePreprocessing2conv']


def fusePreprocessing2conv(
    model: nn.Module,
    use_normalization: bool = True,
    mean: Optional[List[float]] = None, 
    std: Optional[List[float]] = None
) -> bool:
    """
    寻找模型的第一层卷积，并将预处理（/255 和 标准化）融合进权重。
    """
    print("Attempting to fuse preprocessing into the first Conv2d layer...")
    # 自动寻找第一层卷积
    conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv = module
            print(f"   -> Found first Conv2d layer: {name}")
            break
    if conv is None:
        print("⚠️ Warning: No Conv2d found in the model. Skipping fusion.")
        return False

    # 准备参数
    if not use_normalization:
        target_mean = [0.0] * conv.in_channels
        target_std = [1.0] * conv.in_channels
    else:
        target_mean = mean if mean is not None else [0.485, 0.456, 0.406]
        target_std = std if std is not None else [0.229, 0.224, 0.225]

    if len(target_mean) != conv.in_channels or len(target_std) != conv.in_channels:
        print(
            f"Error: Mean/Std length mismatch, Expected {conv.in_channels}, got {len(target_mean)}")

        return False
    
    device = conv.weight.device
    dtype = conv.weight.dtype
    mean_t = torch.tensor(target_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(target_std, device=device, dtype=dtype).view(1, -1, 1, 1)

    # 执行融合
    with torch.no_grad():
        w_original = conv.weight.clone()
        b_original = conv.bias \
            if conv.bias is not None else torch.zeros(conv.out_channels, device=device, dtype=dtype)
        # W_new = W / (255 * std)
        w_new = w_original / (255.0 * std_t)
        # Bias_shift = - W * (mean / std)
        # 这一步把 -mean 做进去了
        constant_term = - mean_t / std_t
        bias_shift = (w_original * constant_term).sum(dim=(1, 2, 3))
        b_new = b_original + bias_shift
        conv.weight.copy_(w_new)
        if conv.bias is None:
            conv.bias = nn.Parameter(b_new)
        else:
            conv.bias.copy_(b_new)
    print("   -> Fusion complete.")

    return True