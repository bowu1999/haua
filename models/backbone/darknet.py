import torch
import torch.nn as nn

from ..block import Conv, C3k2, SPPF, C2PSA


_MODEL_CONFIGS = {
    "n": {
        "depth_mult": 0.50,
        "width_mult": 0.25,
        "num_blocks": [1, 1, 1, 1],
        "block_args": [(False, 0.25), (False, 0.25), (True,), (True,)],
        "max_channels": 1024},
    "s": {
        "depth_mult": 0.50,
        "width_mult": 0.50,
        "num_blocks": [1, 1, 1, 1],
        "block_args": [(False, 0.25), (False, 0.25), (True,), (True,)],
        "max_channels": 1024},
    "m": {
        "depth_mult": 0.50,
        "width_mult": 1.00,
        "num_blocks": [1, 1, 1, 1],
        "block_args": [(True, 0.25), (True, 0.25), (True,), (True,)],
        "max_channels": 512},
    "l": {
        "depth_mult": 1.00,
        "width_mult": 1.00,
        "num_blocks": [2, 2, 2, 2],
        "block_args": [(True, 0.25), (True, 0.25), (True,), (True,)],
        "max_channels": 512},
    "x": {
        "depth_mult": 1.00,
        "width_mult": 1.50,
        "num_blocks": [2, 2, 2, 2],
        "block_args": [(True, 0.25), (True, 0.25), (True,), (True,)],
        "max_channels": 512}}


def make_divisible(x, divisor=8):
    """将通道数调整为8的倍数"""
    return int((x + divisor / 2) // divisor * divisor)


def build_layer(in_channels, out_channels, num_blocks, block_args, downsample=True, dim_increase=False):
    """构建 Darknet 的层，包含下采样 Conv + C3k2 块堆叠"""
    layers = []
    c3k = block_args[0]
    expansion = block_args[1] if len(block_args) > 1 else 0.5
    if downsample:
        layers.append(
            Conv(
                in_channels = in_channels,
                out_channels = in_channels if not dim_increase else out_channels,
                kernel_size = 3,
                stride = 2))
    layers.append(
        C3k2(
            in_channels = in_channels if not dim_increase else out_channels,
            out_channels = out_channels,
            num_blocks = num_blocks,
            c3k = c3k,
            expansion = expansion))
    return nn.Sequential(*layers)


class Darknet(nn.Module):

    base_channels = [128, 256, 512, 512, 1024] # yolo11 L 表示标准

    def __init__(self, model_type="s", in_channels=3):
        super().__init__()
        assert model_type in _MODEL_CONFIGS, f"Unsupported type: {model_type}"
        cfg = _MODEL_CONFIGS[model_type]
        self.model_channels = [make_divisible(c * cfg["width_mult"]) for c in self.base_channels]
        if self.model_channels[-1] > cfg["max_channels"]:
            self.model_channels[-1] = int(self.model_channels[-1] // 2)
        stem_channels = int(self.model_channels[0] // 2)
        self.out_channels = self.model_channels[-3:]
        self.stem = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=1),
            Conv(
                in_channels = stem_channels,
                out_channels = self.model_channels[0],
                kernel_size = 3,
                stride = 2))
        self.layer1 = build_layer(
            in_channels = self.model_channels[0],
            out_channels = self.model_channels[1],
            num_blocks = cfg["num_blocks"][0],
            block_args = cfg["block_args"][0],
            downsample = False)
        self.layer2 = build_layer(
            in_channels = self.model_channels[1],
            out_channels = self.model_channels[2],
            num_blocks = cfg["num_blocks"][1],
            block_args = cfg["block_args"][1])
        self.layer3 = build_layer(
            in_channels = self.model_channels[2],
            out_channels = self.model_channels[3],
            num_blocks = cfg["num_blocks"][2],
            block_args = cfg["block_args"][2])
        self.layer4 = build_layer(
            in_channels = self.model_channels[3],
            out_channels = self.model_channels[4],
            num_blocks = cfg["num_blocks"][3],
            block_args = cfg["block_args"][3],
            dim_increase = True)
        self.sppf = SPPF(in_channels=self.model_channels[4], out_channels=self.model_channels[4])
        self.c2psa = C2PSA(in_channels=self.model_channels[4], out_channels=self.model_channels[4])

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x5 = self.sppf(x5)
        x5 = self.c2psa(x5)

        return x3, x4, x5
