from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import Conv, C3k2


class C3k2PAN(nn.Module):
    def __init__(
        self,
        in_channels: Union[List[int], Tuple[int, int, int]] = (256, 256, 512),
        out_channels: Optional[Union[List[int], Tuple[int, int, int]]] = (256, 256, 512),
        p_channels: List[int] = [512, 256, 128],
        c3_blocks: int = 1
    ):
        super().__init__()
        assert len(in_channels) == 3, "输入必须是三层特征图 (x3, x4, x5)"
        self.c3_channels, self.c4_channels, self.c5_channels = in_channels
        assert len(out_channels) == 3, "输出必须是三层特征图 (n3, n4, n5)"
        self.out3_channels, self.out4_channels, self.out5_channels = out_channels
        self.top2down_c3k2_f_4p4 = C3k2(
            in_channels = self.c5_channels + self.c4_channels,
            out_channels = p_channels[1],
            num_blocks = c3_blocks,
            c3k = False)
        self.top2down_c3k2_f_4p3 = C3k2(
            in_channels = self.c4_channels + p_channels[1],
            out_channels = p_channels[2],
            num_blocks = c3_blocks,
            c3k = False)
        self.bottom2up_downsample_conv_4p3 = Conv(
            in_channels = p_channels[2],
            out_channels = p_channels[2],
            kernel_size = 3,
            stride = 2,
            padding = 1)
        self.bottom2up_c3k2_f_4n4 = C3k2(
            in_channels = p_channels[1] + p_channels[2],
            out_channels = self.out4_channels,
            num_blocks = c3_blocks,
            c3k = False)
        self.bottom2up_downsample_conv_4n4 = Conv(
            in_channels = self.out4_channels,
            out_channels = self.out4_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1)
        self.bottom2up_c3k2_t_4n5 = C3k2(
            in_channels = self.c5_channels + self.out4_channels,
            out_channels = self.out5_channels,
            num_blocks = c3_blocks,
            c3k = True)

    def forward(self, features):
        """features: (x3: 80x80, x4: 40x40, x5: 20x20) Middle: (p3, p4, p5) Returns: (n3, n4, n5)"""
        assert len(features) == 3
        x3, x4, x5 = features
        # ---- top-down ----
        p5 = x5
        p5_upsampled = F.interpolate(x5, size=x4.shape[-2:], mode='nearest')
        p4 = self.top2down_c3k2_f_4p4(torch.cat([p5_upsampled, x4], dim=1))
        p4_upsampled = F.interpolate(p4, size=x3.shape[-2:], mode='nearest')
        p3 = self.top2down_c3k2_f_4p3(torch.cat([p4_upsampled, x3], dim=1))
        # ---- bottom-up ----
        n3 = p3
        p3_down = self.bottom2up_downsample_conv_4p3(p3)
        n4 = self.bottom2up_c3k2_f_4n4(torch.cat([p3_down, p4], dim=1))
        n4_down = self.bottom2up_downsample_conv_4n4(n4)
        n5 = self.bottom2up_c3k2_t_4n5(torch.cat([n4_down, p5], dim=1))

        return n3, n4, n5