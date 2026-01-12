from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...block import ConvBNAct, UpsampleModule
from ..detector import YOLODetector


class PrototypeSegmenter(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int]):
        super().__init__()
        self.proto = nn.Sequential(
            ConvBNAct(in_channels[0], 64, kernel_size=3, stride=1),
            UpsampleModule(64, 64),
            ConvBNAct(64, 64, kernel_size=3, stride=1),
            ConvBNAct(64, 32, kernel_size=3, stride=1))
        self.module_list = nn.ModuleList()
        for i_c in in_channels:
            self.module_list.append(
                nn.Sequential(
                    ConvBNAct(i_c, 32, kernel_size=3, stride=1),
                    ConvBNAct(32, 32, kernel_size=3, stride=1),
                    nn.Conv2d(32, 32, kernel_size=1, stride=1)))
    
    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        prototype_mask = self.proto(feats[0])
        outs = ()
        for i, module in enumerate(self.module_list):
            feat = module(feats[i])       # (B, 32, H, W)
            B, C, H, W = feat.shape
            feat = feat.view(B, C, -1)   # (B, 32, H*W)
            outs += (feat,)

        return prototype_mask, outs


class YOLOSegmenter(nn.Module):
    def __init__(
        self,
        in_channels_list: Tuple[int, int, int],
        num_classes: int,
        locate_hidden_channels: Union[Tuple[int, int, int], List[int], int],
        classify_hidden_channels: Union[Tuple[int, int, int], List[int], int],
    ):
        super().__init__()
        self.segmentation = PrototypeSegmenter(in_channels=in_channels_list)
        self.detector = YOLODetector(
            in_channels_list = in_channels_list,
            num_classes = num_classes,
            locate_hidden_channels = locate_hidden_channels,
            classify_hidden_channels = classify_hidden_channels)
    
    def forward(self, feats: List[torch.Tensor]):
        det_outs = self.detector(feats)
        prototype_mask, seg_outs = self.segmentation(feats)

        return det_outs, seg_outs, prototype_mask