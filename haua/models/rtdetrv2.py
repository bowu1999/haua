from typing import List, Any, Optional, Callable

from ._basemodel import BaseModel
from .backbone import Darknet
from .neck import HybridEncoder
from .head import RTDETRTransformerv2


__all__ = ['RTDETRv2']


class RTDETRv2(BaseModel):
    def __init__(
        self,
        num_classes: int = 80,
        feat_strides: List[int] = [8, 16, 32],
        hidden_dim: int = 256,
        num_encoder_layers: int = 1,
        num_queries: int = 100,
        num_denoising: int = 100,
        custom_postprocess: Optional[Callable[[Any], Any]] = None
    ):
        module_configs = {
            "backbone": {
                "layer_channels": (128, 256, 512, 512, 1024),
                "num_blocks": (2, 2, 2, 2, 2),
                "block_args": [(True, .25), (True, .25), (True,), (True,)]},
            "neck": {
                "in_channels": [512, 512, 1024],
                "feat_strides": feat_strides,
                "hidden_dim": hidden_dim,
                "use_encoder_idx": [2],
                "num_encoder_layers": num_encoder_layers,
                "nhead": 8,
                "dim_feedforward": 1024,
                "dropout": 0.,
                "enc_act": 'gelu',
                # cross
                "expansion": 1.0,
                "depth_mult": 1,
                "act": 'silu',},
            "head": {
                "num_classes": num_classes,
                "feat_channels": [hidden_dim, hidden_dim, hidden_dim],
                "feat_strides": feat_strides,
                "hidden_dim": hidden_dim,
                "num_levels": 3,
                "num_layers": 6,
                "num_queries": num_queries,
                "num_denoising": num_denoising,
                "label_noise_ratio": 0.5,
                "box_noise_scale": 1.0, # 1.0 0.4
                "eval_idx": -1,
                # NEW
                "num_points": [4, 4, 4], # [3,3,3] [2,2,2]
                "cross_attn_method": "default", # default, discrete
                "query_select_method": "default"}} # default, agnostic
        super().__init__(
            Darknet, HybridEncoder, RTDETRTransformerv2, custom_postprocess, module_configs)
