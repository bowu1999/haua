from typing import List, Optional, Union

import math
import copy
import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ..block import get_activation
from ...utils.utility_function import inverseSigmoid


__all__ = [
    'MLP',
    'DETRTransformerEncoder',
    'DETRTransformerEncoderLayer',
    'DETRTransformerDecoder',
    'MSDeformableAttention',
    'DETRTransformerDecoderLayer'
]


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron, MLP）模块

    该模块由若干个全连接层（Linear）级联构成，用于对特征进行逐层非线性映射，常用于：
        - Transformer 中的 FFN（Feed-Forward Network）
        - DETR / RT-DETR 中的回归头（bbox / cls）
        - 特征投影（embedding projection）
        - 查询（query）或隐状态的特征变换

    Module structure description:
        - 总层数由 `num_layers` 指定
        - 前 `num_layers - 1` 层：
              Linear → Activation
        - 最后一层：
              Linear（不使用激活函数）

    这种设计符合深度学习中的常见约定：
        - 中间层引入非线性以增强表达能力
        - 输出层保持线性，便于后续数值建模
          （如回归、logits 计算等）

    Args:
        input_dim (int): 输入特征的维度
        hidden_dim (int): 隐藏层特征维度，当 num_layers > 1 时，所有中间层均使用该维度
        output_dim (int): 输出特征的维度
        num_layers (int): MLP 的总层数（Linear 层数量）
            - num_layers = 1：等价于一个单层 Linear(input_dim → output_dim)
            - num_layers > 1：input_dim → hidden_dim → ... → hidden_dim → output_dim
        act (str | nn.Module): 激活函数类型，用于中间层，通过 `get_activation` 获取具体激活模块。
            常见取值如：
                - 'relu'
                - 'gelu'
                - 'silu'
                - 'leaky_relu'

    Property Description:
        self.layers (nn.ModuleList):
            按顺序存放所有 Linear 层。
            使用 ModuleList 以确保参数被正确注册。

        self.act (Callable):
            激活函数实例，仅作用于非最后一层。

    Froward:
        - 依次遍历所有 Linear 层
        - 对于非最后一层：
              x = act(Linear(x))
        - 对于最后一层：
              x = Linear(x)
        - 返回最终输出

    Design features and engineering considerations:
        1. 不在最后一层使用激活函数：
           - 保证输出分布的线性可控性
           - 适用于回归（bbox）、logits 等任务

        2. 使用 ModuleList 而非 Sequential：
           - 便于对层数和激活位置进行精细控制
           - 更适合 Transformer / DETR 类模型的编码风格

        3. 结构简洁、可复用性高：
           - 可作为通用 MLP 模板
           - 在检测、分割、Transformer 等任务中通用

    Example:
        >>> mlp = MLP(
        ...     input_dim=256,
        ...     hidden_dim=256,
        ...     output_dim=4,
        ...     num_layers=3,
        ...     act='relu'
        ... )
        >>> x = torch.randn(8, 100, 256)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([8, 100, 4])
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x) # type: ignore
        return x


class DETRTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DETRTransformerEncoderLayer(nn.Module):
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, nn.Module] = "relu",
        normalize_before: bool = False
    ):
        super().__init__()
        self.normalize_before = normalize_before

        # Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # Feed Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norms & Dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)() 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Self Attention Block ---
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        
        # 加入位置编码 (Q 和 K 需要位置信息，V 不需要)
        q = k = self.with_pos_embed(src, pos_embed)
        
        # value 使用的是 src (可能经过了 norm，也可能没有)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        # --- Feed Forward Block ---
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        
        # FFN 计算: Linear -> Act -> Dropout -> Linear
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)

        return src


class DETRTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
        target,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask = None,
        memory_mask = None
    ):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                attn_mask,
                memory_mask,
                query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverseSigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverseSigmoid(ref_points))) # type: ignore

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method='default'):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape
        
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1

    elif method == 'discrete':
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1) # type: ignore
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == 'default':
            sampling_value_l = F.grid_sample(
                value_l, 
                sampling_grid_l, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False)
        
        elif method == 'discrete':
            # n * m, seq, n, 2
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1) 
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2) 

            s_idx = torch.arange(
                sampling_coord.shape[0],
                device = value.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1])
            sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]] # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level])
        
        sampling_value_list.append(sampling_value_l) # type: ignore

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    def __init__(self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            'num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2. * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat(
            [torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int],
        value_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], 
                                        range in [0, 1], 
                                        top-left (0,0), 
                                        bottom-right (1, 1), 
                                        including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], 
                                        [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], 
                                - True for non-padding elements,
                                - False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + \
                sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * \
                self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list)

        output = self.output_proj(output)

        return output


class DETRTransformerDecoderLayer(nn.Module):
    def __init__(self,
        d_model = 256,
        n_head = 8,
        dim_feedforward = 1024,
        dropout = 0.,
        activation = 'relu',
        n_levels = 4,
        n_points = 4,
        cross_attn_method = 'default'
    ):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt)))) # type: ignore

    def forward(self,
        target,
        reference_points,
        memory,
        memory_spatial_shapes,
        attn_mask = None,
        memory_mask = None,
        query_pos_embed = None
    ):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target