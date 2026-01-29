import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detector import RTDETRTransformerv2
from .yolo_segmenter import PrototypeSegmenter
from ....utils.utility_function import inverseSigmoid
from ..detector import getContrastiveDenoisingTrainingGroup
from ...block.transformer import MLP, DETRTransformerDecoder


__all__ = [
    'DETRTransformerDecoderWithMask',
    'RTDETRInstanceSegmenter'
]


class DETRTransformerDecoderWithMask(DETRTransformerDecoder):
    """
    继承原有的 DETRTransformerDecoder，重写 forward 以输出 Mask Coefficients。
    """
    def forward(self,
        target,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        mask_head, 
        attn_mask=None,
        memory_mask=None):
        
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_mask_coeffs = [] # 新增：存储每一层的 mask 系数

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

            # Box 和 Score 预测 (保持原逻辑)
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverseSigmoid(ref_points_detach))
            
            # Mask 系数预测 (新增逻辑)
            # mask_head[i] 是一个 MLP，将 hidden_dim 映射到 prototype_dim
            mask_coeffs = mask_head[i](output)

            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out_mask_coeffs.append(mask_coeffs) # 收集系数
                
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverseSigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_mask_coeffs.append(mask_coeffs) # 推理时也需要
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_mask_coeffs))


class RTDETRInstanceSegmenter(RTDETRTransformerv2):
    def __init__(self, 
        feat_channels=[512, 1024, 2048], 
        prototype_dim=32, 
        **kwargs):
        """
        Args:
            feat_channels: Backbone 输出特征层的通道数
            prototype_dim: Mask 原型的通道维度 (默认 32)
            **kwargs: 传递给 RTDETRTransformerv2 的其他参数
        """
        super().__init__(feat_channels=feat_channels, **kwargs)
        
        self.prototype_dim = prototype_dim

        # 1. 初始化 Pixel Decoder (Prototype Segmenter)
        # 注意：PrototypeSegmenter 期望输入是一个 tuple
        self.pixel_decoder = PrototypeSegmenter(tuple(feat_channels), prototype_dim=prototype_dim)

        # 2. 初始化 Mask Head (用于 Decoder)
        # 为每一层 Decoder 创建一个 MLP，将 Query Embedding 映射为 Mask 系数
        self.dec_mask_head = nn.ModuleList([
            MLP(self.hidden_dim, self.hidden_dim, prototype_dim, 3) 
            for _ in range(self.num_layers)])
        
        # 初始化 Mask Head 参数
        for m in self.dec_mask_head:
            for layer in m.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

        # 3. 替换 Decoder
        # 我们需要用支持 Mask 输出的子类替换掉父类初始化的 self.decoder
        # 为了保留父类初始化的参数配置，我们手动创建一个新的 DETRTransformerDecoderWithMask
        # 并复用父类 decoder 的 layers (权重共享)
        new_decoder = DETRTransformerDecoderWithMask(
            self.hidden_dim, 
            self.decoder.layers[0], # 复用 layer 定义
            self.num_layers, 
            self.decoder.eval_idx)
        # 关键：直接把父类已经初始化好的 layers 权重赋给新 decoder，避免重新初始化
        new_decoder.layers = self.decoder.layers 
        self.decoder = new_decoder

    def forward(self, feats, targets=None):
        # 1. 获取 Encoder 输入
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        # 2. 运行 Pixel Decoder 生成原型 (Prototypes)
        # prototype_masks: (B, prototype_dim, H/8, W/8)
        prototype_masks, _ = self.pixel_decoder(feats)

        # 3. 准备去噪训练 (Denoising)
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                getContrastiveDenoisingTrainingGroup(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        # 4. 准备 Decoder 输入
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # 5. 运行新的 Decoder (带 Mask Head)
        out_bboxes, out_logits, out_mask_coeffs = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            mask_head=self.dec_mask_head, # 传入 Mask Head
            attn_mask=attn_mask)

        # 6. 处理去噪训练的数据切分
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_mask_coeffs, out_mask_coeffs = torch.split(
                out_mask_coeffs, dn_meta['dn_num_split'], dim=2)

        # 7. 生成最终 Mask (矩阵乘法: Coefficients @ Prototypes)
        # out_mask_coeffs[-1]: (B, num_queries, prototype_dim)
        # prototype_masks: (B, prototype_dim, H, W)
        # Result: (B, num_queries, H, W)
        pred_masks = torch.einsum('bnc, bchw -> bnhw', out_mask_coeffs[-1], prototype_masks)

        out = {
            'pred_logits': out_logits[-1], 
            'pred_boxes': out_bboxes[-1],
            'pred_masks': pred_masks,
            'pred_mask_coeffs': out_mask_coeffs[-1],
            'proto_masks': prototype_masks
        }

        # 8. 辅助损失 (Aux Loss)
        if self.training and self.aux_loss:
            # 计算中间层的 Masks
            aux_masks_list = [
                torch.einsum('bnc, bchw -> bnhw', coeffs, prototype_masks) 
                for coeffs in out_mask_coeffs[:-1]
            ]
            
            out['aux_outputs'] = self._set_aux_loss(
                out_logits[:-1], out_bboxes[:-1], aux_masks_list)
            
            # Encoder 辅助输出 (Encoder 没有 Mask)
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                # 计算去噪部分的 Masks
                dn_masks_list = [
                    torch.einsum('bnc, bchw -> bnhw', coeffs, prototype_masks) 
                    for coeffs in dn_out_mask_coeffs
                ]
                out['dn_aux_outputs'] = self._set_aux_loss(
                    dn_out_logits, dn_out_bboxes, dn_masks_list)
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask=None):
        if outputs_mask is None:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class, outputs_coord)]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class, outputs_coord, outputs_mask)]