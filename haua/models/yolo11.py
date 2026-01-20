from typing import List, Tuple, Any, Dict, Union, Optional, Callable

from ._basemodel import BaseModel
from .backbone import Darknet
from .neck import C3k2PAN
from .head import YOLODetector, YOLOSegmenter

from .utils import make_divisible, freezeModelParameters, unfreezeModelParameters, ParameterFreezer

import torch.nn as nn


_MODEL_CONFIGS = {
    "n": {
        "depth_mult": .5, "width_mult": .25, "max_channels": 1024,
        "backbone": {"block_args": [(False, .25), (False, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 80}},
    "s": {
        "depth_mult": .5, "width_mult": .5, "max_channels": 1024,
        "backbone": {"block_args": [(False, .25), (False, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 128}},
    "m": {
        "depth_mult": .5, "width_mult": 1., "max_channels": 512,
        "backbone": {"block_args": [(True, .25), (True, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 256}},
    "l": {
        "depth_mult": 1., "width_mult": 1., "max_channels": 512,
        "backbone": {"block_args": [(True, .25), (True, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 256}},
    "x": {
        "depth_mult": 1., "width_mult": 1.50, "max_channels": 512,
        "backbone": {"block_args": [(True, .25), (True, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 96, "classify_hidden_channels": 384}}}


class Yolo11(BaseModel):
    
    base_backbone_layer_channels = (128, 256, 512, 512, 1024)
    base_backbone_num_blocks = (2, 2, 2, 2, 2)
    base_neck_out_channels = (256, 512, 1024)

    def __init__(
        self,
        model_type: str = "l",
        num_classes: int = 80,
        custom_postprocess: Optional[Callable[[Any], Any]] = None
    ):
        assert model_type in _MODEL_CONFIGS, f"Unsupported type: {model_type}"
        cfg = _MODEL_CONFIGS[model_type]
        self.width_mult = cfg["width_mult"]
        self.depth_mult = cfg["depth_mult"]
        self.max_channels = cfg["max_channels"]
        self.backbone_block_args = cfg["backbone"]["block_args"]
        self.head_locate_hidden_channels = cfg["head"]["locate_hidden_channels"]
        self.head_classify_hidden_channels = cfg["head"]["classify_hidden_channels"]
        self.backbone_layer_channels = [
            make_divisible(c * self.width_mult) for c in self.base_backbone_layer_channels]
        self.backbone_layer_channels = self._channel_trimming(self.backbone_layer_channels)
        self.backbone_num_blocks = [int(n * self.depth_mult) for n in self.base_backbone_num_blocks]
        self.neck_out_channels = [
            make_divisible(c * self.width_mult) for c in self.base_neck_out_channels]
        self.neck_out_channels = self._channel_trimming(self.neck_out_channels)
        module_configs = {
            "backbone": {
                "layer_channels": self.backbone_layer_channels,
                "num_blocks": self.backbone_num_blocks,
                "block_args": self.backbone_block_args},
            "neck": {
                "in_channels": self.backbone_layer_channels[-3:],
                "out_channels": self.neck_out_channels},
            "head": {
                "in_channels_list": self.neck_out_channels,
                "num_classes": num_classes,
                "locate_hidden_channels": self.head_locate_hidden_channels,
                "classify_hidden_channels": self.head_classify_hidden_channels}}
        super().__init__(Darknet, C3k2PAN, YOLODetector, custom_postprocess, module_configs)
    
    def _channel_trimming(self, channels: List[int]) -> List[int]:
        if channels[-1] > self.max_channels:
            channels[-1] = int(channels[-1] // 2)
        
        return channels


class Yolo11_train(nn.Module):
    def __init__(
        self,
        model_type: str = "l",
        num_classes: int = 80,
        custom_postprocess: Optional[Callable[[Any], Any]] = None
    ):
        super().__init__()
        self.yolo11 = Yolo11(model_type, num_classes, custom_postprocess)
        self.aux_head = YOLODetector(
            in_channels_list = self.yolo11.neck_out_channels, # type: ignore
            num_classes = num_classes,
            locate_hidden_channels = self.yolo11.head_locate_hidden_channels,
            classify_hidden_channels = self.yolo11.head_classify_hidden_channels)
        self.aux_head.train()
    
    def forward(self, x):
        _, fused, one2one = self.yolo11(x)
        one2many = self.aux_head(fused)

        return {"one2many": one2many, "one2one": one2one}


class Yolo11Seg(BaseModel):
    
    base_backbone_layer_channels = (128, 256, 512, 512, 1024)
    base_backbone_num_blocks = (2, 2, 2, 2, 2)
    base_neck_out_channels = (256, 512, 1024)

    def __init__(
        self,
        model_type: str = "l",
        num_classes: int = 80,
        custom_postprocess: Optional[Callable[[Any], Any]] = None
    ):
        assert model_type in _MODEL_CONFIGS, f"Unsupported type: {model_type}"
        cfg = _MODEL_CONFIGS[model_type]
        self.width_mult = cfg["width_mult"]
        self.depth_mult = cfg["depth_mult"]
        self.max_channels = cfg["max_channels"]
        self.backbone_block_args = cfg["backbone"]["block_args"]
        self.head_locate_hidden_channels = cfg["head"]["locate_hidden_channels"]
        self.head_classify_hidden_channels = cfg["head"]["classify_hidden_channels"]
        self.backbone_layer_channels = [
            make_divisible(c * self.width_mult) for c in self.base_backbone_layer_channels]
        self.backbone_layer_channels = self._channel_trimming(self.backbone_layer_channels)
        self.backbone_num_blocks = [int(n * self.depth_mult) for n in self.base_backbone_num_blocks]
        self.neck_out_channels = [
            make_divisible(c * self.width_mult) for c in self.base_neck_out_channels]
        self.neck_out_channels = self._channel_trimming(self.neck_out_channels)
        module_configs = {
            "backbone": {
                "layer_channels": self.backbone_layer_channels,
                "num_blocks": self.backbone_num_blocks,
                "block_args": self.backbone_block_args},
            "neck": {
                "in_channels": self.backbone_layer_channels[-3:],
                "out_channels": self.neck_out_channels},
            "head": {
                "in_channels_list": self.neck_out_channels,
                "num_classes": num_classes,
                "locate_hidden_channels": self.head_locate_hidden_channels,
                "classify_hidden_channels": self.head_classify_hidden_channels}}
        super().__init__(Darknet, C3k2PAN, YOLOSegmenter, custom_postprocess, module_configs)
    
    def _channel_trimming(self, channels: List[int]) -> List[int]:
        if channels[-1] > self.max_channels:
            channels[-1] = int(channels[-1] // 2)
        
        return channels


# 预设冻结配置
YOLO11SEG_FREEZE_PRESETS = {
    # 不冻结
    'none': [],
    # 只训练分割头（最常用）
    'seg_only': [
        'yolo11seg.backbone',
        'yolo11seg.neck',
        'yolo11seg.head.detector',
        'aux_head'],
    # 训练分割头和neck
    'seg_and_neck': [
        'yolo11seg.backbone',
        'yolo11seg.head.detector',
        'aux_head'],
    # 只冻结backbone
    'backbone_only': [
        'yolo11seg.backbone'],
    # 冻结所有检测相关（backbone + detector）
    'freeze_detection': [
        'yolo11seg.backbone',
        'yolo11seg.neck',
        'yolo11seg.head.detector',
        'aux_head'],
    # 渐进式训练 - 阶段1（只训练分割）
    'stage1_seg_warmup': [
        'yolo11seg.backbone',
        'yolo11seg.neck',
        'yolo11seg.head.detector',
        'aux_head'],
    # 渐进式训练 - 阶段2（训练分割+neck）
    'stage2_neck_finetune': [
        'yolo11seg.backbone',
        'yolo11seg.head.detector',
        'aux_head'],
    # 渐进式训练 - 阶段3（全部训练）
    'stage3_full_finetune': []}


class Yolo11Seg_train(nn.Module):
    """
    YOLO11 分割训练模型
    
    Features:
        - 支持灵活的参数冻结（通过 freeze_patterns 或 freeze_preset）
        - 提供便捷的冻结/解冻方法
        - 自动统计和打印冻结信息
    """
    
    def __init__(
        self,
        model_type: str = "l",
        num_classes: int = 80,
        custom_postprocess: Optional[Callable[[Any], Any]] = None,
        freeze_patterns: Optional[Union[str, List[str]]] = None,
        freeze_preset: Optional[str] = None,
        verbose_freeze: bool = True
    ):
        """
        Args:
            model_type: 模型类型 ('n', 's', 'm', 'l', 'x')
            num_classes: 类别数量
            custom_postprocess: 自定义后处理函数
            freeze_patterns: 冻结模式列表
                - None: 不冻结任何参数
                - ['yolo11seg.backbone']: 只冻结backbone
                - ['yolo11seg.backbone', 'yolo11seg.neck', 'aux_head']: 冻结多个模块
            freeze_preset: 使用预设配置（与freeze_patterns二选一，preset优先）
                - 'none': 不冻结
                - 'seg_only': 只训练分割头（推荐用于初始训练）
                - 'seg_and_neck': 训练分割头和neck
                - 'backbone_only': 只冻结backbone
                - 'stage1_seg_warmup': 渐进式训练阶段1
                - 'stage2_neck_finetune': 渐进式训练阶段2
                - 'stage3_full_finetune': 渐进式训练阶段3
            verbose_freeze: 是否打印冻结详情
        
        Examples:
            >>> # 示例1: 只训练分割头
            >>> model = Yolo11Seg_train(
            ...     model_type='l',
            ...     freeze_preset='seg_only'
            ... )
            
            >>> # 示例2: 自定义冻结
            >>> model = Yolo11Seg_train(
            ...     model_type='l',
            ...     freeze_patterns=['yolo11seg.backbone', 'aux_head']
            ... )
            
            >>> # 示例3: 不冻结（全参数训练）
            >>> model = Yolo11Seg_train(model_type='l')
        """
        super().__init__()
        
        # 构建模型
        self.yolo11seg = Yolo11Seg(model_type, num_classes, custom_postprocess)
        self.aux_head = YOLODetector(
            in_channels_list=self.yolo11seg.neck_out_channels,  # type: ignore
            num_classes=num_classes,
            locate_hidden_channels=self.yolo11seg.head_locate_hidden_channels,
            classify_hidden_channels=self.yolo11seg.head_classify_hidden_channels)
        self.aux_head.train()
        
        # 保存配置
        self.verbose_freeze = verbose_freeze
        self._freezer = None  # 懒加载
        
        # 应用冻结策略（预设优先）
        if freeze_preset is not None:
            self.freeze_by_preset(freeze_preset)
        elif freeze_patterns is not None:
            self.freeze_parameters(freeze_patterns)
        elif self.verbose_freeze:
            print("✅ 未指定冻结参数，所有参数均可训练")
    
    def freeze_parameters(
        self,
        patterns: Union[str, List[str]],
        mode: str = 'prefix',
        verbose: Optional[bool] = None
    ):
        """
        冻结指定的参数
        
        Args:
            patterns: 冻结模式
                - 字符串: 'yolo11seg.backbone'
                - 列表: ['yolo11seg.backbone', 'yolo11seg.neck', 'aux_head']
            mode: 匹配模式
                - 'prefix': 前缀匹配（默认）
                - 'regex': 正则表达式匹配
                - 'exact': 精确匹配
                - 'contains': 包含匹配
            verbose: 是否打印详情（默认使用初始化时的设置）
        
        Returns:
            dict: 冻结统计信息
        
        Examples:
            >>> model = Yolo11Seg_train('l')
            >>> # 冻结backbone和neck
            >>> model.freeze_parameters(['yolo11seg.backbone', 'yolo11seg.neck'])
            >>> # 使用正则表达式冻结所有bn层
            >>> model.freeze_parameters([r'.*\.bn\d*\.'], mode='regex')
        """
        if verbose is None:
            verbose = self.verbose_freeze
        
        stats = freezeModelParameters(
            self,
            patterns=patterns,
            mode=mode,
            verbose=verbose)
        
        return stats
    
    def unfreeze_parameters(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        verbose: Optional[bool] = None
    ):
        """
        解冻指定的参数
        
        Args:
            patterns: 解冻模式
                - None 或 'all': 解冻所有参数
                - 列表: ['yolo11seg.neck'] 只解冻指定模块
            verbose: 是否打印详情
        
        Returns:
            dict: 解冻统计信息
        
        Examples:
            >>> # 解冻neck
            >>> model.unfreeze_parameters(['yolo11seg.neck'])
            >>> # 解冻所有
            >>> model.unfreeze_parameters('all')
        """
        if verbose is None:
            verbose = self.verbose_freeze
        
        stats = unfreezeModelParameters(
            self,
            patterns=patterns,
            verbose=verbose)
        
        return stats
    
    def freeze_by_preset(self, preset_name: str):
        """
        使用预设配置冻结参数
        
        Args:
            preset_name: 预设名称
        
        Available presets:
            - 'none': 不冻结
            - 'seg_only': 只训练分割头（推荐）
            - 'seg_and_neck': 训练分割头和neck
            - 'backbone_only': 只冻结backbone
            - 'stage1_seg_warmup': 渐进式训练阶段1
            - 'stage2_neck_finetune': 渐进式训练阶段2
            - 'stage3_full_finetune': 渐进式训练阶段3
        
        Examples:
            >>> model = Yolo11Seg_train('l')
            >>> model.freeze_by_preset('seg_only')
        """
        if preset_name not in YOLO11SEG_FREEZE_PRESETS:
            available = ', '.join(f"'{k}'" for k in YOLO11SEG_FREEZE_PRESETS.keys())
            raise ValueError(
                f"❌ 未知预设: '{preset_name}'\n"
                f"   可用预设: {available}")
        
        patterns = YOLO11SEG_FREEZE_PRESETS[preset_name]
        
        if self.verbose_freeze:
            print(f"\n🎯 应用冻结预设: '{preset_name}'")
        
        if patterns:
            return self.freeze_parameters(patterns)
        else:
            if self.verbose_freeze:
                print("✅ 预设为不冻结，所有参数均可训练")
            return {}
    
    def get_freezer(self) -> ParameterFreezer:
        """
        获取 ParameterFreezer 实例（懒加载）
        用于需要高级功能时（如参数分组、详细检查等）
        
        Returns:
            ParameterFreezer 实例
        
        Examples:
            >>> model = Yolo11Seg_train('l')
            >>> freezer = model.get_freezer()
            >>> freezer.print_trainable_summary()
            >>> freezer.inspect_available_patterns()
        """
        if self._freezer is None:
            self._freezer = ParameterFreezer(self, verbose=self.verbose_freeze)
        return self._freezer
    
    def print_trainable_summary(self):
        """
        打印可训练参数摘要
        
        Examples:
            >>> model = Yolo11Seg_train('l', freeze_preset='seg_only')
            >>> model.print_trainable_summary()
        """
        freezer = self.get_freezer()
        freezer.print_trainable_summary()
    
    def print_module_summary(self):
        """
        打印模块摘要（包含各模块的冻结状态）
        
        Examples:
            >>> model = Yolo11Seg_train('l', freeze_preset='seg_only')
            >>> model.print_module_summary()
        """
        print(f"\n{'='*80}")
        print(f"{'Yolo11Seg_train 模块摘要':^80}")
        print(f"{'='*80}")
        
        # 定义模块
        modules = {
            'yolo11seg.backbone': 'Backbone',
            'yolo11seg.neck': 'Neck',
            'yolo11seg.head.detector': 'Detector',
            'yolo11seg.head.segmentation': 'Segmentation',
            'aux_head': 'Aux Head'
        }
        
        print(f"{'模块':<30} {'总参数':>15} {'可训练':>15} {'状态':>10}")
        print(f"{'-'*80}")
        
        total_params = 0
        total_trainable = 0
        
        for module_path, module_name in modules.items():
            # 获取模块
            parts = module_path.split('.')
            module = self
            try:
                for part in parts:
                    module = getattr(module, part)
                
                # 统计参数
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                total_params += total
                total_trainable += trainable
                
                # 状态
                if trainable == 0:
                    status = '🔒 冻结'
                elif trainable == total:
                    status = '🔓 训练'
                else:
                    status = '⚡ 部分'
                
                print(
                    f"{module_name:<30} "
                    f"{total:>15,} "
                    f"{trainable:>15,} "
                    f"{status:>10}")
            except AttributeError:
                print(f"{module_name:<30} {'N/A':>15} {'N/A':>15} {'❌ 未找到':>10}")
        
        print(f"{'-'*80}")
        print(
            f"{'总计':<30} "
            f"{total_params:>15,} "
            f"{total_trainable:>15,} "
            f"({100*total_trainable/total_params:.1f}%)")
        print(f"{'='*80}\n")
    
    def inspect_freeze_patterns(self, max_depth: int = 3):
        """
        检查可用的冻结模式
        
        Args:
            max_depth: 最大层级深度
        
        Examples:
            >>> model = Yolo11Seg_train('l')
            >>> model.inspect_freeze_patterns()
        """
        freezer = self.get_freezer()
        return freezer.inspect_available_patterns(max_depth)
    
    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_neck: float = 5e-5,
        lr_detector: float = 1e-4,
        lr_segmenter: float = 1e-3,
        lr_aux: float = 1e-4,
        weight_decay: float = 5e-4
    ):
        """
        获取参数分组（用于创建优化器）
        
        Args:
            lr_backbone: Backbone学习率
            lr_neck: Neck学习率
            lr_detector: Detector学习率
            lr_segmenter: Segmentation学习率
            lr_aux: Aux Head学习率
            weight_decay: 权重衰减
        
        Returns:
            List[Dict]: 参数分组列表，可直接传入优化器
        
        Examples:
            >>> model = Yolo11Seg_train('l', freeze_preset='seg_only')
            >>> param_groups = model.get_parameter_groups(lr_segmenter=1e-3)
            >>> optimizer = torch.optim.AdamW(param_groups)
        """
        freezer = self.get_freezer()
        
        group_patterns = {
            'backbone': ['yolo11seg.backbone'],
            'neck': ['yolo11seg.neck'],
            'detector': ['yolo11seg.head.detector'],
            'segmenter': ['yolo11seg.head.segmentation'],
            'aux_head': ['aux_head']}
        
        param_dict = freezer.get_parameter_groups(group_patterns, mode='prefix')
        
        # 构建参数组
        param_groups = []
        
        if 'backbone' in param_dict and param_dict['backbone']:
            param_groups.append({
                'params': param_dict['backbone'],
                'lr': lr_backbone,
                'weight_decay': weight_decay})
        
        if 'neck' in param_dict and param_dict['neck']:
            param_groups.append({
                'params': param_dict['neck'],
                'lr': lr_neck,
                'weight_decay': weight_decay})
        
        if 'detector' in param_dict and param_dict['detector']:
            param_groups.append({
                'params': param_dict['detector'],
                'lr': lr_detector,
                'weight_decay': weight_decay})
        
        if 'segmenter' in param_dict and param_dict['segmenter']:
            param_groups.append({
                'params': param_dict['segmenter'],
                'lr': lr_segmenter,
                'weight_decay': weight_decay})
        
        if 'aux_head' in param_dict and param_dict['aux_head']:
            param_groups.append({
                'params': param_dict['aux_head'],
                'lr': lr_aux,
                'weight_decay': weight_decay})
        
        return param_groups

    def forward(self, x):
        """前向传播"""
        _, fused, seg_outs = self.yolo11seg(x)
        one2one, seg_out, prototype_mask = seg_outs
        one2many = self.aux_head(fused)

        return {
            "one2many": one2many,
            "one2one": one2one,
            "seg_out": seg_out,
            "prototype_mask": prototype_mask}