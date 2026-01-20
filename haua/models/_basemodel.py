from typing import Union, Optional, Type, Callable, Any, Dict, List

import torch
import torch.nn as nn

from ..utils import get_script_name
from .utils import FreezeMixin


_SCRIPT_NAME = get_script_name(__file__)


BASE_MODEL_FREEZE_PRESETS = {
    # 基础预设
    'none': [],
    'all': ['backbone', 'neck', 'head'],
    
    # 单模块冻结
    'backbone_only': ['backbone'],
    'neck_only': ['neck'],
    'head_only': ['head'],
    
    # 组合冻结
    'freeze_backbone_neck': ['backbone', 'neck'],
    'freeze_backbone_head': ['backbone', 'head'],
    'freeze_neck_head': ['neck', 'head'],
    
    # 常用训练策略
    'train_head_only': ['backbone', 'neck'],           # 只训练head
    'train_neck_head': ['backbone'],                    # 训练neck和head
    'finetune_all_from_backbone': [],                   # 全部训练（微调）
    
    # 渐进式解冻阶段
    'stage1_head_warmup': ['backbone', 'neck'],
    'stage2_neck_tuning': ['backbone'],
    'stage3_full_tuning': []}


class BaseModel(nn.Module, FreezeMixin):
    """
    通用模型基类，支持以下功能：
        1. 灵活的模块定义（实例/类）
        2. 自动参数冻结（继承自 FreezeMixin）
        3. 模型融合（fuse）
        4. 自定义后处理
    
    特性：
        - 所有继承此类的模型自动获得参数冻结能力
        - 提供针对 backbone/neck/head 结构的便捷冻结方法
        - 支持渐进式训练的预设配置
    """

    def __init__(
        self,
        backbone: Union[nn.Module, Type[nn.Module]],
        neck: Optional[Union[nn.Module, Type[nn.Module]]] = None,
        head: Optional[Union[nn.Module, Type[nn.Module]]] = None,
        custom_postprocess: Optional[Callable[[Any], Any]] = None,
        module_configs: Optional[Dict[str, dict]] = None,
        freeze_patterns: Optional[List[str]] = None,
        freeze_preset: Optional[str] = None,
        verbose_freeze: bool = True,
    ):
        """
        Args:
            backbone: Backbone 模块（实例或类）
            neck: Neck 模块（实例或类）
            head: Head 模块（实例或类）
            custom_postprocess: 自定义后处理函数
            module_configs: 模块配置字典
            freeze_patterns: 冻结模式列表，例如 ['backbone', 'neck']
            freeze_preset: 使用预设冻结配置，例如 'train_head_only'
            verbose_freeze: 是否打印冻结详情
        """
        super().__init__()
        # 初始化模块配置
        self.module_configs = module_configs or {}
        # 初始化三大模块
        self.backbone = self._init_module("backbone", backbone)
        self.neck = self._init_module("neck", neck) if neck else nn.Identity()
        self.head = self._init_module("head", head) if head else nn.Identity()
        # 后处理函数
        self.custom_postprocess = custom_postprocess
        # 初始化 Freezer（来自 FreezeMixin）
        self.setup_freezer(verbose=verbose_freeze)
        # 应用冻结策略（预设优先）
        if freeze_preset:
            self.freeze_by_preset(freeze_preset)
        elif freeze_patterns:
            self.freeze_parameters(freeze_patterns)

    def _init_module(
        self,
        name: str,
        module_or_class: Union[nn.Module, Type[nn.Module]]
    ) -> nn.Module:
        """初始化模块：实例直接返回；类则用 config 实例化"""
        if isinstance(module_or_class, nn.Module):
            return module_or_class
        elif isinstance(module_or_class, type) and issubclass(module_or_class, nn.Module):
            cfg = self.module_configs.get(name, {})
            return module_or_class(**cfg)
        else:
            raise TypeError(
                f"{name} 必须是 nn.Module 实例 或 nn.Module 子类，"
                f"当前类型为 {type(module_or_class)}")
    
    def forward_backbone(self, x: torch.Tensor) -> Any:
        """Backbone前向传播"""
        return self.backbone(x)

    def forward_neck(self, feats: Any) -> Any:
        """Neck前向传播"""
        return self.neck(feats)

    def forward_head(self, feats: Any) -> Any:
        """Head前向传播"""
        return self.head(feats)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """完整前向传播"""
        feats = self.forward_backbone(x)
        fused = self.forward_neck(feats)
        out = self.forward_head(fused)
        
        if self.custom_postprocess is not None:
            out = self.custom_postprocess(out)

        return feats, fused, out
    
    def freeze_by_preset(self, preset_name: str):
        """
        使用预设配置冻结参数
        
        Args:
            preset_name: 预设名称，可选：
                - 'none': 不冻结
                - 'backbone_only': 只冻结backbone
                - 'train_head_only': 只训练head
                - 'stage1_head_warmup': 渐进式训练阶段1
                等等，详见 BASE_MODEL_FREEZE_PRESETS
        """
        if preset_name not in BASE_MODEL_FREEZE_PRESETS:
            raise ValueError(
                f"未知预设: {preset_name}\n"
                f"可用预设: {list(BASE_MODEL_FREEZE_PRESETS.keys())}")
        
        patterns = BASE_MODEL_FREEZE_PRESETS[preset_name]
        
        if self._freezer.verbose:
            print(f"\n🎯 应用冻结预设: '{preset_name}'")
        
        return self.freeze_parameters(patterns)
    
    def freeze_backbone(self):
        """便捷方法：冻结Backbone"""
        return self.freeze_parameters(['backbone'])
    
    def freeze_neck(self):
        """便捷方法：冻结Neck"""
        return self.freeze_parameters(['neck'])
    
    def freeze_head(self):
        """便捷方法：冻结Head"""
        return self.freeze_parameters(['head'])
    
    def unfreeze_backbone(self):
        """便捷方法：解冻Backbone"""
        return self.unfreeze_parameters(['backbone'])
    
    def unfreeze_neck(self):
        """便捷方法：解冻Neck"""
        return self.unfreeze_parameters(['neck'])
    
    def unfreeze_head(self):
        """便捷方法：解冻Head"""
        return self.unfreeze_parameters(['head'])
    
    def get_standard_parameter_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_neck: float = 1e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 5e-4
    ) -> List[Dict]:
        """
        获取标准的参数分组（用于优化器）
        
        Args:
            lr_backbone: Backbone学习率
            lr_neck: Neck学习率
            lr_head: Head学习率
            weight_decay: 权重衰减
        
        Returns:
            参数分组列表，可直接传入优化器
        
        Example:
            >>> param_groups = model.get_standard_parameter_groups()
            >>> optimizer = torch.optim.AdamW(param_groups)
        """
        group_patterns = {
            'backbone': ['backbone'],
            'neck': ['neck'],
            'head': ['head']}
        
        param_dict = self.get_parameter_groups(group_patterns)
        
        param_groups = []
        if 'backbone' in param_dict:
            param_groups.append({
                'params': param_dict['backbone'],
                'lr': lr_backbone,
                'weight_decay': weight_decay})
        if 'neck' in param_dict:
            param_groups.append({
                'params': param_dict['neck'],
                'lr': lr_neck,
                'weight_decay': weight_decay})
        if 'head' in param_dict:
            param_groups.append({
                'params': param_dict['head'],
                'lr': lr_head,
                'weight_decay': weight_decay})
        
        return param_groups
    
    def print_module_summary(self):
        """打印模块摘要（包含冻结状态）"""
        print(f"\n{'='*80}")
        print(f"{'模型摘要':^80}")
        print(f"{'='*80}")
        print(f"模型类名: {self.class_name}")
        print(f"{'-'*80}")
        
        # 统计各模块参数
        module_stats = {}
        for module_name in ['backbone', 'neck', 'head']:
            module = getattr(self, module_name)
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen = total - trainable
            
            module_stats[module_name] = {
                'class': module.__class__.__name__,
                'total': total,
                'trainable': trainable,
                'frozen': frozen,
                'status': '🔓 训练中' if trainable > 0 else '🔒 已冻结'}
        
        # 打印表格
        print(f"{'模块':<15} {'类名':<30} {'总参数':>15} {'可训练':>15} {'状态':>10}")
        print(f"{'-'*80}")
        
        for name, stats in module_stats.items():
            print(
                f"{name:<15} "
                f"{stats['class']:<30} "
                f"{stats['total']:>15,} "
                f"{stats['trainable']:>15,} "
                f"{stats['status']:>10}")
        
        print(f"{'-'*80}")
        total_all = sum(s['total'] for s in module_stats.values())
        trainable_all = sum(s['trainable'] for s in module_stats.values())
        print(
            f"{'合计':<15} "
            f"{'':<30} "
            f"{total_all:>15,} "
            f"{trainable_all:>15,} "
            f"({100*trainable_all/total_all:.1f}%)")
        print(f"{'='*80}")
        
        if self.custom_postprocess:
            print(f"✓ 自定义后处理: 已启用")
        print()
    
    def summary(self) -> Dict[str, str]:
        """返回模型摘要字典"""
        return {
            "backbone": self.backbone.__class__.__name__,
            "neck": self.neck.__class__.__name__,
            "head": self.head.__class__.__name__,
            "has_custom_postprocess": str(self.custom_postprocess is not None)}

    def set_postprocess(self, fn: Callable[[Any], Any]):
        """设置后处理函数"""
        self.custom_postprocess = fn

    def fuse(self, verbose: bool = True) -> "BaseModel":
        """自动 fuse 模型中所有支持 .fuse() 的子模块（如 ConvBNAct）"""
        fused_count = 0

        def _fuse_recursive(module: nn.Module) -> int:
            count = 0
            if hasattr(module, 'fuse') and callable(getattr(module, 'fuse')):
                if hasattr(module, 'bn'):  # 针对 ConvBNAct 类
                    module.fuse()
                    count += 1
            for child in module.children():
                count += _fuse_recursive(child)
            return count

        for name in ["backbone", "neck", "head"]:
            module = getattr(self, name, None)
            if module is not None:
                fused_count += _fuse_recursive(module)
        
        if verbose:
            print(f"✅ BaseModel.fuse(): 成功融合 {fused_count} 个可融合模块。")

        return self
    
    @property
    def class_name(self):
        """获取类名"""
        return self.__class__.__name__