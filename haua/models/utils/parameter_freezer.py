"""
──────────────────────────────────────────────────────────
                  parameter_freezer.py                     
──────────────────────────────────────────────────────────
    ParameterFreezer (核心引擎)                         
    • 完整的冻结/解冻逻辑                               
    • 多种匹配模式（前缀/正则/精确/包含）               
    • 详细的统计和日志                                  
    • 参数分组功能                                      
──────────────────────────────────────────────────────────
                           ▲                                  
                           │ 使用                             
          ┌────────────────┴───────────────┐
          │                                │
      FreezeMixin                  freezeModelParameters
       (Mixin类)                       (独立函数)         
      继承方式使用                     函数式使用         

"""
import re
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Callable
from collections import defaultdict


class ParameterFreezer:
    """
    通用参数冻结工具类
    
    功能:
        - 支持前缀匹配、正则表达式匹配、精确匹配
        - 提供详细的冻结统计和日志
        - 支持参数分组（用于优化器配置）
        - 支持动态冻结/解冻

    Example:
        1. 多种匹配模式
            model = YourModel()
            freezer = ParameterFreezer(model, verbose=True)

            # 模式1: 前缀匹配 (默认，最常用)
            freezer.freeze(['backbone', 'neck'], mode='prefix')
            # 匹配: backbone.layer1.weight, neck.conv1.bias, ...

            # 模式2: 正则表达式
            freezer.freeze([r'.*\.conv\d+\.weight'], mode='regex')
            # 匹配所有名为 convX.weight 的参数

            # 模式3: 精确匹配
            freezer.freeze(['backbone.layer1.conv.weight'], mode='exact')
            # 只匹配完全相同的参数名

            # 模式4: 包含匹配
            freezer.freeze(['bn', 'bias'], mode='contains')
        2. 排除模式（白名单）
            # 冻结所有 backbone，但保持 BN 层可训练
            freezer.freeze(
                patterns=['backbone'],
                exclude_patterns=['bn', 'batch_norm'],  # 排除包含这些关键字的参数
                mode='prefix'
            )

            # 冻结所有参数，但保持最后几层可训练
            freezer.freeze(
                patterns=['.*'],  # 匹配所有
                exclude_patterns=[r'.*\.head\..*'],  # 排除head
                mode='regex'
            )
        3. BatchNorm 特殊处理
            # 默认不冻结 BN 层（推荐，保持统计信息更新）
            freezer.freeze(
                patterns=['backbone'],
                include_bn=False  # 默认值
            )

            # 强制冻结包括 BN 在内的所有层
            freezer.freeze(
                patterns=['backbone'],
                include_bn=True
            )
        4. 参数分组（用于优化器）
            # 定义分组规则
            group_patterns = {
                'backbone': ['backbone'],
                'neck': ['neck'],
                'head_conv': ['head.conv'],
                'head_fc': ['head.fc']
            }

            # 获取分组
            param_groups = freezer.get_parameter_groups(group_patterns)

            # 为不同组设置不同学习率
            optimizer = torch.optim.AdamW([
                {'params': param_groups['backbone'], 'lr': 1e-5},
                {'params': param_groups['neck'], 'lr': 5e-5},
                {'params': param_groups['head_conv'], 'lr': 1e-4},
                {'params': param_groups['head_fc'], 'lr': 1e-3},
            ])
        5. 统计和检查
            # 冻结后查看统计
            stats = freezer.freeze(['backbone', 'neck'])
            print(stats)
            # {
            #     'total_params': 52800000,
            #     'frozen_params': 38000000,
            #     'trainable_params': 14800000,
            #     'frozen_layers': ['backbone.layer1.weight', ...],
            #     'trainable_layers': ['head.conv1.weight', ...]
            # }

            # 打印详细摘要
            freezer.print_trainable_summary()

            # 检查可用的冻结模式
            available_patterns = freezer.inspect_available_patterns(max_depth=3)
            # {'backbone': 28400000, 'backbone.layer1': 5600000, ...}
        
        高级功能
        ---------
        1. 迁移学习 - 逐步解冻
            model = PretrainedModel()
            freezer = ParameterFreezer(model, verbose=True)

            # 阶段1: 冻结所有，只训练分类头 (5 epochs)
            freezer.freeze(['features'], mode='prefix')
            train(model, epochs=5, lr=1e-3)

            # 阶段2: 解冻最后几层 (10 epochs)
            freezer.unfreeze(['features.layer4'])
            train(model, epochs=10, lr=1e-4)

            # 阶段3: 解冻所有 (20 epochs)
            freezer.unfreeze('all')
            train(model, epochs=20, lr=1e-5)
        
        2. 多任务学习 - 选择性冻结
            model = MultiTaskModel()  # 有 shared_backbone 和 task1_head, task2_head
            freezer = ParameterFreezer(model)

            # 训练任务1，冻结任务2的头
            freezer.freeze(['task2_head'])
            train_task1(model)

            # 训练任务2，冻结任务1的头
            freezer.freeze(['task1_head'])
            freezer.unfreeze(['task2_head'])
            train_task2(model)

            # 联合训练，解冻所有
            freezer.unfreeze('all')
            train_joint(model)
        
        3. 知识蒸馏 - 冻结教师模型
            teacher = TeacherModel()
            student = StudentModel()

            # 冻结教师模型的所有参数
            teacher_freezer = ParameterFreezer(teacher, verbose=False)
            teacher_freezer.freeze('all')

            # 只训练学生模型
            student_freezer = ParameterFreezer(student, verbose=True)
            # 学生模型全部可训练（不冻结）

            # 训练
            for batch in dataloader:
                with torch.no_grad():
                    teacher_output = teacher(batch)
                
                student_output = student(batch)
                loss = distillation_loss(student_output, teacher_output)
                loss.backward()
                optimizer.step()
    """
    
    def __init__(self, model: nn.Module, verbose: bool = True):
        """
        Args:
            model: PyTorch模型
            verbose: 是否打印详细信息
        """
        self.model = model
        self.verbose = verbose
        self._freeze_history = []
    
    def freeze(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        mode: str = 'prefix',
        exclude_patterns: Optional[Union[str, List[str]]] = None,
        include_bn: bool = False
    ) -> Dict:
        """
        冻结参数
        
        Args:
            patterns: 匹配模式，可以是单个字符串或列表
                - None: 不冻结任何参数
                - 'all': 冻结所有参数
                - ['backbone', 'neck']: 匹配多个模式
            mode: 匹配模式
                - 'prefix': 前缀匹配 (默认)
                - 'regex': 正则表达式匹配
                - 'exact': 精确匹配
                - 'contains': 包含匹配
            exclude_patterns: 排除模式（即使匹配也不冻结）
            include_bn: 是否冻结 BatchNorm 层
        
        Returns:
            dict: 冻结统计信息
        """
        if patterns is None or patterns == []:
            if self.verbose:
                print("❌ 未指定冻结模式，跳过冻结")
            return {'frozen': 0, 'trainable': sum(p.numel() for p in self.model.parameters())}
        
        # 标准化输入
        if isinstance(patterns, str):
            patterns = [patterns]
        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]
        elif exclude_patterns is None:
            exclude_patterns = []
        
        # 特殊处理 'all'
        if 'all' in patterns:
            return self._freeze_all(exclude_patterns)
        
        # 统计信息
        stats = {
            'total_params': 0,
            'frozen_params': 0,
            'trainable_params': 0,
            'frozen_layers': [],
            'trainable_layers': [],
            'skipped_bn': []}
        
        # 遍历所有参数
        for name, param in self.model.named_parameters():
            stats['total_params'] += param.numel()
            
            # 检查是否是 BatchNorm 参数
            is_bn = self._is_batchnorm_param(name)
            if is_bn and not include_bn:
                stats['skipped_bn'].append(name)
                continue
            
            # 检查是否匹配排除模式
            if self._match_patterns(name, exclude_patterns, mode):
                param.requires_grad = True
                stats['trainable_params'] += param.numel()
                stats['trainable_layers'].append(name)
                continue
            
            # 检查是否匹配冻结模式
            if self._match_patterns(name, patterns, mode):
                param.requires_grad = False
                stats['frozen_params'] += param.numel()
                stats['frozen_layers'].append(name)
            else:
                param.requires_grad = True
                stats['trainable_params'] += param.numel()
                stats['trainable_layers'].append(name)
        
        # 记录历史
        self._freeze_history.append({
            'patterns': patterns,
            'mode': mode,
            'stats': stats})
        
        # 打印摘要
        if self.verbose:
            self._print_freeze_summary(patterns, mode, stats)
        
        return stats
    
    def unfreeze(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        mode: str = 'prefix'
    ) -> Dict:
        """
        解冻参数
        
        Args:
            patterns: 匹配模式
                - None 或 'all': 解冻所有参数
                - ['backbone']: 只解冻指定模式
            mode: 匹配模式
        
        Returns:
            dict: 解冻统计信息
        """
        if patterns is None or patterns == 'all':
            return self._unfreeze_all()
        
        if isinstance(patterns, str):
            patterns = [patterns]
        
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if self._match_patterns(name, patterns, mode):
                param.requires_grad = True
                unfrozen_count += 1
        
        if self.verbose:
            print(f"✅ 解冻了 {unfrozen_count} 个参数组")
        
        return {'unfrozen': unfrozen_count}
    
    def get_parameter_groups(
        self,
        group_patterns: Dict[str, List[str]],
        mode: str = 'prefix'
    ) -> Dict[str, List[nn.Parameter]]:
        """
        根据模式分组参数（用于优化器配置）
        
        Args:
            group_patterns: 分组模式字典
                例如: {
                    'backbone': ['backbone'],
                    'neck': ['neck'],
                    'head': ['head']
                }
            mode: 匹配模式
        
        Returns:
            dict: 参数分组
        """
        groups = {name: [] for name in group_patterns.keys()}
        groups['others'] = []  # 未匹配的参数
        
        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            matched = False
            for group_name, patterns in group_patterns.items():
                if self._match_patterns(param_name, patterns, mode):
                    groups[group_name].append(param)
                    matched = True
                    break
            
            if not matched:
                groups['others'].append(param)
        
        # 移除空组
        groups = {k: v for k, v in groups.items() if v}
        
        if self.verbose:
            self._print_parameter_groups(groups)
        
        return groups
    
    def print_trainable_summary(self):
        """打印当前可训练参数摘要"""
        total = 0
        trainable = 0
        frozen = 0
        
        trainable_layers = []
        frozen_layers = []
        
        for name, param in self.model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                trainable_layers.append((name, param.numel()))
            else:
                frozen += param.numel()
                frozen_layers.append((name, param.numel()))
        
        print(f"\n{'='*80}")
        print(f"{'参数状态摘要':^80}")
        print(f"{'='*80}")
        print(f"总参数:      {total:>15,} (100.00%)")
        print(f"可训练参数:  {trainable:>15,} ({100*trainable/total:>6.2f}%)")
        print(f"冻结参数:    {frozen:>15,} ({100*frozen/total:>6.2f}%)")
        print(f"{'='*80}")
        
        if trainable_layers:
            print(f"\n✓ 可训练层 ({len(trainable_layers)}):")
            self._print_layer_list(trainable_layers, max_display=15)
        
        if frozen_layers:
            print(f"\n✗ 冻结层 ({len(frozen_layers)}):")
            self._print_layer_list(frozen_layers, max_display=15)
        
        print(f"{'='*80}\n")
    
    def inspect_available_patterns(self, max_depth: int = 3) -> Dict[str, int]:
        """
        检查模型中可用的冻结模式
        
        Args:
            max_depth: 最大层级深度
        
        Returns:
            dict: {pattern: param_count}
        """
        patterns = defaultdict(int)
        
        for name, param in self.model.named_parameters():
            parts = name.split('.')
            for depth in range(1, min(len(parts), max_depth) + 1):
                pattern = '.'.join(parts[:depth])
                patterns[pattern] += param.numel()
        
        # 排序并打印
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"{'可用的冻结模式':^80}")
            print(f"{'='*80}")
            print(f"{'模式':<50} {'参数数量':>15} {'层数':>10}")
            print(f"{'-'*80}")
            
            # 统计每个模式的层数
            pattern_counts = defaultdict(int)
            for name, _ in self.model.named_parameters():
                parts = name.split('.')
                for depth in range(1, min(len(parts), max_depth) + 1):
                    pattern = '.'.join(parts[:depth])
                    pattern_counts[pattern] += 1
            
            for pattern in sorted(patterns.keys()):
                print(f"{pattern:<50} {patterns[pattern]:>15,} {pattern_counts[pattern]:>10}")
            print(f"{'='*80}\n")
        
        return dict(patterns)
    
    def _match_patterns(
        self,
        name: str,
        patterns: List[str],
        mode: str
    ) -> bool:
        """检查参数名是否匹配任何模式"""
        for pattern in patterns:
            if mode == 'prefix':
                if name.startswith(pattern):
                    return True
            elif mode == 'regex':
                if re.search(pattern, name):
                    return True
            elif mode == 'exact':
                if name == pattern:
                    return True
            elif mode == 'contains':
                if pattern in name:
                    return True
        return False
    
    def _is_batchnorm_param(self, name: str) -> bool:
        """检查是否是 BatchNorm 参数"""
        bn_keywords = ['bn', 'batch_norm', 'batchnorm', 'norm']
        name_lower = name.lower()
        return any(kw in name_lower for kw in bn_keywords)
    
    def _freeze_all(self, exclude_patterns: List[str]) -> Dict:
        """冻结所有参数"""
        stats = {'total_params': 0, 'frozen_params': 0, 'trainable_params': 0}
        
        for name, param in self.model.named_parameters():
            stats['total_params'] += param.numel()
            
            if exclude_patterns and self._match_patterns(name, exclude_patterns, 'prefix'):
                param.requires_grad = True
                stats['trainable_params'] += param.numel()
            else:
                param.requires_grad = False
                stats['frozen_params'] += param.numel()
        
        if self.verbose:
            print(f"\n✅ 冻结所有参数")
            if exclude_patterns:
                print(f"   排除模式: {exclude_patterns}")
            print(f"   冻结: {stats['frozen_params']:,} / {stats['total_params']:,}")
        
        return stats
    
    def _unfreeze_all(self) -> Dict:
        """解冻所有参数"""
        count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            count += param.numel()
        
        if self.verbose:
            print(f"✅ 解冻所有参数: {count:,}")
        
        return {'unfrozen': count}
    
    def _print_freeze_summary(self, patterns: List[str], mode: str, stats: Dict):
        """打印冻结摘要"""
        total = stats['total_params']
        frozen = stats['frozen_params']
        trainable = stats['trainable_params']
        
        print(f"\n{'='*80}")
        print(f"{'冻结参数摘要':^80}")
        print(f"{'='*80}")
        print(f"匹配模式: {patterns}")
        print(f"匹配类型: {mode}")
        print(f"{'-'*80}")
        print(f"总参数:      {total:>15,} (100.00%)")
        print(f"冻结参数:    {frozen:>15,} ({100*frozen/total:>6.2f}%)")
        print(f"可训练参数:  {trainable:>15,} ({100*trainable/total:>6.2f}%)")
        
        if stats['skipped_bn']:
            print(f"跳过BN层:    {len(stats['skipped_bn']):>15} 层")
        
        print(f"{'='*80}\n")
        
        # 打印冻结的层
        if stats['frozen_layers']:
            print(f"✗ 冻结的层 ({len(stats['frozen_layers'])}):")
            self._print_layer_list(
                [(name, None) for name in stats['frozen_layers']],
                max_display=10)
        
        # 打印可训练的层
        if stats['trainable_layers']:
            print(f"\n✓ 可训练的层 ({len(stats['trainable_layers'])}):")
            self._print_layer_list(
                [(name, None) for name in stats['trainable_layers']],
                max_display=10)
        
        print(f"{'='*80}\n")
    
    def _print_parameter_groups(self, groups: Dict[str, List[nn.Parameter]]):
        """打印参数分组"""
        print(f"\n{'='*80}")
        print(f"{'参数分组':^80}")
        print(f"{'='*80}")
        print(f"{'分组名称':<30} {'层数':>15} {'参数数量':>20}")
        print(f"{'-'*80}")
        
        for group_name, params in groups.items():
            if params:
                num_params = sum(p.numel() for p in params)
                print(f"{group_name:<30} {len(params):>15} {num_params:>20,}")
        
        print(f"{'='*80}\n")
    
    def _print_layer_list(self, layers: List[tuple], max_display: int = 10):
        """打印层列表（带省略）"""
        if len(layers) <= max_display:
            for name, size in layers:
                if size is not None:
                    print(f"  • {name}: {size:,}")
                else:
                    print(f"  • {name}")
        else:
            half = max_display // 2
            for name, size in layers[:half]:
                if size is not None:
                    print(f"  • {name}: {size:,}")
                else:
                    print(f"  • {name}")
            print(f"  ... (省略 {len(layers) - max_display} 层)")
            for name, size in layers[-half:]:
                if size is not None:
                    print(f"  • {name}: {size:,}")
                else:
                    print(f"  • {name}")


class FreezeMixin:
    """
    参数冻结 Mixin 类
    
    任何继承此类的模型都自动获得参数冻结功能
    """
    
    def setup_freezer(self, verbose: bool = True):
        """初始化 Freezer（在 __init__ 中调用）"""
        self._freezer = ParameterFreezer(self, verbose=verbose)
    
    def freeze_parameters(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        mode: str = 'prefix',
        exclude_patterns: Optional[Union[str, List[str]]] = None,
        include_bn: bool = False
    ) -> Dict:
        """冻结参数（简化接口）"""
        if not hasattr(self, '_freezer'):
            self.setup_freezer()
        return self._freezer.freeze(patterns, mode, exclude_patterns, include_bn)
    
    def unfreeze_parameters(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        mode: str = 'prefix'
    ) -> Dict:
        """解冻参数"""
        if not hasattr(self, '_freezer'):
            self.setup_freezer()
        return self._freezer.unfreeze(patterns, mode)
    
    def get_parameter_groups(
        self,
        group_patterns: Dict[str, List[str]],
        mode: str = 'prefix'
    ) -> Dict[str, List[nn.Parameter]]:
        """获取参数分组"""
        if not hasattr(self, '_freezer'):
            self.setup_freezer()
        return self._freezer.get_parameter_groups(group_patterns, mode)
    
    def print_trainable_summary(self):
        """打印可训练参数摘要"""
        if not hasattr(self, '_freezer'):
            self.setup_freezer()
        self._freezer.print_trainable_summary()
    
    def inspect_freeze_patterns(self, max_depth: int = 3):
        """检查可用的冻结模式"""
        if not hasattr(self, '_freezer'):
            self.setup_freezer()
        return self._freezer.inspect_available_patterns(max_depth)


def freezeModelParameters(
    model: nn.Module,
    patterns: Optional[Union[str, List[str]]] = None,
    mode: str = 'prefix',
    verbose: bool = True
) -> Dict:
    """
    独立的冻结函数（不需要继承）
    
    Args:
        model: PyTorch模型
        patterns: 冻结模式
        mode: 匹配类型
        verbose: 是否打印信息
    
    Example:
        class Model(nn.Module):
            def __init__(
                self,
                ...,
                freeze_patterns: Optional[List[str]] = None
            ):
                super().__init__()
                
                # 构建模型
                self.yolo11seg = Yolo11Seg(model_type, num_classes, custom_postprocess)
                self.aux_head = YOLODetector(...)
                
                # 创建 Freezer 实例
                self.freezer = ParameterFreezer(self, verbose=True)
                
                # 应用冻结
                if freeze_patterns:
                    self.freezer.freeze(freeze_patterns)
            
            def freeze_parameters(self, patterns, **kwargs):
                # 暴露冻结接口
                return self.freezer.freeze(patterns, **kwargs)
            
            def unfreeze_parameters(self, patterns=None):
                # 暴露解冻接口
                return self.freezer.unfreeze(patterns)
            
            def print_trainable_summary(self):
                # 暴露摘要接口
                self.freezer.print_trainable_summary()
        
        1. 创建时冻结：
        model = Model(
            ...,
            freeze_patterns=[
                'backbone',
                'neck'
            ]
        )
    
    Returns:
        dict: 冻结统计
    """
    freezer = ParameterFreezer(model, verbose=verbose)
    return freezer.freeze(patterns, mode=mode)


def unfreezeModelParameters(
    model: nn.Module,
    patterns: Optional[Union[str, List[str]]] = None,
    verbose: bool = True
) -> Dict:
    """独立的解冻函数"""
    freezer = ParameterFreezer(model, verbose=verbose)
    return freezer.unfreeze(patterns)