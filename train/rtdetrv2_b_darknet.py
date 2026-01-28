from typing import List, Tuple, Optional, Any, Dict, Dict, Sequence, Union, Callable

import torch

from mmengine.model import BaseModel
from mmengine.registry import DATASETS, MODELS, METRICS, FUNCTIONS, HOOKS
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.hooks import Hook
from mmengine.runner import Runner

from haua.datasets import RTDETRInstanceDataset, BatchImageCollateFuncion, get_train_transforms
from haua.models import RTDETRv2
from haua.criteriones import RTDETRCriterionv2, HungarianMatcher
from haua.tools import hauarun


rtdetr_collate_fn = BatchImageCollateFuncion(
    scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], stop_epoch=71)


class RTDETRCollateEpochHook(Hook):
    """
    用于在训练过程中动态更新 Collate Function 的 epoch 属性。
    """
    def __init__(self):
        super().__init__()

    def before_train_epoch(self, runner: Runner):
        # 获取当前的 DataLoader
        dataloader = runner.train_loop.dataloader # type: ignore
        
        # 获取 collate_fn
        # 注意：如果使用了多进程，collate_fn 是被序列化到 worker 里的，
        # 但在主进程修改它通常会通过 copy-on-write 或共享内存机制生效，
        # 或者对于 persistent_workers=False 的情况，每个 epoch 重新创建 loader 时会生效。
        # 对于 persistent_workers=True，这种修改可能需要更深层的 hack，
        # 但在大多数标准 MMEngine 设置下，修改 runner 里的引用是有效的。
        collate_fn = dataloader.collate_fn

        # 检查是否有 set_epoch 方法并调用
        if hasattr(collate_fn, 'set_epoch'):
            # runner.epoch 从 0 开始
            collate_fn.set_epoch(runner.epoch) # type: ignore
            # 打印日志方便调试 (可选)
            # runner.logger.info(f"Updated collate_fn epoch to {runner.epoch}")
            
        # 处理可能被 functools.partial 包裹的情况
        elif hasattr(collate_fn, 'func') and hasattr(collate_fn.func, 'set_epoch'): # type: ignore
            collate_fn.func.set_epoch(runner.epoch) # type: ignore


class RTDETRCOCO(RTDETRInstanceDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        return_masks: bool = False
    ):
        super().__init__(
            root = root,
            ann_file = ann_file,
            # transforms = get_train_transforms(640, mode='letterbox'),
            transforms = get_train_transforms(640),
            return_masks = return_masks,
            remap_mscoco_category = True)


class TrainRTDETRv2(BaseModel):
    """MMEngine 封装的多任务模型"""
    def __init__(
        self,
        model_config: dict,
        loss_config: dict,
        init_checkpoint = None
    ):
        super().__init__()
        self.matcher = HungarianMatcher(
            weight_dict = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            alpha = 0.25,
            gamma = 2.0)
        self.model = RTDETRv2(**model_config)
        if init_checkpoint is not None:
            init_static_dict = torch.load(init_checkpoint)
            self.model.load_state_dict(init_static_dict)
        self.loss_module = RTDETRCriterionv2(self.matcher, **loss_config)

    def forward(self, inputs, data_samples=None, mode: str = "tensor"): # type: ignore
        """
        Args:
            batch_inputs: 输入张量 (B, ...)
            data_samples: list[dict]，每个dict包含标签
            mode: "tensor" | "loss" | "predict"
        """
        preds = self.model(inputs, head_kwargs={'targets': data_samples})
        if mode == "tensor":
            return preds
        elif mode == "loss":
            assert data_samples is not None, "训练时必须提供 data_samples"
            return self.loss(preds[2], data_samples) # type: ignore
        # elif mode == "predict":
        #     return self.predict(preds)
        else:
            raise ValueError(f"Invalid mode {mode}")

    def loss(self,
        outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        data_samples: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss = self.loss_module(outputs, data_samples)

        return loss

    # def predict(self, outputs):
    #     B = outputs['gender'].shape[0]
    #     results = []
    #     for i in range(B):
    #         results.append(self.output_parsing(outputs['gender'][i], outputs['age'][i]))

    #     return results


if __name__ == '__main__':
    DATASETS.register_module(module=RTDETRCOCO)
    MODELS.register_module(module=TrainRTDETRv2)
    FUNCTIONS.register_module(name='rtdetr_collate_fn', module=rtdetr_collate_fn) # type: ignore
    HOOKS.register_module(module=RTDETRCollateEpochHook)
    hauarun()