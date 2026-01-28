import math

import torch

__all__ = ['inverseSigmoid', 'biasInitWithProb']

def inverseSigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    """反 sigmoid 函数"""
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def biasInitWithProb(prior_prob=0.01):
    """根据给定的概率值初始化卷积层/全连接层的偏置值"""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init