# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import torch
from timm.loss import SoftTargetCrossEntropy

from timm.models.layers import DropPath

from .infinity import Infinity, sample_with_top_k_top_p_also_inplace_modifying_logits_

def _ex_repr(self):
    """把模块公开属性整理成紧凑字符串，方便日志里快速查看配置。"""
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy):  # 统一这些 loss 的打印格式，避免日志过长。
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'

DropPath.__repr__ = lambda self: f'{type(self).__name__}(...)'

alias_dict = {}
for d in range(6, 40+2, 2):
    alias_dict[f'd{d}'] = f'infinity_d{d}'
alias_dict_inv = {v: k for k, v in alias_dict.items()}
