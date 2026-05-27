# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
from collections import OrderedDict

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    让 EMA 模型朝当前模型权重移动一步。
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # 待办：可考虑只作用于 require_grad=True 的参数，避免 pos_embed 出现很小的数值变化。
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    批量设置模型所有参数的 `requires_grad` 标志。
    """
    for p in model.parameters():
        p.requires_grad = flag