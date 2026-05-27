# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import copy
import torch
from collections import OrderedDict


def get_ema_model(model):
    """复制一个只用于推理的 EMA 模型，并冻结其参数梯度。"""
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False
    return ema_model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    用指数滑动平均更新 EMA 参数。
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # 这里的公式是 `ema = decay * ema + (1 - decay) * param`。
        # 未来可考虑只更新 requires_grad 的参数，减少位置编码之类常量的数值漂移。
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
