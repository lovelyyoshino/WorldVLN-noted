# Copyright 2020 Ross Wightman
# 带 SAME padding 的 Conv2d 工具。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import math
from typing import List, Tuple
# 曾从 padding 模块导入 pad_same 和 get_padding_value。

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    """按卷积参数为输入 ``x`` 动态补齐 TensorFlow 风格的 ``SAME`` padding。"""
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def get_same_padding(x: int, k: int, s: int, d: int):
    """计算单个维度上 TensorFlow 风格 ``SAME`` 卷积需要补的总 padding。"""
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    """解析 padding 参数，并返回实际 padding 值以及是否需要动态 padding。"""
    dynamic = False
    if isinstance(padding, str):
        # 字符串 padding 会在这里统一换算成实际数值，共三种处理方式。
        padding = padding.lower()
        if padding == 'same':
            # TF 兼容的 SAME padding，可能增加运行时间和 GPU 显存开销。
            if is_static_pad(kernel_size, **kwargs):
                # 静态 padding 情况下没有额外运行期开销。
                padding = get_padding(kernel_size, **kwargs)
            else:
                # 动态 SAME padding 会在运行时带来额外开销。
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # VALID padding 等价于 padding=0。
            padding = 0
        else:
            # 默认退回 PyTorch 风格的近似 same 对称 padding。
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    """先动态补齐 ``SAME`` padding，再调用 ``F.conv2d`` 执行 2D 卷积。"""
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """TensorFlow 风格 ``SAME`` padding 的 2D 卷积包装层。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        """初始化卷积层参数，但把父类 padding 固定为 0，由 forward 动态补齐。"""
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        """对输入 ``x`` 执行带动态 SAME padding 的卷积。"""
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    """根据 padding 配置创建普通 ``nn.Conv2d`` 或动态 ``Conv2dSame``。"""
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
