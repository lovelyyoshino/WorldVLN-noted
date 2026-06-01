# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""自定义算子和小型网络模块。

中文导读：
    本文件包含 SlowFast/X3D 用到的小工具：
        - ``Swish`` / ``SwishEfficient``：Swish 激活函数及其节省显存的 autograd 实现。
          公式：Swish(x) = x * sigmoid(x)。
        - ``SE``：Squeeze-and-Excitation 通道注意力模块（用于 X3D 路径），由 AvgPool +
          两层 1x1x1 conv + sigmoid 组成，把通道权重乘回输入。

    WorldVLN 动作解码器主路径仅使用 ``vit.py`` 中的 ``Block`` 与 ``Attention``，并不会
    调用 ``Swish`` 或 ``SE``；保留它们是为了让上游 ``video_model_builder`` 能正常 import。
"""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish 激活函数模块，计算公式为 ``x * sigmoid(x)``。"""

    def __init__(self):
        """初始化 Swish 激活模块。"""
        super(Swish, self).__init__()

    def forward(self, x):
        """对输入 ``x`` 应用节省显存的 Swish 实现。"""
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """带自定义反向传播的 Swish 激活函数。"""

    @staticmethod
    def forward(ctx, x):
        """前向计算 ``x * sigmoid(x)``，并保存 ``x`` 供反向传播使用。"""
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """根据 Swish 的导数把上游梯度传回输入。"""
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) 模块：AvgPool、FC、激活、FC、Sigmoid。"""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
                根据宽度倍率把通道数取整到 divisor 的倍数。

                参数：
                    width (int): 输入通道数。
                    multiplier (float): 通道数缩放倍率。
                    min_width (int): 缩放后的最小通道数。
                    divisor (int): 新通道数需要能被该值整除。

        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, dim_in, ratio, relu_act=True):
        """
                初始化 SE 模块中的池化、两层 1x1x1 卷积和激活函数。

                参数：
                    dim_in (int): 输入通道数。
                    ratio (float): squeeze 阶段的通道压缩比例。
                    relu_act (bool): 是否使用 ReLU；为 False 时使用 Swish。

        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)

        self.fc2_sig = nn.Sigmoid()

    def forward(self, x):
        """计算 SE 权重，并把它乘回原始输入 ``x``。"""
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x
