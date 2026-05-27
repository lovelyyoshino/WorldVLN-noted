"""Linear 层的替代实现。"""
import torch
import torch.nn.functional as F
from torch import nn as nn

class Linear(nn.Linear):
    """在 TorchScript 下自动把权重和 bias 转成输入 dtype 的 Linear 层。"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """对 ``input`` 执行线性变换，并在脚本模式下对齐 dtype。"""
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)
