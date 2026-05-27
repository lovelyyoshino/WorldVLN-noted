# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

# 代码/形状说明：from timm.models.layers import DropPath
import torch

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """按样本执行 Stochastic Depth。

    它不是把单个元素置零，而是把残差分支整条路径随机丢弃。若保留概率为
    `keep_prob = 1 - drop_prob`，则在 `scale_by_keep=True` 时会再除以 `keep_prob`，
    以保持输出期望不变。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 支持不同维度张量，不只服务于 2D ConvNet。
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """`drop_path` 的模块封装，便于插入残差块。"""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        """记录路径丢弃概率与是否执行期望缩放。"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """在训练态对输入张量应用随机路径丢弃。"""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        """返回模块摘要，便于打印网络结构。"""
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
