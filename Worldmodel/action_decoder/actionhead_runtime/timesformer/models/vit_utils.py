# Copyright 2020 Ross Wightman
# ViT 相关的通用工具函数。

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F

from timesformer.models.helpers import load_pretrained
from .build import MODEL_REGISTRY
from itertools import repeat

# 当前文件中的 PyTorch 主版本和次版本。
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """在 no_grad 环境下用截断正态分布原地初始化 ``tensor``。"""

    def norm_cdf(x):
        """计算标准正态分布的累积分布函数。"""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean 距离 [a, b] 超过 2 个 std，nn.init.trunc_normal_ "
                      "生成的取值分布可能不正确。",
                      stacklevel=2)

    with torch.no_grad():
        # 先在截断后的均匀分布中采样，再通过正态分布的反 CDF 转换。
        # 计算上下界对应的 CDF 值。
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 在 [l, u] 中均匀填充，然后平移到 [2l-1, 2u-1]。
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # 使用反 CDF 变换得到截断后的标准正态分布。
        tensor.erfinv_()

        # 转换到目标 mean 和 std。
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # 裁剪到合法范围内。
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # 形状/映射说明：type: (Tensor, float, float, float, float) -> Tensor
    """用截断正态分布中的值填充输入 Tensor。

        这些值等价于从正态分布
        :math:`\mathcal{N}(\text{mean}, \text{std}^2)` 中采样，并把
        :math:`[a, b]` 范围外的值重采样到范围内。当
        :math:`a \leq \text{mean} \leq b` 时，这种采样方式效果最好。

        参数：
            tensor: n 维 `torch.Tensor`。
            mean: 正态分布均值。
            std: 正态分布标准差。
            a: 最小截断值。
            b: 最大截断值。

        示例：
            说明：>>> w = torch.empty(3, 5)
            说明：>>> nn.init.trunc_normal_(w)

    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# 来自 PyTorch 内部工具的简化版本。
def _ntuple(n):
    """返回一个函数，用于把标量扩展成长度为 n 的 tuple。"""

    def parse(x):
        """如果 ``x`` 已可迭代就原样返回，否则复制成长度为 n 的 tuple。"""
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    """计算卷积使用的对称 padding 大小。"""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def get_padding_value(padding, kernel_size, **kwargs):
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

def get_same_padding(x: int, k: int, s: int, d: int):
    """计算单个维度上 TensorFlow 风格 ``SAME`` 卷积需要补的总 padding。"""
    return max((int(math.ceil(x // s)) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    """判断给定卷积参数是否能用静态 SAME padding 表达。"""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# 按卷积参数为输入 x 动态补齐 SAME padding。
# 保留的上游调试/兼容代码：def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
def pad_same(x, k, s, d=(1, 1), value= 0):
    """按卷积参数为输入 ``x`` 动态补齐 TensorFlow 风格的 ``SAME`` padding。"""
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def adaptive_pool_feat_mult(pool_type='avg'):
    """返回自适应池化输出通道倍率；catavgmax 会把 avg 和 max 拼接成 2 倍。"""
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """按样本执行 DropPath（Stochastic Depth），常用于残差分支主路径。

    这个实现和 EfficientNet 等网络里的 DropConnect 类似，但论文中的
    ``Drop Connect`` 指的是另一种 dropout。这里保留 ``drop path`` 这个命名，
    避免把层名 DropConnect 和参数 survival rate 混在一起。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 支持不同维度的 tensor，不只支持 2D ConvNet。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化。
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """按样本执行 DropPath（Stochastic Depth）的模块包装。"""

    def __init__(self, drop_prob=None):
        """保存 DropPath 的丢弃概率。"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """按当前训练/评估模式对输入 ``x`` 应用 DropPath。"""
        return drop_path(x, self.drop_prob, self.training)
