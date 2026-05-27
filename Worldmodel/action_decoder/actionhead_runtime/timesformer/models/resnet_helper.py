# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""视频模型中的 ResNet 基础模块。"""

import torch
import torch.nn as nn

from timesformer.models.nonlocal_helper import Nonlocal
from timesformer.models.operators import SE, Swish

from torch import einsum
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from torch.nn.modules.module import Module
# 保留的上游调试/兼容代码：from torch.nn.modules.linear import _LinearWithBias
from torch.nn.modules.activation import MultiheadAttention

import numpy as np

def get_trans_func(name):
    """
    按名称取得残差块内部使用的变换模块类。
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
        "x3d_transform": X3DTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]




class BasicTransform(nn.Module):
    """
    基础残差变换：Tx3x3 后接 1x3x3，其中 T 是 temporal kernel 大小。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1x1=None,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
                参数：
                    dim_in (int): 输入通道数。
                    dim_out (int): 输出通道数。
                    temp_kernel_size (int): basic block 中第一个卷积的 temporal kernel 大小。
                    stride (int): 空间维度上的步幅。
                    dim_inner (None): BasicTransform 不使用内部通道数。
                    num_groups (int): 卷积分组数；BasicTransform 中始终为 1。
                    stride_1x1 (None): BasicTransform 不使用 stride_1x1。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。

        """
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, norm_module)

    def _construct(self, dim_in, dim_out, stride, norm_module):
        """
        构建 BasicTransform 的两个卷积分支。
        """
        # 中文说明：Tx3x3, BN, ReLU。
        self.a = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=[self.temp_kernel_size, 3, 3],
            stride=[1, stride, stride],
            padding=[int(self.temp_kernel_size // 2), 1, 1],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        # 1x3x3, BN。
        self.b = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False,
        )
        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )

        self.b_bn.transform_final_bn = True

    def forward(self, x):
        """
        依次执行两个卷积子层并返回残差分支输出。
        """
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class X3DTransform(nn.Module):
    """
    X3D 残差变换：1x1x1、Tx3x3（可 channelwise）、1x1x1。
    中间的 Tx3x3 输出可以接可选的 SE（squeeze-excitation）注意力，
    T 是 temporal kernel 大小，默认通常为 3。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        se_ratio=0.0625,
        swish_inner=True,
        block_idx=0,
    ):
        """
                参数：
                    dim_in (int): 输入通道数。
                    dim_out (int): 输出通道数。
                    temp_kernel_size (int): bottleneck 中间卷积的 temporal kernel 大小。
                    stride (int): 空间维度上的步幅。
                    dim_inner (int): block 内部通道数。
                    num_groups (int): 卷积分组数；1 表示标准 ResNet，>1 常用于 ResNeXt。
                    stride_1x1 (bool): 为 True 时 stride 放在 1x1 conv，否则放在 3x3 conv。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    dilation (int): 空洞卷积 dilation 大小。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。
                    se_ratio (float): >0 时对 Tx3x3 conv 输出应用 SE，SE 通道数为该比例。
                    swish_inner (bool): 为 True 时中间层用 Swish，否则用 ReLU。

        """
        super(X3DTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._se_ratio = se_ratio
        self._swish_inner = swish_inner
        self._stride_1x1 = stride_1x1
        self._block_idx = block_idx
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        """
        构建 X3DTransform 的 1x1x1、Tx3x3、1x1x1 三段结构。
        """
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 中文说明：1x1x1, BN, ReLU。
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 中文说明：Tx3x3, BN, ReLU。
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [self.temp_kernel_size, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[int(self.temp_kernel_size // 2), dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )

        # 按 block 位置决定是否应用 SE attention。
        use_se = True if (self._block_idx + 1) % 2 else False
        if self._se_ratio > 0.0 and use_se:
            self.se = SE(dim_inner, self._se_ratio)

        if self._swish_inner:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN。
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        """
        按子模块顺序执行 X3DTransform。
        """
        for block in self.children():
            x = block(x)
        return x

class BottleneckTransform(nn.Module):
    """
    Bottleneck 残差变换：Tx1x1、1x3x3、1x1x1。
    其中 T 是 temporal kernel 大小。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
                参数：
                    dim_in (int): 输入通道数。
                    dim_out (int): 输出通道数。
                    temp_kernel_size (int): bottleneck 第一个卷积的 temporal kernel 大小。
                    stride (int): 空间维度上的步幅。
                    dim_inner (int): block 内部通道数。
                    num_groups (int): 卷积分组数；1 表示标准 ResNet，>1 常用于 ResNeXt。
                    stride_1x1 (bool): 为 True 时 stride 放在 1x1 conv，否则放在 3x3 conv。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    dilation (int): 空洞卷积 dilation 大小。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。

        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        """
        构建 BottleneckTransform 的三个卷积阶段。
        """
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 保留的上游调试/兼容代码：print(str1x1, str3x3)

        # 中文说明：Tx1x1, BN, ReLU。
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 中文说明：1x3x3, BN, ReLU。
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN。
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        """
        逐层执行 bottleneck 残差分支。
        """
        # 显式执行每一层。
        # 中文说明：Branch2a。
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # 中文说明：Branch2b。
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # 中文说明：Branch2c。
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    ResNet 残差块。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
                构建一个残差块。更多背景见：
                    说明：Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
                    说明："Deep residual learning for image recognition."
                    引用/来源：https://arxiv.org/abs/1512.03385
                参数：
                    dim_in (int): 输入通道数。
                    dim_out (int): 输出通道数。
                    temp_kernel_size (int): bottleneck 中间卷积的 temporal kernel 大小。
                    stride (int): 空间维度上的步幅。
                    trans_func (string): 用来构建残差分支的变换函数。
                    dim_inner (int): block 内部通道数。
                    num_groups (int): 卷积分组数；1 表示标准 ResNet，>1 常用于 ResNeXt。
                    stride_1x1 (bool): 为 True 时 stride 放在 1x1 conv，否则放在 3x3 conv。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    dilation (int): 空洞卷积 dilation 大小。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。
                    drop_connect_rate (float): block 被随机丢弃的基础比例，通常随深度增加。

        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        """
        构建残差块的 shortcut 分支和主变换分支。
        """
        # 如果通道数或空间分辨率改变，就用投影 shortcut。
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def _drop_connect(self, x, drop_ratio):
        """对输入 x 应用 dropconnect。"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty(
            [x.shape[0], 1, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    def forward(self, x):
        """
        执行残差分支、可选 dropconnect、shortcut 相加和 ReLU。
        """
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """
        3D ResNet 的一个 stage。
        它既支持单 pathway 输入（C2D、I3D、Slow），也支持多 pathway 输入（SlowFast）。
        更多背景见：

            说明：Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
            说明："SlowFast networks for video recognition."
            引用/来源：https://arxiv.org/pdf/1812.03982.pdf

    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        nonlocal_inds,
        nonlocal_group,
        nonlocal_pool,
        dilation,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
        drop_connect_rate=0.0,
    ):
        """
                初始化 ResStage。
                ResStage 会构建 p 条 pathway，其中 p >= 1。
                参数：
                    dim_in (list): p 条 pathway 的输入通道数列表。
                    dim_out (list): p 条 pathway 的输出通道数列表。
                    temp_kernel_sizes (list): p 条 pathway 中 bottleneck 的 temporal kernel 列表。
                    stride (list): p 条 pathway 的空间步幅列表。
                    num_blocks (list): 每条 pathway 中 block 数量列表。
                    dim_inner (list): 每条 pathway 的内部通道数列表。
                    num_groups (list): 每条 pathway 的卷积分组数；1 表示标准 ResNet，>1 常用于 ResNeXt。
                    num_block_temp_kernel (list): 前多少个 block 使用 temp_kernel_sizes，
                        剩余 block 的 temporal kernel 填 1。
                    nonlocal_inds (list): 为空时不加 Non-local；非空时在指定 block 后添加。
                    dilation (list): 每条 pathway 的空洞卷积 dilation 大小。
                    nonlocal_group (list): 每条 pathway 的 Non-local 分组数；用于在执行
                        Non-local 前把 temporal 维折叠到 batch 维。
                        引用/来源：https://github.com/facebookresearch/video-nonlocal-net.
                    instantiation (string): Non-local 层的归一化方式，支持 "dot_product" 和 "softmax"。
                    trans_func_name (string): stage 中使用的变换函数名称。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。
                    drop_connect_rate (float): block 被随机丢弃的基础比例，通常随深度增加。

        """
        super(ResStage, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
        norm_module,
    ):
        """
        为每条 pathway 逐个创建 ResBlock，并按配置插入 Non-local 模块。
        """
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # 取得残差分支使用的变换函数。
                trans_func = get_trans_func(trans_func_name)
                # 构建当前残差块。
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(
                        dim_out[pathway],
                        dim_out[pathway] // 2,
                        nonlocal_pool[pathway],
                        instantiation=instantiation,
                        norm_module=norm_module,
                    )
                    self.add_module(
                        "pathway{}_nonlocal{}".format(pathway, i), nln
                    )

    def forward(self, inputs):
        """
        对每条 pathway 顺序执行 stage 内的 ResBlock 和可选 Non-local 模块。
        """
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
                if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                    nln = getattr(
                        self, "pathway{}_nonlocal{}".format(pathway, i)
                    )
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[pathway] > 1:
                        # 将 temporal 维折叠到 batch 维，便于分组执行 Non-local。
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(
                            b * self.nonlocal_group[pathway],
                            t // self.nonlocal_group[pathway],
                            c,
                            h,
                            w,
                        )
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[pathway] > 1:
                        # 将折叠后的 batch 维恢复回 temporal 维。
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)

        return output
