# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t 3D stem 辅助模块。"""

import torch.nn as nn


def get_stem_func(name):
    """
    按名称取得 stem 模块类。
    """
    trans_funcs = {"x3d_stem": X3DStem, "basic_stem": ResNetBasicStem}
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class VideoModelStem(nn.Module):
    """
    视频 3D stem 模块。
    对一个或多个 pathway 的输入张量执行 Conv、BN、ReLU、MaxPool 等开头层操作。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        stem_func_name="basic_stem",
    ):
        """
                初始化 VideoModelStem。
                单 pathway 模型（C2D、I3D、Slow 等）的列表长度为 1，
                双 pathway 模型（SlowFast）的列表长度为 2。

                参数：
                    dim_in (list): 各输入 pathway 的通道数列表。
                    dim_out (list): 各 stem 卷积输出通道数列表。
                    kernel (list): stem 卷积核大小列表，顺序为 temporal、height、width。
                    stride (list): stem 卷积步幅列表，顺序为 temporal、height、width。
                    padding (list): stem 卷积 padding 列表，顺序为 temporal、height、width。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。
                    stem_func_name (string): 要应用到输入上的 stem 函数名称。

        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "输入 pathway 的维度配置不一致。"
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # 构建 stem 层。
        self._construct_stem(dim_in, dim_out, norm_module, stem_func_name)

    def _construct_stem(self, dim_in, dim_out, norm_module, stem_func_name):
        """
        为每条 pathway 创建对应的 stem 子模块并注册到当前模块中。
        """
        trans_func = get_stem_func(stem_func_name)

        for pathway in range(len(dim_in)):
            stem = trans_func(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        """
        对每条 pathway 的输入分别执行 stem 子模块。
        """
        assert (
            len(x) == self.num_pathways
        ), "输入张量不包含 {} 条 pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem 模块。
    先执行时空卷积、BN、ReLU，再执行时空池化。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
                初始化 ResNetBasicStem。

                参数：
                    dim_in (int): 输入通道数；RGB 通常为 3，光流通常为 2 或 3。
                    dim_out (int): stem 卷积输出通道数。
                    kernel (list): stem 卷积核大小，顺序为 temporal、height、width。
                    stride (list): stem 卷积步幅，顺序为 temporal、height、width。
                    padding (int): stem 卷积 padding，顺序为 temporal、height、width。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。

        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # 构建 stem 层。
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        """
        构建基础 stem 的 Conv3d、BN、ReLU 和 MaxPool3d。
        """
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

    def forward(self, x):
        """
        执行基础 stem 的前向计算。
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class X3DStem(nn.Module):
    """
    X3D 的 3D stem 模块。
    先做空间卷积，再做 depthwise temporal 卷积，随后接 BN 和 ReLU。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
                初始化 X3DStem。

                参数：
                    dim_in (int): 输入通道数；RGB 通常为 3，光流通常为 2 或 3。
                    dim_out (int): stem 卷积输出通道数。
                    kernel (list): stem 卷积核大小，顺序为 temporal、height、width。
                    stride (list): stem 卷积步幅，顺序为 temporal、height、width。
                    padding (int): stem 卷积 padding，顺序为 temporal、height、width。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。

        """
        super(X3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # 构建 stem 层。
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        """
        构建 X3D stem 的空间卷积、depthwise temporal 卷积、BN 和 ReLU。
        """
        self.conv_xy = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, self.kernel[1], self.kernel[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.padding[1], self.padding[2]),
            bias=False,
        )
        self.conv = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(self.kernel[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.padding[0], 0, 0),
            bias=False,
            groups=dim_out,
        )

        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        """
        执行 X3D stem 的前向计算。
        """
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
