# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t 分类头辅助模块。"""

import torch
import torch.nn as nn

class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 的 3D 分类头。
    训练时输入通常已经池化到 1x1x1，本层相当于做全连接投影。
    测试时如果输入仍大于 1x1x1，则以全卷积方式在每个位置上投影。
    如果有多个 pathway 输入，会先分别池化，再在通道维拼接。
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
                初始化 ResNetBasicHead。
                这个分类头接收 p 个 pathway 输入，其中 p >= 1。

                参数：
                    dim_in (list): p 个输入 pathway 的通道数列表。
                    num_classes (int): 最终分类类别数。
                    pool_size (list): p 个时空池化核大小列表，顺序为 temporal、height、width。
                    dropout_rate (float): dropout 比例；为 0.0 时不使用 dropout。
                    act_func (string): 输出激活函数，支持 'softmax' 或 'sigmoid'。

        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway 维度不一致。"
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # 以全卷积推理兼容的方式执行 FC；Linear 的初始化 std 与卷积层不同。
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # 评估和测试阶段使用的输出激活。
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "不支持 {} 作为激活函数。".format(act_func)
            )

    def forward(self, inputs):
        """
        对一个或多个 pathway 特征做池化、拼接和分类投影。
        """
        assert (
            len(inputs) == self.num_pathways
        ), "输入张量不包含 {} 条 pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C)。
        x = x.permute((0, 2, 3, 4, 1))
        # 按需执行 dropout。
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # 测试阶段执行全卷积式推理，并对时空位置取平均。
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class X3DHead(nn.Module):
    """
    X3D 分类头。
    训练时输入通常已经池化到 1x1x1，本层相当于做全连接投影。
    测试时如果输入仍大于 1x1x1，则以全卷积方式在每个位置上投影。
    X3D 当前实现只处理单 pathway 输入。
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
                初始化 X3DHead。
                输入是 5 维特征张量，形状为 BxCxTxHxW。

                参数：
                    dim_in (float): 输入通道数 C。
                    num_classes (int): 最终分类类别数。
                    pool_size (float): TxHxW 维度上的时空池化核大小。
                    dropout_rate (float): dropout 比例；为 0.0 时不使用 dropout。
                    act_func (string): 输出激活函数，支持 'softmax' 或 'sigmoid'。
                    inplace_relu (bool): 为 True 时在原张量上计算 ReLU，减少额外内存。
                    eps (float): batch norm 的 epsilon。
                    bn_mmt (float): batch norm 动量；PyTorch 中的含义是 Caffe2 的 1 - momentum。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。
                    bn_lin5_on (bool): 为 True 时在分类器前对特征做归一化。

        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):
        """
        构建 X3DHead 内部的 1x1x1 卷积、池化、dropout 和分类投影层。
        """

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # 以全卷积推理兼容的方式执行 FC；Linear 的初始化 std 与卷积层不同。
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # 评估和测试阶段使用的输出激活。
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "不支持 {} 作为激活函数。".format(self.act_func)
            )

    def forward(self, inputs):
        """
        对单 pathway 的 X3D 特征做分类头前向计算。
        """
        # 当前设计中，X3D head 只支持单 pathway 输入。
        assert len(inputs) == 1, "输入张量不包含 1 条 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C)。
        x = x.permute((0, 2, 3, 4, 1))
        # 按需执行 dropout。
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # 测试阶段执行全卷积式推理，并对时空位置取平均。
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x
