# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Non-local 模块辅助代码。"""

import torch
import torch.nn as nn


class Nonlocal(nn.Module):
    """
    构建 Non-local Neural Network 模块，用来捕获长距离依赖。
    它会把某个位置的输出表示成所有位置特征的加权和，因此可以让远距离
    时空位置互相交换信息。这个模块可以插入很多计算机视觉网络中。
    论文链接：https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
                参数：
                    dim (int): 输入特征的通道数。
                    dim_inner (int): Non-local 模块内部使用的通道数。
                    pool_size (list): 时空池化核大小，顺序为 temporal、height、width；
                        默认为 None，表示不使用池化。
                    instantiation (string): 相似度矩阵的归一化方式：
                        "dot_product" 表示用位置数量做缩放，"softmax" 表示用 Softmax。
                    zero_init_final_conv (bool): 为 True 时，将最后的卷积初始化为零。
                    zero_init_final_norm (bool): 为 True 时，将最后的 BatchNorm 初始化为零。
                    norm_module (nn.Module): 归一化层类型，默认是 nn.BatchNorm3d。

        """
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(
        self, zero_init_final_conv, zero_init_final_norm, norm_module
    ):
        """
        构建 theta、phi、g 三个投影分支，以及输出投影和可选池化层。
        """
        # 三个 1x1x1 卷积分支：theta、phi 和 g。
        self.conv_theta = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # 最后的输出卷积，用于把内部通道数变回输入通道数。
        self.conv_out = nn.Conv3d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        # 记录是否对最终卷积做零初始化。
        self.conv_out.zero_init = zero_init_final_conv

        # 待办：未来可把字段名改成 `norm`，这里先保持旧命名兼容检查点。
        self.bn = norm_module(
            num_features=self.dim,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # 记录是否对最终 bn 做零初始化。
        self.bn.transform_final_bn = zero_init_final_norm

        # 可选的时空池化，用来降低后续注意力计算量。
        if self.use_pool:
            self.pool = nn.MaxPool3d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0, 0],
            )

    def forward(self, x):
        """
        对输入特征执行 Non-local 注意力，并与原输入做残差相加。
        """
        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)

        # 对时空维度做池化以减少计算量。
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # 形状/映射说明：(N, C, TxHxW) * (N, C, TxHxW) => (N, TxHxW, TxHxW)。
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))
        # 原始 Non-local 论文中，affinity tensor 主要有两种归一化方式：
        #   1) Softmax 归一化。
        #   2) dot_product 归一化。
        if self.instantiation == "softmax":
            # 在 softmax 前按通道数缩放 theta_phi。
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "未知归一化类型 {}".format(self.instantiation)
            )

        # 形状/映射说明：(N, TxHxW, TxHxW) * (N, C, TxHxW) => (N, C, TxHxW)。
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))

        # 形状/映射说明：(N, C, TxHxW) => (N, C, T, H, W)。
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        return x_identity + p
