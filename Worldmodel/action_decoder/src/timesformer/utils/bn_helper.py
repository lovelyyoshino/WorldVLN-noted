# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""BatchNorm 精确统计量更新工具。"""

import itertools
import torch


@torch.no_grad()
def compute_and_update_bn_stats(model, data_loader, num_batches=200):
    """用若干 batch 重新估计 BatchNorm 统计量，使验证更稳定。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """

    # 准备所有 BN 层。
    bn_layers = [
        m
        for m in model.modules()
        if any(
            (
                isinstance(m, bn_type)
                for bn_type in (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                )
            )
        )
    ]

    # 为了让 running stats 只反映当前 batch，
    # 这里禁用 momentum。
    # 中文说明：bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # 把 momentum 设为 1.0，用无动量方式统计。
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    # 计算精确 BN 统计需要迭代的次数。
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_square_mean = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    for ind, (inputs, _, _) in enumerate(
        itertools.islice(data_loader, num_batches)
    ):
        # 前向运行模型以更新 BN 统计量。
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].float().cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        model(inputs)

        for i, bn in enumerate(bn_layers):
            # 累计 BN 统计量。
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            # 中文说明：$E(x^2) = Var(x) + E(x)^2$.
            cur_square_mean = bn.running_var + bn.running_mean ** 2
            running_square_mean[i] += (
                cur_square_mean - running_square_mean[i]
            ) / (ind + 1)

    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_mean[i]
        # 中文说明：Var(x) = $E(x^2) - E(x)^2$.
        bn.running_var = running_square_mean[i] - bn.running_mean ** 2
        # 写回精确 BN 统计量。
        bn.momentum = momentum_actual[i]
