# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""BatchNorm (BN) 工具函数和自定义小批量 BN 实现。"""

from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.function import Function

import timesformer.utils.distributed as du


def get_norm(cfg):
    """
        根据配置返回模型要使用的归一化层类型。

        参数：
            cfg (CfgNode): 模型构建配置，字段含义见配置文件中的说明。

        返回：
            nn.Module: 归一化层类或带参数的构造函数。

    """
    if cfg.BN.NORM_TYPE == "batchnorm":
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == "sub_batchnorm":
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == "sync_batchnorm":
        return partial(
            NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES
        )
    else:
        raise NotImplementedError(
            "不支持归一化类型 {}".format(cfg.BN.NORM_TYPE)
        )


class SubBatchNorm3d(nn.Module):
    """
    将一个 batch 按 batch 维切成多份，并分别计算 3D BatchNorm 统计量。

    标准 BN 会在一个 GPU 上的全部样本上统计均值和方差。有些训练方式
    （例如 multigrid training）希望只在子集样本上统计。这个模块会把
    batch 维分成 N 份，对每一份单独做 BN；评估前再把各份 running stats
    聚合回一个普通 BN。
    """

    def __init__(self, num_splits, **args):
        """
                初始化分组 BatchNorm 需要的普通 BN 和 split BN。

                参数：
                    num_splits (int): batch 维要切分的份数。
                    args (dict): 传给 ``nn.BatchNorm3d`` 的其他参数。

        """
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        num_features = args["num_features"]
        # 只保留一组可学习的 weight 和 bias。
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = num_features * num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
                把多组均值和方差合并成一组 running stats。

                参数：
                    means (tensor): 各 split 的均值。
                    stds (tensor): 各 split 的方差。
                    n (int): split 的数量。

        """
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        聚合 ``running_mean`` 和 ``running_var``，通常在 eval 前调用。
        """
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        """按训练或评估模式对输入 ``x`` 执行 SubBatchNorm3d。"""
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x


class GroupGather(Function):
    """
    在本地进程/GPU 分组内做 all_gather 并汇总统计量。
    """

    @staticmethod
    def forward(ctx, input, num_sync_devices, num_groups):
        """
        前向阶段收集同一同步分组内各进程/GPU 的统计量。
        """
        ctx.num_sync_devices = num_sync_devices
        ctx.num_groups = num_groups

        input_list = [
            torch.zeros_like(input) for k in range(du.get_local_size())
        ]
        dist.all_gather(
            input_list, input, async_op=False, group=du._LOCAL_PROCESS_GROUP
        )

        inputs = torch.stack(input_list, dim=0)
        if num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // num_sync_devices
            inputs = inputs[
                group_idx
                * num_sync_devices : (group_idx + 1)
                * num_sync_devices
            ]
        inputs = torch.sum(inputs, dim=0)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向阶段收集同一同步分组内各进程/GPU 的梯度。
        """
        grad_output_list = [
            torch.zeros_like(grad_output) for k in range(du.get_local_size())
        ]
        dist.all_gather(
            grad_output_list,
            grad_output,
            async_op=False,
            group=du._LOCAL_PROCESS_GROUP,
        )

        grads = torch.stack(grad_output_list, dim=0)
        if ctx.num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // ctx.num_sync_devices
            grads = grads[
                group_idx
                * ctx.num_sync_devices : (group_idx + 1)
                * ctx.num_sync_devices
            ]
        grads = torch.sum(grads, dim=0)
        return grads, None, None


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    """朴素版同步 3D BatchNorm，在本地 GPU 分组间同步均值和方差。"""

    def __init__(self, num_sync_devices, **args):
        """
                初始化同步 3D BatchNorm 的设备分组信息。

                参数：
                    num_sync_devices (int): 每个同步分组中的设备数量。
                    args (dict): 传给 ``nn.BatchNorm3d`` 的其他参数。

        """
        self.num_sync_devices = num_sync_devices
        if self.num_sync_devices > 0:
            assert du.get_local_size() % self.num_sync_devices == 0, (
                du.get_local_size(),
                self.num_sync_devices,
            )
            self.num_groups = du.get_local_size() // self.num_sync_devices
        else:
            self.num_sync_devices = du.get_local_size()
            self.num_groups = 1
        super(NaiveSyncBatchNorm3d, self).__init__(**args)

    def forward(self, input):
        """在训练时跨设备同步统计量，并返回归一化后的 ``input``。"""
        if du.get_local_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm 不支持空输入"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (
            1.0 / self.num_sync_devices
        )

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias
