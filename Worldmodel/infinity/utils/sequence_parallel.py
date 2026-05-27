# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.distributed as dist
from .comm.pg_utils import ProcessGroupManager
from .comm.comm import set_sp_comm_group, split_sequence, gather_sequence, all_to_all_comm
from .comm.operation import gather_forward_split_backward

class SequenceParallelManager:
    """中文说明：`SequenceParallelManager` 封装序列并行（sequence parallel）状态管理器中的状态和子模块。

    新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    _SP_GROUP = None
    _SP_SIZE = 0

    @staticmethod
    def sp_on():
        """中文说明：`sp_on` 实现序列并行（sequence parallel）状态管理器中的 `sp_on` 步骤，供训练、推理或调试流程复用。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return SequenceParallelManager._SP_GROUP is not None

    @staticmethod
    def init_sp(sp_size):
        """中文说明：`init_sp` 实现序列并行（sequence parallel）状态管理器中的 `init_sp` 步骤，供训练、推理或调试流程复用。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if SequenceParallelManager._SP_GROUP is not None:
            print("警告：sequence parallel group 已经初始化，跳过重复初始化")
            return

        if sp_size <= 1:
            print(f"警告：sequence parallel size 必须 > 1，当前 sp_size={sp_size}")
            return

        world_size = dist.get_world_size()
        assert world_size % sp_size == 0, f"world_size {world_size} 必须能被 sp_size({sp_size}) 整除"
        SequenceParallelManager._SP_SIZE = sp_size

        pm = ProcessGroupManager(
            world_size // sp_size,
            sp_size,
            dp_axis=0,
            sp_axis=1,
        )
        pm_group = pm.sp_group
        set_sp_comm_group(pm_group)
        SequenceParallelManager._SP_GROUP = pm_group
        return

    @staticmethod
    def get_sp_group():
        """中文说明：`get_sp_group` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return SequenceParallelManager._SP_GROUP

    @staticmethod
    def get_sp_size():
        """中文说明：`get_sp_size` 实现序列并行（sequence parallel）状态管理器中的 `get_sp_size` 步骤，供训练、推理或调试流程复用。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return SequenceParallelManager._SP_SIZE

    @staticmethod
    def get_sp_group_nums():
        # 例：sp_size=2 且总共 8 个 rank 时，group 数量为 4
        """中文说明：`get_sp_group_nums` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if SequenceParallelManager.sp_on():
            world_size = torch.distributed.get_world_size()
            return world_size // SequenceParallelManager._SP_SIZE
        else:
            return 0

    @staticmethod
    def get_sp_rank():
        """中文说明：`get_sp_rank` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if SequenceParallelManager.sp_on():
            global_rank = torch.distributed.get_rank()
            sp_rank = global_rank % SequenceParallelManager._SP_SIZE
            return sp_rank
        else:
            return 0

    def get_sp_group_rank():
        """中文说明：`get_sp_group_rank` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点看 sp_group、sp_size、sp_rank，它们决定序列维切分和通信分组。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if SequenceParallelManager.sp_on():
            global_rank = torch.distributed.get_rank()
            sp_group_rank = global_rank // SequenceParallelManager._SP_SIZE
            return sp_group_rank
        else:
            return 0

def sp_split_sequence_by_dim(seq, seqlen_dim=1) -> torch.Tensor:
    """
    按 `seqlen_dim` 拆分原始序列。
    """
    return split_sequence(seq, SequenceParallelManager.get_sp_group(), seqlen_dim, 'down')

def sp_gather_sequence_by_dim(seq, seqlen_dim=1) -> torch.Tensor:
    """
    沿 `seqlen_dim` 聚合以恢复原始序列。
    """
    return gather_sequence(seq, SequenceParallelManager.get_sp_group(), seqlen_dim, 'up')

def sp_all_to_all(ts, scatter_dim, gather_dim):
    """
        重排张量维度，例如从 `[raw_seq_len/sp_size, hidden_dim]` 变为 `[raw_seq_len, hidden_dim/sp_size]`。

        scatter_dim：拆分张量的维度。
        说明：gather_dim: the dimension to concatenate

    """

    return all_to_all_comm(ts, SequenceParallelManager.get_sp_group(), scatter_dim, gather_dim)
