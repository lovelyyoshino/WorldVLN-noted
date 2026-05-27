# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import math
import torch
import torch.nn as nn
import torch.distributed as dist

import infinity.models.videovae.utils.diffdist.functional as distops

class ContextParallelUtils:
    """中文说明：`ContextParallelUtils` 封装 VideoVAE 上下文并行（context parallel）通信工具中的状态和子模块。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    _CONTEXT_PARALLEL_GROUP = None
    _CONTEXT_PARALLEL_SIZE = 0
    _CONTEXT_PARALLEL_ON = False

    """
    {
        "cp_size": 2,
    }
    """
    CP_CONFIG = None

    @staticmethod
    def set_cp_on(on=True):
        """中文说明：`set_cp_on` 实现 VideoVAE 上下文并行（context parallel）通信工具中的 `set_cp_on` 步骤，供训练、推理或调试流程复用。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        ContextParallelUtils._CONTEXT_PARALLEL_ON = on

    @staticmethod
    def cp_on():
        """中文说明：`cp_on` 实现 VideoVAE 上下文并行（context parallel）通信工具中的 `cp_on` 步骤，供训练、推理或调试流程复用。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return ContextParallelUtils._CONTEXT_PARALLEL_ON

    @staticmethod
    def get_cp_cfg():
        """中文说明：`get_cp_cfg` 实现 VideoVAE 上下文并行（context parallel）通信工具中的 `get_cp_cfg` 步骤，供训练、推理或调试流程复用。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return ContextParallelUtils.CP_CONFIG

    @staticmethod
    def is_cp_initialized():
        """中文说明：`is_cp_initialized` 实现 VideoVAE 上下文并行（context parallel）通信工具中的 `is_cp_initialized` 步骤，供训练、推理或调试流程复用。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if ContextParallelUtils._CONTEXT_PARALLEL_GROUP is None:
            return False
        else:
            return True

    @staticmethod
    def initialize_context_parallel(cp_config:dict):
        """中文说明：`initialize_context_parallel` 实现 VideoVAE 上下文并行（context parallel）通信工具中的 `initialize_context_parallel` 步骤，供训练、推理或调试流程复用。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        assert ContextParallelUtils._CONTEXT_PARALLEL_GROUP is None, "context parallel group 已经初始化，请勿重复初始化"

        context_parallel_size = cp_config["cp_size"]
        if context_parallel_size > 1:
            ContextParallelUtils.CP_CONFIG = cp_config
        else:
            print(f"警告：context parallel size 必须 > 1，当前 context_parallel_size={context_parallel_size}")
            return

        ContextParallelUtils._CONTEXT_PARALLEL_SIZE = context_parallel_size

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        for i in range(0, world_size, context_parallel_size):
            ranks = range(i, i + context_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                ContextParallelUtils._CONTEXT_PARALLEL_GROUP = group
                break

    @staticmethod
    def get_cp_group():
        """中文说明：`get_cp_group` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return ContextParallelUtils._CONTEXT_PARALLEL_GROUP

    @staticmethod
    def get_cp_size():
        """中文说明：`get_cp_size` 实现 VideoVAE 上下文并行（context parallel）通信工具中的 `get_cp_size` 步骤，供训练、推理或调试流程复用。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return ContextParallelUtils._CONTEXT_PARALLEL_SIZE

    @staticmethod
    def get_cp_world_size():
        """中文说明：`get_cp_world_size` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if ContextParallelUtils.is_cp_initialized():
            world_size = torch.distributed.get_world_size()
            return world_size // ContextParallelUtils._CONTEXT_PARALLEL_SIZE
        else:
            return 0

    @staticmethod
    def get_cp_rank():
        """中文说明：`get_cp_rank` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if ContextParallelUtils.is_cp_initialized():
            global_rank = torch.distributed.get_rank()
            cp_rank = global_rank % ContextParallelUtils._CONTEXT_PARALLEL_SIZE
            return cp_rank
        else:
            return 0

    def get_cp_group_rank():
        """中文说明：`get_cp_group_rank` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        if ContextParallelUtils.is_cp_initialized():
            rank = torch.distributed.get_rank()
            cp_group_rank = rank // ContextParallelUtils._CONTEXT_PARALLEL_SIZE
            return cp_group_rank
        else:
            return 0


def _gather_tensor_shape(local_ts):
    """中文说明：`_gather_tensor_shape` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    cp_size = ContextParallelUtils.get_cp_size()
    local_shape = torch.tensor(local_ts.shape, dtype=torch.int64, device=local_ts.device)
    gathered_shapes = [torch.zeros(len(local_shape), dtype=torch.int64, device=local_ts.device) for _ in range(cp_size)]
    dist.all_gather(gathered_shapes, local_shape, group=ContextParallelUtils._CONTEXT_PARALLEL_GROUP)
    return [shape.tolist() for shape in gathered_shapes]

@torch.compiler.disable()
def dist_encoder_gather_result(res)->list:
    """中文说明：`dist_encoder_gather_result` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    cp_size = ContextParallelUtils.get_cp_size()
    if cp_size < 2:
        return res

    shape_list = _gather_tensor_shape(res) # 代码/形状说明：[[1,2,3,4],[x,x,x,x]] list of shapes on different rank
    encs=[torch.zeros(s, device=res.device, dtype=res.dtype) for s in shape_list]

    dist.barrier()
    encs = distops.all_gather(encs, res, group=ContextParallelUtils._CONTEXT_PARALLEL_GROUP)
    return encs

@torch.compiler.disable()
def dist_decoder_gather_result(res)->list:
    """中文说明：`dist_decoder_gather_result` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    cp_size = ContextParallelUtils.get_cp_size()
    if cp_size < 2:
        return res

    shape_list = _gather_tensor_shape(res) # 代码/形状说明：[[1,2,3,4],[x,x,x,x]] list of shapes on different rank
    decs = [torch.zeros(s, device=res.device, dtype=res.dtype) for s in shape_list]

    dist.barrier()
    decs = distops.all_gather(decs, res, group=ContextParallelUtils._CONTEXT_PARALLEL_GROUP)
    return decs


def _send_with_shape(local_ts, next_rank):
    """中文说明：`_send_with_shape` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    local_shape = torch.tensor(local_ts.shape, dtype=torch.int64, device=local_ts.device)
    torch.distributed.send(local_shape.contiguous(), next_rank)
    torch.distributed.send(local_ts.contiguous(), next_rank)

def _recv_with_shape(pre_rank):
    """中文说明：`_recv_with_shape` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    shape = torch.zeros(5, dtype=torch.int64, device=device)
    torch.distributed.recv(shape, pre_rank)
    ts = torch.zeros(shape.tolist(), device=device)
    torch.distributed.recv(ts, pre_rank)
    return ts


@torch.compiler.disable()
def dist_conv_cache_send(conv_cache):

    """中文说明：`dist_conv_cache_send` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    cp_rank = ContextParallelUtils.get_cp_rank()
    global_rank = torch.distributed.get_rank()
    cp_size = ContextParallelUtils.get_cp_size()

    if cp_rank == cp_size - 1:
        return
    if conv_cache is None:
        return

    next_rank = global_rank + 1
    _send_with_shape(conv_cache, next_rank)

@torch.compiler.disable()
def dist_conv_cache_recv():
    """中文说明：`dist_conv_cache_recv` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点跟踪 rank、world size、shape gather/send/recv，确认张量在时间维或上下文维如何拆分再合并。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    cp_rank = ContextParallelUtils.get_cp_rank()
    global_rank = torch.distributed.get_rank()

    if cp_rank == 0:
        return None

    pre_rank = global_rank - 1
    return _recv_with_shape(pre_rank)
