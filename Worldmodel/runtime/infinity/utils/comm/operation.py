# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup


class AllToAll(torch.autograd.Function):
    """中文说明：通过 all_to_all_single 将输入张量 [e, c, h] 分发给所有专家（expert）。
    对 `torch.distributed` 通信操作的封装。
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if dist.get_world_size(group) == 1:
            return inputs, None
        output = torch.empty_like(inputs)
        if not overlap:
            dist.all_to_all_single(output, inputs, group=group)
            return output, None
        else:
            handle = dist.all_to_all_single(output, inputs, group=group, async_op=True)
            return output, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return (
            AllToAll.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class AsyncAllGatherForTwo(torch.autograd.Function):
    """中文说明：`AsyncAllGatherForTwo` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor,
        sp_rank: int,
        sp_size: int,
        group: Optional[ProcessGroup] = None,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        from torch.distributed._functional_collectives import all_gather_tensor

        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size

        # 聚合 all_gather 输入
        all_inputs = all_gather_tensor(inputs.unsqueeze(0), 0, group)
        # 计算本地 qkv
        local_qkv = F.linear(inputs, weight, bias).unsqueeze(0)

        # 远端计算
        remote_inputs = all_inputs[1 - sp_rank].view(list(local_qkv.shape[:-1]) + [-1])
        # 计算远端 qkv
        remote_qkv = F.linear(remote_inputs, weight, bias)

        # 拼接本地与远端 qkv
        if sp_rank == 0:
            qkv = torch.cat([local_qkv, remote_qkv], dim=0)
        else:
            qkv = torch.cat([remote_qkv, local_qkv], dim=0)
        qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        ctx.save_for_backward(inputs, weight, remote_inputs)
        return qkv

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        from torch.distributed._functional_collectives import reduce_scatter_tensor

        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        inputs, weight, remote_inputs = ctx.saved_tensors

        # 拆分 qkv 梯度
        qkv_grad = grad_outputs[0]
        qkv_grad = rearrange(qkv_grad, "b (sp n) c -> sp b n c", sp=sp_size)
        qkv_grad = torch.chunk(qkv_grad, 2, dim=0)
        if sp_rank == 0:
            local_qkv_grad, remote_qkv_grad = qkv_grad
        else:
            remote_qkv_grad, local_qkv_grad = qkv_grad

        # 计算远端梯度
        remote_inputs_grad = torch.matmul(remote_qkv_grad, weight).squeeze(0)
        weight_grad = torch.matmul(remote_qkv_grad.transpose(-1, -2), remote_inputs).squeeze(0).sum(0)
        bias_grad = remote_qkv_grad.squeeze(0).sum(0).sum(0)

        # 启动异步 reduce_scatter
        remote_inputs_grad_zero = torch.zeros_like(remote_inputs_grad)
        if sp_rank == 0:
            remote_inputs_grad = torch.cat([remote_inputs_grad_zero, remote_inputs_grad], dim=0)
        else:
            remote_inputs_grad = torch.cat([remote_inputs_grad, remote_inputs_grad_zero], dim=0)
        remote_inputs_grad = reduce_scatter_tensor(remote_inputs_grad, "sum", 0, group)

        # 计算本地梯度并等待 reduce_scatter
        local_input_grad = torch.matmul(local_qkv_grad, weight).squeeze(0)
        weight_grad += torch.matmul(local_qkv_grad.transpose(-1, -2), inputs).squeeze(0).sum(0)
        bias_grad += local_qkv_grad.squeeze(0).sum(0).sum(0)

        # 合并远端与本地梯度
        inputs_grad = remote_inputs_grad + local_input_grad
        return inputs_grad, weight_grad, bias_grad, None, None, None


class AllGather(torch.autograd.Function):
    """中文说明：`AllGather` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: Optional[ProcessGroup] = None,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        # 调试输出：print(f"XW 调试：All Gather 的通信组大小为 {comm_size}")
        if comm_size == 1:
            return inputs.unsqueeze(0), None

        buffer_shape = (comm_size,) + inputs.shape
        outputs = torch.empty(buffer_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(outputs, comm_size, dim=0))
        # 调试代码：buffer_list = list([
        # 中文说明：等价写法为 [t.squeeze(0) for t in torch.chunk(outputs, comm_size, dim=0)]
        # ])

        if not overlap:
            # 调试输出：print("buffer_list 长度", len(buffer_list), [t.shape for t in buffer_list])
            # 调试输出：print("inputs 张量", inputs.shape, inputs.is_contiguous())
            # 调试输出：print(group)

            dist.all_gather(buffer_list, inputs, group=group)
            return outputs, None
        else:
            handle = dist.all_gather(buffer_list, inputs, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return (
            ReduceScatter.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class ReduceScatter(torch.autograd.Function):
    """中文说明：`ReduceScatter` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        返回：
            outputs：输出张量。
            handle：当 overlap=True 时返回异步 Work 句柄。
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.squeeze(0), None

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        output_shape = inputs.shape[1:]
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(inputs, comm_size, dim=0))
        if not overlap:
            dist.reduce_scatter(outputs, buffer_list, group=group)
            return outputs, None
        else:
            handle = dist.reduce_scatter(outputs, buffer_list, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        # 待办：支持异步 backward
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return (
            AllGather.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


# 使用 all_to_all_single API 执行 all-to-all 通信
def _all_to_all_single(input_, seq_world_size, group, scatter_dim, gather_dim):
    """中文说明：`_all_to_all_single` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    if scatter_dim < 2:
        input_t = input_.reshape([seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]).contiguous()
    else:
        input_t = (
            input_.reshape([-1, seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :])
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    if scatter_dim < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_dim]
        + [
            inp_shape[gather_dim] * seq_world_size,
        ]
        + inp_shape[gather_dim + 1 :]
    ).contiguous()


# 使用 all_to_all API 执行 all-to-all 通信
def _all_to_all(input_, world_size, group, scatter_dim, gather_dim):
    """中文说明：`_all_to_all` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """中文说明：执行 all-to-all 通信。

    参数：
        input_：输入矩阵。
        process_group：通信进程组。
        scatter_dim：执行 scatter 的维度。
        gather_dim：执行 gather 的维度。
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)
        bsz, _, _ = input_.shape

        # 待办：尝试让 all_to_all_single 兼容更大的批大小（batch size）
        if bsz == 1:
            return _all_to_all_single(input_, world_size, process_group, scatter_dim, gather_dim)
        else:
            return _all_to_all(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


def model_sharding(model: torch.nn.Module):
    """中文说明：`model_sharding` 实现Infinity 序列并行通信算子中的 `model_sharding` 步骤，供训练、推理或调试流程复用。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    """中文说明：`all_to_all_comm` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


def _gather(input_, dim=-1, process_group=None):
    # 只有一个 rank 参与时跳过通信
    """中文说明：`_gather` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # 聚合 all_gather 结果
    input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=process_group)

    # 拼接
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _split(input_, dim=-1, process_group=None):
    # 只有一个 rank 参与时跳过通信
    """中文说明：`_split` 实现Infinity 序列并行通信算子中的 `_split` 步骤，供训练、推理或调试流程复用。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # 沿最后一维拆分。
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"要拆分的维度大小 ({dim_size}) 必须是 world size ({world_size}) 的整数倍，"
        f"否则无法把 tensor 均匀切分"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = dist.get_rank(process_group)
    output = tensor_list[rank].clone().contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """中文说明：从模型并行区域 gather 输入并拼接。

        参数：
            input_：输入矩阵。.
            说明：parallel_mode：并行模式。
            dim：通信作用的维度。

    """

    @staticmethod
    def forward(ctx, input_, dim, process_group):
        """中文说明：`forward` 执行Infinity 序列并行通信算子的前向计算；重点核对输入输出张量 shape 是否与调用方约定一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        ctx.process_group = process_group
        ctx.dim = dim
        return _gather(input_, dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        """中文说明：`backward` 实现Infinity 序列并行通信算子的反向传播；返回梯度顺序必须和 forward 输入顺序一致。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return _split(grad_output, ctx.dim, ctx.process_group), None, None


def gather_forward_split_backward(input_, dim, process_group):
    """中文说明：`gather_forward_split_backward` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    return _GatherForwardSplitBackward.apply(input_, dim, process_group)
