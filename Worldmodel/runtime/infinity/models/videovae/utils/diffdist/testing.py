# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import infinity.models.videovae.utils.diffdist.functional as distops
import torch.distributed as dist
import torch
import infinity.models.videovae.utils.diffdist.extra_collectives as extra_comm


def test_reduce_scatter():
    """中文说明：`test_reduce_scatter` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if dist.get_rank() == 0:
        print("reduce_scatter 测试\n")
    x = torch.arange(dist.get_world_size()).float().split(1)
    buff = torch.tensor(0.)
    extra_comm.reduce_scatter(buff, x)
    print(dist.get_rank(), x)
    print(dist.get_rank(), buff)
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)


def test_all_gather():
    """中文说明：`test_all_gather` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if dist.get_rank() == 0:
        print("all_gather 测试\n")
    dist.barrier()
    x = torch.tensor(3., requires_grad=True)
    y = (dist.get_rank() + 1) * x

    print(dist.get_rank(), "发送 y:", y)
    z = distops.all_gather(list(torch.zeros(dist.get_world_size())),
                           y,
                           next_backprop=None,
                           inplace=True)
    print(dist.get_rank(), "收到 tensor:", z)
    l = torch.sum(torch.stack(z))
    l = l * (dist.get_rank() + 1)
    l.backward()

    print(dist.get_rank(), "MPI 梯度:", x.grad)
    dist.barrier()
    if dist.get_rank() == 0:
        print()
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        res = []
        for i in range(1, dist.get_world_size() + 1):
            res.append(i * x[i - 1])

        res2 = []
        for i in range(dist.get_world_size()):
            temp = []
            for j in range(dist.get_world_size()):
                temp.append(torch.clone(res[j]))
            res2.append(temp)
        l_s = [torch.sum(torch.stack(i)) for i in res2]
        final = [(i + 1) * k for i, k in enumerate(l_s)]
        for i in range(dist.get_world_size() - 1):
            final[i].backward(retain_graph=True)
        final[-1].backward()
        for i, x_i in enumerate(x):
            print(i, "单进程梯度:", x_i.grad)
        print('-' * 50)


def test_scatter():
    """中文说明：`test_scatter` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if dist.get_rank() == 0:
        print("scatter 测试\n")
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        y = [2 * x_i for x_i in x]

        print("发送 y:", y)
        buffer = torch.tensor(0.)
        z = distops.scatter(buffer, y, src=0, inplace=False)
    else:
        buffer = torch.tensor(0., requires_grad=True)
        z = distops.scatter(buffer, src=0, inplace=False)

    print(dist.get_rank(), "收到 tensor:", z)
    # 计算过程
    k = (dist.get_rank() + 1) * z
    k.backward()

    if dist.get_rank() == 0:
        print("MPI 梯度:", [x_i.grad for x_i in x])

    if dist.get_rank() == 0:
        print()
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        y = [2 * x_i for x_i in x]
        res = []
        for i in range(dist.get_world_size()):
            res.append((i + 1) * y[i])

        for i, k in enumerate(res):
            k.backward()
        print("单进程梯度:", [x_i.grad for x_i in x])
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)


def test_gather():
    """中文说明：`test_gather` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if dist.get_rank() == 0:
        print("gather 测试\n")
    dist.barrier()
    x = torch.tensor(3., requires_grad=True)
    y = (dist.get_rank() + 1) * x

    print(dist.get_rank(), "发送 y:", y)
    if dist.get_rank() == 0:
        z = distops.gather(y,
                           torch.zeros(dist.get_world_size()).split(1),
                           dst=0,
                           next_backprop=None,
                           inplace=True)
        print(dist.get_rank(), "收到 tensor:", z)
        l = torch.sum(torch.stack(z))
        l.backward()
    else:
        dummy = distops.gather(y, dst=0, next_backprop=None, inplace=True)
        dummy.backward(torch.tensor([]))
    print(dist.get_rank(), "MPI 梯度:", x.grad)
    dist.barrier()
    if dist.get_rank() == 0:
        print()
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        res = []
        for i in range(1, dist.get_world_size() + 1):
            res.append(i * x[i - 1])

        z = torch.stack(res)
        l = torch.sum(z)
        l.backward()
        for i, x_i in enumerate(x):
            print(i, "单进程梯度:", x_i.grad)
        print('-' * 50)


def test_broadcast():
    """中文说明：`test_broadcast` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if dist.get_rank() == 0:
        print("broadcast 测试\n")
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x

        print(dist.get_rank(), "发送 y:", y)
        z = distops.broadcast(y, src=0, inplace=False)
        print(dist.get_rank(), "收到 tensor:", z)

        # 计算过程
        k = 3 * z
        k.backward()
        print("MPI 梯度:", x.grad)

        print()
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x
        res = [3 * y]
        for i in range(1, dist.get_world_size()):
            res.append(9 * y)

        for i, k in enumerate(res):
            if i == (len(res) - 1):
                k.backward()
            else:
                k.backward(retain_graph=True)
        print("单进程梯度:", x.grad)
    else:
        x = torch.tensor(5., requires_grad=True)
        y = 7 * x

        buffer = torch.tensor(0.)
        z = distops.broadcast(buffer, src=0, next_backprop=y)
        print(dist.get_rank(), "收到 tensor:", z)
        k = 9 * z
        k.backward()
        print(dist.get_rank(), "断开部分的梯度:", x.grad)
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)


def test_consume_variable():
    """中文说明：`test_consume_variable` 实现可求导分布式通信封装中的 `test_consume_variable` 步骤，供训练、推理或调试流程复用。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    x = torch.tensor(5., requires_grad=True)
    y = 2 * x

    z = 3 * y
    j = 4 * y

    z = distops.consume_variable(j, [z], set_ones_grad=True)[0]
    print(z)
    z.backward()
    print(x.grad)
    print()
    x = torch.tensor(5., requires_grad=True)
    y = 2 * x

    z = 3 * y
    j = 4 * y

    z.backward(retain_graph=True)
    j.backward()
    print(x.grad)


def test_send_recv():
    """中文说明：`test_send_recv` 执行分布式通信包装；阅读时把张量切分维度、rank 方向和 backward 的反向通信配对检查。

    新手提示：这些包装把 send/recv/gather/scatter 接到 autograd，阅读时要把 forward 的通信和 backward 的梯度回传成对看。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    if dist.get_rank() == 0:
        print("SEND/RECV 测试\n")
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x

        print("发送 y 之前:", y)
        connector = distops.send(y, dst=1)
        # 计算发生在 1 号进程
        buffer = torch.tensor(0.)
        z, _ = distops.recv(buffer, src=1, next_backprop=connector)
        print("接收之后:", z)

        k = 3 * z
        k.backward()
        print("MPI 梯度:", x.grad)

        print()
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x
        l = y * 10
        k = 3 * l
        k.backward()
        print("单进程梯度:", x.grad)
        print('-' * 50)
    elif dist.get_rank() == 1:
        buffer = torch.tensor(0., requires_grad=True)
        y, _ = distops.recv(buffer, src=0)

        l = y * 10

        connector = distops.send(l, dst=0)
        connector.backward(torch.tensor([]))


if __name__ == '__main__':
    dist.init_process_group('mpi')

    print(f'当前 rank 是 {dist.get_rank()}')
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)

    if dist.get_rank() == 0:
        print("额外 collectives")

    test_reduce_scatter()

    if dist.get_rank() == 0:
        print('-' * 50)

    test_send_recv()

    test_broadcast()

    test_gather()

    test_scatter()

    test_all_gather()
