# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

# 来源：colossalai 和 opendit
#
import itertools
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
from torch.distributed import ProcessGroup


def prod(nums: List[int]) -> int:
    """中文说明：计算数字列表的乘积。

    参数：
        nums (List[int])：数字列表。

    返回：
        int：这些数字的乘积。
    """
    return reduce(mul, nums)


class ProcessGroupMesh:
    """中文说明：管理 process group mesh 的辅助类；只描述进程组如何组织，与具体并行方法解耦。
        它只负责初始化并缓存 process group；具体并行方法负责使用这些 group 完成并行计算。

        这里用 N 维 tuple 表示 process group mesh，并用 N 维坐标表示每个进程。
        例如在形状为 `(2,2,2)` 的 3D mesh 中，坐标 `(0,1,0)` 表示 rank 为 2 的进程。

        参数：
            *size (int)：process group mesh 每个维度的大小；所有维度乘积必须等于 world size。

        属性：
            公式/形状说明：shape (Tuple[int, ...])：process group mesh 的形状。
            公式/形状说明：rank (int)：当前进程的 rank。

    """

    def __init__(self, *size: int) -> None:
        """中文说明：`__init__` 初始化Infinity 序列并行通信算子需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        assert dist.is_initialized(), "请先初始化 torch.distributed。"
        assert prod(size) == dist.get_world_size(), f"`size` 各维度的乘积必须等于 world size；当前分别是 {prod(size)} 和 {dist.get_world_size()}。"
        self._shape = size
        self._rank = dist.get_rank()
        self._coord = ProcessGroupMesh.unravel(self._rank, self._shape)
        self._ranks_to_group: Dict[Tuple[int, ...], ProcessGroup] = {}
        self._group_to_ranks: Dict[ProcessGroup, Tuple[int, ...]] = {}

    @property
    def shape(self) -> Tuple[int, ...]:
        """中文说明：`shape` 实现Infinity 序列并行通信算子中的 `shape` 步骤，供训练、推理或调试流程复用。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return self._shape

    @property
    def rank(self) -> int:
        """中文说明：`rank` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return self._rank

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """中文说明：获取 process group mesh 的尺寸。

        参数：
            dim (Optional[int]，可选)：process group mesh 的维度；`None` 表示所有维度，默认 None。

        返回：
            Union[int, Tuple[int, ...]]: 目标维度或整个 process group mesh 的尺寸。
        """
        if dim is None:
            return self._shape
        else:
            return self._shape[dim]

    def coordinate(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """中文说明：获取当前 rank 在 process group mesh 中的坐标。

        参数：
            dim (Optional[int]，可选)：process group mesh 的维度；`None` 表示所有维度，默认 None。

        返回：
            Union[int, Tuple[int, ...]]: 目标维度或整个 process group mesh 的坐标。
        """
        if dim is None:
            return self._coord
        else:
            return self._coord[dim]

    @staticmethod
    def unravel(rank: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """中文说明：把 rank 转换为 mesh 坐标。

        参数：
            rank (int): 待转换的 rank。
            shape (Tuple[int, ...]): process group mesh 的形状。

        返回：
            Tuple[int, ...]: 该 rank 对应的坐标。
        """
        res = np.unravel_index(rank, shape)
        return tuple(int(i) for i in res)

    @staticmethod
    def ravel(coord: Tuple[int, ...], shape: Tuple[int, ...], mode: str = "raise") -> int:
        """中文说明：把 mesh 坐标转换为 rank。
           mode 可选 `raise/wrap/clip`，含义见 https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html。
           使用 `wrap` 时，越界索引会按周期回绕。
           例如 `ravel((0, i, 0), (1, 2, 1), 'wrap')` 返回 `i % 2`。

        参数：
            coords (Tuple[int, ...]): 待转换的坐标。
            shape (Tuple[int, ...]): process group mesh 的形状。
            mode (Optional[str]): 传给 `numpy.ravel_multi_index` 的越界处理模式。

        返回：
            int: 该坐标对应的 rank。
        """

        assert mode in ["raise", "wrap", "clip"]
        return int(np.ravel_multi_index(coord, shape, mode))

    def get_group(self, ranks_in_group: List[int], backend: Optional[str] = None) -> ProcessGroup:
        """中文说明：获取指定 ranks 对应的 process group；如果不存在则创建。

        参数：
            ranks_in_group (List[int]): process group 中的 rank 列表。
            backend (Optional[str]，可选)：process group 后端，默认 None。

        返回：
            ProcessGroup: 由给定 ranks 组成的 process group。
        """
        ranks_in_group = sorted(ranks_in_group)
        if tuple(ranks_in_group) not in self._group_to_ranks:
            group = dist.new_group(ranks_in_group, backend=backend)
            self._ranks_to_group[tuple(ranks_in_group)] = group
            self._group_to_ranks[group] = tuple(ranks_in_group)
        return self._ranks_to_group[tuple(ranks_in_group)]

    def get_ranks_in_group(self, group: ProcessGroup) -> List[int]:
        """中文说明：获取给定 process group 中的 ranks；该 group 必须由本类创建。

        参数：
            group (ProcessGroup): process group 对象。

        返回：
            List[int]: process group 中的 rank 列表。
        """
        return list(self._group_to_ranks[group])

    @staticmethod
    def get_coords_along_axis(
        base_coord: Tuple[int, ...], axis: int, indices_at_axis: List[int]
    ) -> List[Tuple[int, ...]]:
        """中文说明：获取指定轴方向上的坐标列表。

        参数：
            base_coord (Tuple[int, ...]): 沿指定轴生成坐标时使用的基准坐标。
            axis (int): 生成坐标所在的轴。
            indices_at_axis (List[int]): 该轴上的索引列表。

        返回：
            List[Tuple[int, ...]]: 沿该轴生成的坐标列表。
        """
        coords_in_group = []
        for idx in indices_at_axis:
            coords_in_group.append(base_coord[:axis] + (idx,) + base_coord[axis + 1 :])
        return coords_in_group

    def create_group_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        """中文说明：沿指定轴创建所有 process group，并返回当前进程所在的 group。

        参数：
            axis (int): 创建 process group 所沿的轴。
            indices_at_axis (Optional[List[int]]，可选)：该轴上的索引列表，默认 None。
            backend (Optional[str]，可选)：process group 后端，默认 None。

        返回：
            ProcessGroup: 当前进程在给定轴上所属的 process group。
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        reduced_shape = list(self._shape)
        # 该轴上的可选项缩减为 1，因为它已经由 `indices_at_axis` 决定
        reduced_shape[axis] = 1
        target_group = None
        # 使用笛卡尔积生成坐标组合
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(base_coord, axis, indices_at_axis)
            ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
            group = self.get_group(ranks_in_group, backend=backend)
            if self._rank in ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        """中文说明：获取当前进程在指定轴上所属的 process group；如果不存在则创建。

        参数：
            axis (int): 创建 process group 所沿的轴。
            indices_at_axis (Optional[List[int]]，可选)：该轴上的索引列表，默认 None。
            backend (Optional[str]，可选)：process group 后端，默认 None。

        返回：
            ProcessGroup: 当前进程在给定轴上所属的 process group。
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        coords_in_group = ProcessGroupMesh.get_coords_along_axis(self._coord, axis, indices_at_axis)
        ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
        if ranks_in_group not in self._ranks_to_group:
            # 不需要显式缓存，因为 `create_group_along_axis` 会缓存它
            return self.create_group_along_axis(axis, indices_at_axis, backend=backend)
        return self._ranks_to_group[ranks_in_group]

from torch.distributed import ProcessGroup


class ProcessGroupManager(ProcessGroupMesh):
    """中文说明：`ProcessGroupManager` 封装Infinity 序列并行通信算子中的状态和子模块。

    新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
    关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
    """
    def __init__(self, *size: int, dp_axis, sp_axis):
        """中文说明：`__init__` 初始化Infinity 序列并行通信算子需要的配置、缓存或子模块，不直接执行训练/推理主循环。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        super().__init__(*size)
        self.dp_axis = dp_axis
        self.sp_axis = sp_axis
        self._dp_group: ProcessGroup = self.get_group_along_axis(self.dp_axis)
        self._sp_group: ProcessGroup = self.get_group_along_axis(self.sp_axis)

    @property
    def dp_group(self) -> ProcessGroup:
        """中文说明：`dp_group` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return self._dp_group

    @property
    def sp_group(self) -> ProcessGroup:
        """中文说明：`sp_group` 读取或构造分布式 rank/group/world-size 信息，是理解多卡行为的基础。

        新手提示：重点公式是 split/gather 在 sequence 维互逆，forward 的 all-to-all 与 backward 的 reduce/scatter 必须配对。
        关键关系：forward 负责数据分发或聚合，backward 通常执行互逆通信来传回梯度。
        """
        return self._sp_group
