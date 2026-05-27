# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multigrid 训练的辅助工具。"""

import numpy as np
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler


class ShortCycleBatchSampler(Sampler):
    """
        扩展采样器，使它支持短周期采样。

        细节可参考论文 "A Multigrid Method for Efficiently Training Video Models",
        引用/来源：Wu et al., 2019 (https://arxiv.org/abs/1912.00998)。

    """

    def __init__(self, sampler, batch_size, drop_last, cfg):
        """根据配置计算三种批大小，并保存底层采样器。"""
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler 应是 torch.utils.data.Sampler 的实例，"
                "但实际收到 sampler={}".format(sampler)
            )
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size 应是正整数，"
                "但实际收到 batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last 应是布尔值，但实际收到 "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.drop_last = drop_last

        bs_factor = [
            int(
                round(
                    (
                        float(cfg.DATA.TRAIN_CROP_SIZE)
                        / (s * cfg.MULTIGRID.DEFAULT_S)
                    )
                    ** 2
                )
            )
            for s in cfg.MULTIGRID.SHORT_CYCLE_FACTORS
        ]

        self.batch_sizes = [
            batch_size * bs_factor[0],
            batch_size * bs_factor[1],
            batch_size,
        ]

    def __iter__(self):
        """按短周期顺序产出带周期标记的批数据。"""
        counter = 0
        batch_size = self.batch_sizes[0]
        batch = []
        for idx in self.sampler:
            batch.append((idx, counter % 3))
            if len(batch) == batch_size:
                yield batch
                counter += 1
                batch_size = self.batch_sizes[counter % 3]
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        """返回按平均批大小估算后的批数量。"""
        avg_batch_size = sum(self.batch_sizes) / 3.0
        if self.drop_last:
            return int(np.floor(len(self.sampler) / avg_batch_size))
        else:
            return int(np.ceil(len(self.sampler) / avg_batch_size))
