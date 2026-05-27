#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO 训练器。

这里只是复用 `InfinityTrainer` 的主体训练流程，并通过
`args.trainer_type=grpo` 打开奖励加权目标。
"""

from infinity.trainer.sft_trainer import InfinityTrainer


class GRPOTrainer(InfinityTrainer):
    """GRPO 训练入口，占位类，主要行为由父类中的 GRPO 分支实现。"""
    pass
