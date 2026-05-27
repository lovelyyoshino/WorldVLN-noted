# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_trainer(args):
    """返回默认训练器实现，这个分支始终使用 `InfinityTrainer`。"""
    from infinity.trainer.sft_trainer import InfinityTrainer as Trainer
    return Trainer
