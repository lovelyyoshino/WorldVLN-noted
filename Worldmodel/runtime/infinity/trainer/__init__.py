# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_trainer(args):
    """根据配置选择训练器实现，在 SFT 与 GRPO 之间切换。"""
    trainer_type = str(getattr(args, "trainer_type", "sft") or "sft").strip().lower()
    if trainer_type == "grpo":
        from infinity.trainer.GRPO_trainer import GRPOTrainer as Trainer
    else:
        from infinity.trainer.sft_trainer import InfinityTrainer as Trainer
    return Trainer
