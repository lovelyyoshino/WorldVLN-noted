
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_lambda(args):
    """中文说明：`get_lambda` 计算学习率调度倍率；最终学习率通常是 base_lr 乘以当前 step 的 lambda。

    新手提示：重点公式通常是 lr = base_lr * lambda(step)，lambda 由 warmup/衰减阶段决定。
    关键公式：lr(step) = base_lr * lambda(step)。
    """
    if args.scheduler == "linear":
        def lr_lambda(step):
            """中文说明：`lr_lambda` 计算学习率调度倍率；最终学习率通常是 base_lr 乘以当前 step 的 lambda。

            新手提示：重点公式通常是 lr = base_lr * lambda(step)，lambda 由 warmup/衰减阶段决定。
            关键公式：lr(step) = base_lr * lambda(step)。
            """
            warmup_steps = args.warmup_steps
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.
        return lr_lambda
    else:
        raise NotImplementedError
