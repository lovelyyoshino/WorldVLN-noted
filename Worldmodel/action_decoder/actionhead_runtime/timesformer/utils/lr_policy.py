# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""学习率策略。"""

import math


def get_lr_at_epoch(cfg, cur_epoch):
    """
    获取当前 epoch 的学习率，并在训练初期按需执行预热。

    参数：
        cfg (CfgNode): 配置对象，细节见 `slowfast/config/defaults.py`。
        cur_epoch (float): 当前训练阶段的 epoch 编号。
    """
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
    # 执行预热。
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
        lr_start = cfg.SOLVER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
            cfg, cfg.SOLVER.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
        按余弦学习率调度计算指定 epoch 的学习率。

        具体形式可参考：
        说明：Ilya Loshchilov, and  Frank Hutter
        说明：SGDR: Stochastic Gradient Descent With Warm Restarts.

        参数：
            cfg (CfgNode): 配置对象，细节见 `slowfast/config/defaults.py`。
            cur_epoch (float): 当前训练阶段的 epoch 编号。

    """
    assert cfg.SOLVER.COSINE_END_LR < cfg.SOLVER.BASE_LR
    return (
        cfg.SOLVER.COSINE_END_LR
        + (cfg.SOLVER.BASE_LR - cfg.SOLVER.COSINE_END_LR)
        * (math.cos(math.pi * cur_epoch / cfg.SOLVER.MAX_EPOCH) + 1.0)
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    按分段相对学习率策略计算指定 epoch 的学习率。

    参数：
        cfg (CfgNode): 配置对象，细节见 `slowfast/config/defaults.py`。
        cur_epoch (float): 当前训练阶段的 epoch 编号。
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind] * cfg.SOLVER.BASE_LR


def get_step_index(cfg, cur_epoch):
    """
    获取给定 epoch 对应的学习率 step 索引。

    参数：
        cfg (CfgNode): 配置对象，细节见 `slowfast/config/defaults.py`。
        cur_epoch (float): 当前训练阶段的 epoch 编号。
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    根据策略名返回对应的学习率函数。

    参数：
        lr_policy (string): 当前任务使用的学习率策略名称。
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("未知学习率策略：{}".format(lr_policy))
    else:
        return globals()[policy]
