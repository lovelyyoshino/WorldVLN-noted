# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""优化器构建和学习率设置工具。"""

import torch

import timesformer.utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
        构建带 momentum 的随机梯度下降优化器，或 ADAM/AdamW 优化器。

        相关算法可参考：
        说明：Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
        以及
        说明：Diederik P.Kingma, and Jimmy Ba.
        说明："Adam: A Method for Stochastic Optimization."

        参数：
            model (model): 需要由 SGD、ADAM 或 AdamW 优化的模型。
            cfg (config): SGD/ADAM/AdamW 的超参数配置，包括 base learning rate、
            momentum、weight_decay、dampening 等。

    """
    # BatchNorm 参数。
    bn_params = []
    # 非 BatchNorm 参数。
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # 对 BatchNorm 和非 BatchNorm 参数使用不同的 weight decay。
    # Caffe2 分类代码里 BatchNorm 的 weight decay 是 0.0。
    # 给 BatchNorm 设置不同的 weight decay 可能导致性能下降。
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ]
    # 确认所有参数都会被传给 optimizer。
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
        按 lr policy 查询指定 epoch 的学习率。

        参数：
            cfg (config): 优化器和 lr policy 的超参数配置。
            cur_epoch (float): 当前训练阶段的 epoch 编号。

    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
        把 optimizer 的 lr 设置为指定值。

        参数：
            optimizer (optim): 当前网络使用的优化器。
            new_lr (float): 要设置的新学习率。

    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
