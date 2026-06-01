# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""损失函数表。

中文导读：
    本模块仅提供一个名字 -> 损失类的小字典 ``_LOSSES``，方便 SlowFast 配置里
    通过字符串 ``cfg.MODEL.LOSS_FUNC`` 选 ``cross_entropy / bce / bce_logit``。

    WorldVLN 动作解码器实际训练 6D delta 回归，使用 ``MSE/L1`` 损失，并不在本文件
    定义；保留此文件是为了与上游 SlowFast/X3D head 兼容（它们的分类输出仍走
    ``CrossEntropyLoss``）。
"""

import torch.nn as nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(loss_name):
    """
    根据损失函数名称返回对应实现。
    参数 (int)：
        loss_name: 要使用的损失函数名称。
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("不支持损失函数 {}".format(loss_name))
    return _LOSSES[loss_name]
