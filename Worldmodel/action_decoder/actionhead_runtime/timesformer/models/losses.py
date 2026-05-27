# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""损失函数注册和查询工具。"""

import torch.nn as nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(loss_name):
    """
        根据损失函数名称返回对应的 PyTorch loss 类。

        参数 (int):
            loss_name: 要使用的损失函数名称。

    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("不支持损失函数 {}".format(loss_name))
    return _LOSSES[loss_name]
