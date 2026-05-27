# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
数据集注册表。

注册后的对象会以 `obj(cfg, split)` 的形式被调用。
调用结果应返回一个 `torch.utils.data.Dataset` 对象。
"""


def build_dataset(dataset_name, cfg, split):
    """
        根据 `dataset_name` 构建数据集。
        参数：
            dataset_name (str): 要构建的数据集名称。
            cfg (CfgNode): 配置。详见
                说明：slowfast/config/defaults.py
            split (str): 数据加载器的切分。可选 `train`、
                说明：`val` 和 `test`。
        返回：
            Dataset: 按 `dataset_name` 构建出的数据集对象。

    """
    # 将 dataset_name 首字母大写：配置里可能是小写，但数据集类名始终以大写开头。
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)
