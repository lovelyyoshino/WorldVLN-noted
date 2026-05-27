# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
数据集注册表。

注册对象会以 `obj(cfg, split)` 的形式被调用。
调用结果应返回一个 `torch.utils.data.Dataset` 对象。
"""


def build_dataset(dataset_name, cfg, split):
    """
        按 `dataset_name` 构建数据集对象。

        参数：
            dataset_name (str): 要构建的数据集名称。
            cfg (CfgNode): 配置，字段含义可参考 slowfast/config/defaults.py。
            split (str): 数据加载器使用的数据划分，取值包括 `train`、`val` 和 `test`。

        返回：
            Dataset: 由 dataset_name 指定并构建好的数据集。

    """
    # 配置里的 dataset_name 可能是小写，但数据集类名约定首字母大写。
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)
