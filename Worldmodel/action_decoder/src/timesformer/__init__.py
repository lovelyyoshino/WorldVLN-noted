# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
TimeSformer 包初始化。

原始仓库会在导入时调用依赖 `fvcore` 的 `setup_environment()`。
对于只需要模型定义的轻量脚本（例如 `timesformer.models.vit`），允许在
未安装 `fvcore` 时完成导入。
"""

try:
    from timesformer.utils.env import setup_environment

    setup_environment()
except ModuleNotFoundError:
    # 允许在缺少可选依赖时完成最小化导入（例如 VisionTransformer）。
    pass
