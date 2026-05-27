# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
timesformer.models 包初始化。

上游项目会导入依赖 `fvcore` 的 build 工具。
对于轻量用法（例如只导入 `timesformer.models.vit.VisionTransformer`），
允许在未安装 `fvcore` 时完成导入。
"""

try:
    from .build import MODEL_REGISTRY, build_model  # noqa
    from .custom_video_model_builder import *  # noqa
    from .video_model_builder import ResNet, SlowFast  # noqa
except ModuleNotFoundError:
    # 允许在缺少可选依赖时完成最小化导入。
    pass
