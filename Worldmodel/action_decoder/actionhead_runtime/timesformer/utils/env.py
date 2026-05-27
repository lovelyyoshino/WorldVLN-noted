# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""设置 TimeSformer 运行环境。"""

import timesformer.utils.logging as logging

_ENV_SETUP_DONE = False


def setup_environment():
    """只执行一次全局环境初始化，避免重复设置共享状态。"""
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
