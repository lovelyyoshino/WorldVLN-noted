# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from typing import Tuple
import tempfile
import os
import fsspec

TMP_DIR = None


def get_fsspec(path: str):
    """中文说明：`get_fsspec` 实现文件系统与本地缓存辅助工具中的 `get_fsspec` 步骤，供训练、推理或调试流程复用。

    新手提示：它把远端路径、本地临时目录和 fsspec 协议统一起来，方便 checkpoint/数据读取。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    def get_protocol(path: str) -> Tuple[fsspec.spec.AbstractFileSystem, str]:
        """中文说明：`get_protocol` 实现文件系统与本地缓存辅助工具中的 `get_protocol` 步骤，供训练、推理或调试流程复用。

        新手提示：它把远端路径、本地临时目录和 fsspec 协议统一起来，方便 checkpoint/数据读取。
        阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
        """
        return fsspec.core.url_to_fs(path)

    if isinstance(path, str):
        return get_protocol(path)

    # 未知路径类型默认按本地路径处理
    return fsspec.filesystem("local"), path


def get_temp_dir():
    """中文说明：`get_temp_dir` 实现文件系统与本地缓存辅助工具中的 `get_temp_dir` 步骤，供训练、推理或调试流程复用。

    新手提示：它把远端路径、本地临时目录和 fsspec 协议统一起来，方便 checkpoint/数据读取。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    global TMP_DIR
    if TMP_DIR:
        return TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()
    return TMP_DIR


def retrieve_local_path(path: str, worker_id):
    """中文说明：`retrieve_local_path` 实现文件系统与本地缓存辅助工具中的 `retrieve_local_path` 步骤，供训练、推理或调试流程复用。

    新手提示：它把远端路径、本地临时目录和 fsspec 协议统一起来，方便 checkpoint/数据读取。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    local_path = os.path.join("/dev/shm/", get_temp_dir().name.lstrip("/"), str(worker_id), path.lstrip("/"))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path
