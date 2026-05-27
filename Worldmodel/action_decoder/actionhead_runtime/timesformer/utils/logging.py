# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""日志工具。"""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
try:
    import simplejson as _json  # type: ignore
except Exception:  # pragma: no cover
    import json as _json
from fvcore.common.file_io import PathManager

import timesformer.utils.distributed as du


def _suppress_print():
    """
    屏蔽当前进程的 print 输出。
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        """替代 print 的空函数，用于让非主进程保持安静。"""
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    """打开并缓存日志文件流，进程退出时自动关闭。"""
    io = PathManager.open(filename, "a", buffering=1024)
    atexit.register(io.close)
    return io


def setup_logging(output_dir=None):
    """
    为多进程训练设置日志。

    只有 master 进程会正常输出日志，其他进程会屏蔽 print，避免重复刷屏。
    """
    # 设置日志格式。
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_master_proc():
        # 为 master 进程启用日志。
        logging.root.handlers = []
    else:
        # 屏蔽非 master 进程的输出。
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if du.is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None and du.is_master_proc(du.get_world_size()):
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
        获取指定名称的 logger。

        如果 name 为 None，则返回日志层级中的 root logger。
        参数：
            name (string): logger 的名称。

    """
    return logging.getLogger(name)


# 兼容下游旧代码：这个文件过去会直接导入 simplejson。
simplejson = _json


def log_json_stats(stats):
    """
        以 JSON 字符串形式记录统计信息。

        参数：
            stats (dict): 需要写入日志的统计信息字典。

    """
    stats = {
        k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
