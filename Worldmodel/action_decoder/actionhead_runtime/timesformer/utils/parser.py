# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""命令行参数解析函数。"""

import argparse
import sys

import timesformer.utils.checkpoint as cu
from timesformer.config.defaults import get_cfg


def parse_args():
    """
        为 PySlowFast 用户解析默认命令行参数。

        参数说明：
            shard_id (int): 当前机器的 shard id，从 0 到 `num_shards - 1`；若只
                使用单机，则设置为 0。
            num_shards (int): 当前任务使用的 shard 数量。
            init_method (str): 多设备启动时的初始化方式，可选 TCP 或共享文件系统；
                细节见
                引用/来源：https://pytorch.org/docs/stable/distributed.html#tcp-initialization
            cfg (str): 配置文件路径。
            opts (argument): 命令行额外配置，会覆盖文件中的配置项。

    """
    parser = argparse.ArgumentParser(
        description="提供 SlowFast/TimeSformer 视频训练与测试流水线的命令行入口。"
    )
    parser.add_argument(
        "--shard_id",
        help="当前节点的 shard id，范围是 0 到 num_shards - 1。",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="当前任务使用的 shard 总数。",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="分布式初始化方式，例如 TCP 或共享文件系统。",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="配置文件路径。",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="额外配置覆盖项；完整选项见 slowfast/config/defaults.py。",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    根据命令行参数加载并初始化配置。

    参数：
        args (argument): 包含 `shard_id`、`num_shards`、`init_method`、
            `cfg_file` 和 `opts` 等字段。
    """
    # 初始化 cfg。
    cfg = get_cfg()
    # 从配置文件加载配置。
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # 从命令行覆盖配置项。
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # 继承命令行参数中的公共字段。
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # 创建 checkpoint 目录。
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
