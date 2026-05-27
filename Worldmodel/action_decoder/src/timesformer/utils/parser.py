# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""命令行参数和配置加载工具。"""

import argparse
import sys

import timesformer.utils.checkpoint as cu
from timesformer.config.defaults import get_cfg


def parse_args():
    """解析 TimeSformer/PySlowFast 风格命令行参数。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
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
    """从配置文件和命令行 opts 合并生成最终配置。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或checkpoint流程。
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
