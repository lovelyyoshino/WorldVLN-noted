# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""TimeSformer 数据加载性能测试工具。"""

import numpy as np
import pprint
import torch
import tqdm
from fvcore.common.timer import Timer

import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
from timesformer.datasets import loader
from timesformer.utils.env import setup_environment

logger = logging.get_logger(__name__)


def benchmark_data_loading(cfg):
    """压测数据加载速度，帮助定位数据加载器或解码瓶颈。 小白阅读时先看函数签名中的参数，再顺着函数体查看张量形状或评估字段如何变化。

    数据流提示：输入参数进入函数后通常会被裁剪、变形、聚合或跨进程同步；返回值会继续交给 数据加载器、模型、评估器或检查点流程。
    """
    # 设置运行环境。
    setup_environment()
    # 根据配置设置随机种子。
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # 设置日志格式。
    logging.setup_logging(cfg.OUTPUT_DIR)

    # 打印配置。
    logger.info("使用以下配置压测数据加载：")
    logger.info(pprint.pformat(cfg))

    timer = Timer()
    dataloader = loader.construct_loader(cfg, "train")
    logger.info("初始化 loader 用时 {:.2f} 秒。".format(timer.seconds()))
    # 统计跨机器的总 batch size。
    batch_size = cfg.TRAIN.BATCH_SIZE * cfg.NUM_SHARDS
    log_period = cfg.BENCHMARK.LOG_PERIOD
    epoch_times = []
    # 只测试少量 epoch 的加载速度。
    for cur_epoch in range(cfg.BENCHMARK.NUM_EPOCHS):
        timer = Timer()
        timer_epoch = Timer()
        iter_times = []
        if cfg.BENCHMARK.SHUFFLE:
            loader.shuffle_dataset(dataloader, cur_epoch)
        for cur_iter, _ in enumerate(tqdm.tqdm(dataloader)):
            if cur_iter > 0 and cur_iter % log_period == 0:
                iter_times.append(timer.seconds())
                ram_usage, ram_total = misc.cpu_mem_usage()
                logger.info(
                    "Epoch {}: {} 次迭代（{} 个视频）用时 {:.2f} 秒。"
                    "RAM 使用量：{:.2f}/{:.2f} GB。".format(
                        cur_epoch,
                        log_period,
                        log_period * batch_size,
                        iter_times[-1],
                        ram_usage,
                        ram_total,
                    )
                )
                timer.reset()
        epoch_times.append(timer_epoch.seconds())
        ram_usage, ram_total = misc.cpu_mem_usage()
        logger.info(
            "Epoch {}: 总计 {} 次迭代（{} 个视频）用时 {:.2f} 秒。"
            "RAM 使用量：{:.2f}/{:.2f} GB。".format(
                cur_epoch,
                len(dataloader),
                len(dataloader) * batch_size,
                epoch_times[-1],
                ram_usage,
                ram_total,
            )
        )
        logger.info(
            "Epoch {}: 平均每 {} 次迭代（{} 个视频）用时 {:.2f}/{:.2f} "
            "秒（均值/标准差）。".format(
                cur_epoch,
                log_period,
                log_period * batch_size,
                np.mean(iter_times),
                np.std(iter_times),
            )
        )
    logger.info(
        "On average every epoch ({} videos) takes {:.2f}/{:.2f} "
        "(avg/std) seconds.".format(
            len(dataloader) * batch_size,
            np.mean(epoch_times),
            np.std(epoch_times),
        )
    )
