# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""数据加载器。"""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from timesformer.datasets.multigrid_helper import ShortCycleBatchSampler

from . import utils as utils
from .build import build_dataset


def detection_collate(batch):
    """
    检测任务的数据合并函数。将不同样本的 bboxes、labels 和 metadata
    沿第一维拼接，而不是堆叠出 batch 维。
    参数：
        batch (tuple or list): 待合并的一批数据。
    返回：
        (tuple): 拼接后的检测任务 batch。
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # 在拼接前给每个 bbox 追加样本索引信息。
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data


def construct_loader(cfg, split, is_precise_bn=False):
    """
        为给定数据集构建数据加载器。
        参数：
            cfg (CfgNode): 配置。详见
                说明：slowfast/config/defaults.py
            split (str): 数据加载切分。可选 `train`、
                说明：`val` 和 `test`。

    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # 构建数据集。
    dataset = build_dataset(dataset_name, cfg, split)

    if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
        # 为多进程训练创建采样器。
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        batch_sampler = ShortCycleBatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
        )
        # 创建数据加载器。
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:
        # 为多进程训练创建采样器。
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        # 创建数据加载器。
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    打乱数据顺序。
    参数：
        loader (loader): 要打乱读取顺序的数据加载器。
        cur_epoch (int): 当前 epoch 编号。
    """
    sampler = (
        loader.batch_sampler.sampler
        if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
        else loader.sampler
    )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "不支持的采样器类型 '{}'".format(type(sampler))
    # 随机采样器会自动处理 shuffle。
    if isinstance(sampler, DistributedSampler):
        # 分布式采样器会根据 epoch 打乱数据。
        sampler.set_epoch(cur_epoch)
