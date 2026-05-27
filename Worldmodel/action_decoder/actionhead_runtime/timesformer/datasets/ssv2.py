# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import random
from itertools import chain as chain
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ssv2(torch.utils.data.Dataset):
    """
        Something-Something v2 (SSV2) 视频加载器。
        它先构建视频帧列表，再从视频中采样 clip。训练和验证时，每个视频随机采样
        一个 clip，并配合随机裁剪、缩放和翻转。测试时，每个视频均匀采样多个
        clip，并做固定裁剪：宽大于高时取左、中、右，反之取上、中、下。

    """

    def __init__(self, cfg, mode, num_retries=10):
        """
                加载 Something-Something V2 的帧路径、标签等信息到 Dataset 对象。
                数据集可从 Something-Something 官网下载：
                引用/来源：https://20bn.com/datasets/something-something。
                数据格式说明见 datasets/DATASET.md。

                参数：
                    cfg (CfgNode): 配置对象。
                    mode (string): 取值包括 `train`、`val` 或 `test`。
                        train/val 模式从对应集合读取数据，每个视频采样一个 clip；
                        test 模式从测试集合读取数据，每个视频采样多个 clip。
                    num_retries (int): 从磁盘读取帧失败后的重试次数。

        """
        # 仅支持 train、val 和 test 三种模式。
        assert mode in [
            "train",
            "val",
            "test",
        ], "Something-Something V2 不支持 split='{}'".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # 训练或验证时，每个视频只采样一个 clip。
        # 测试时，每个视频采样 NUM_ENSEMBLE_VIEWS 个 clip；
        # 每个 clip 再按 NUM_SPATIAL_CROPS 做空间裁剪。
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("正在构建 Something-Something V2 {} 数据集...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        构建视频加载列表。
        """
        # 加载标签名称。
        with PathManager.open(
            os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "something-something-v2-labels.json",
            ),
            "r",
        ) as f:
            label_dict = json.load(f)

        # 加载标签。
        label_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "something-something-v2-{}.json".format(
                "train" if self.mode == "train" else "validation"
            ),
        )
        with PathManager.open(label_file, "r") as f:
            label_json = json.load(f)

        self._video_names = []
        self._labels = []
        for video in label_json:
            video_name = video["id"]
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_dict[template])
            self._video_names.append(video_name)
            self._labels.append(label)

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "{}.csv".format("train" if self.mode == "train" else "val"),
        )
        assert PathManager.exists(path_to_file), "未找到 {} 目录".format(
            path_to_file
        )

        self._path_to_videos, _ = utils.load_image_lists(
            path_to_file, self.cfg.DATA.PATH_PREFIX
        )

        assert len(self._path_to_videos) == len(self._video_names), (
            len(self._path_to_videos),
            len(self._video_names),
        )


        # 从 dict 转成按样本顺序排列的 list。
        new_paths, new_labels = [], []
        for index in range(len(self._video_names)):
            if self._video_names[index] in self._path_to_videos:
                new_paths.append(self._path_to_videos[self._video_names[index]])
                new_labels.append(self._labels[index])

        self._labels = new_labels
        self._path_to_videos = new_paths

        # 测试阶段 self._num_clips > 1 时，按 clip 数扩展样本列表。
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )
        logger.info(
            "已从 {} 构建 Something-Something V2 数据加载列表，样本数为 {}".format(
                path_to_file, len(self._path_to_videos)
            )
        )

    def __getitem__(self, index):
        """
                按视频索引读取采样帧、标签和视频索引。

                参数：
                    index (int): PyTorch 采样器提供的视频索引。

                返回：
                    frames (tensor): 从视频采样到的帧，维度为
                        说明：`channel` x `num frames` x `height` x `width`。
                    label (int): 当前视频的标签。
                    index (int): 当前视频索引。

        """
        short_cycle_idx = None
        # 使用 short cycle 时，输入 index 是一个 tuple。
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]: # 中文说明：or self.cfg.MODEL.ARCH in ['resformer', 'vit']:
            # -1 表示随机采样。
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
            )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # 减小 scale 等价于在采样网格中使用更大的 span。
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # `spatial_sample_index` 取 [0, 1, 2]。
            # 宽大于高时分别表示左、中、右；
            # 高大于宽时分别表示上、中、下。
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1

            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # 测试阶段是确定性的，不应做随机抖动。
            # min_scale、max_scale 和 crop_size 应保持相同。
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError("不支持 {} 模式".format(self.mode))

        label = self._labels[index]

        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(self._path_to_videos[index])


        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,
            )
        )

        # 执行颜色归一化。
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # 将张量维度从 `T H W C` 调整为 `C T H W`。
        frames = frames.permute(3, 0, 1, 2)
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        # 保留的上游调试/兼容代码：if not self.cfg.RESFORMER.ACTIVE:
        if not self.cfg.MODEL.ARCH in ['vit']:
            frames = utils.pack_pathway_output(self.cfg, frames)
        else:
            # 从快速通路中执行时间采样。
            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )
        return frames, label, index, {}

    def __len__(self):
        """
                返回：
                    (int): 数据集中的视频数量。

        """
        return len(self._path_to_videos)
