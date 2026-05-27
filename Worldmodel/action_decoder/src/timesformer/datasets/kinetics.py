# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics 视频加载器。构建后会从视频中采样片段。训练和验证阶段会对
    每个视频随机采样一个片段，并进行随机裁剪、缩放和翻转；测试阶段则会
    对每个视频均匀采样多个片段，并执行确定性的空间裁剪。若宽大于高，则
    取左、中、右裁剪；否则取上、中、下裁剪。
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
                用给定的 csv 文件构建 Kinetics 视频加载器。csv 格式如下：
                ```
                说明：path_to_video_1 label_1
                说明：path_to_video_2 label_2
                ...
                说明：path_to_video_N label_N
                ```
                参数：
                    cfg (CfgNode): 配置对象。
                    mode (string): 可选 `train`、`val` 或 `test`。
                        训练和验证模式从对应集合中读取数据，并为每个视频采样一个片段；
                        测试模式从测试集读取数据，并为每个视频采样多个片段。
                    num_retries (int): 重试次数。

        """
        # 仅支持 train、val 和 test 三种模式。
        assert mode in [
            "train",
            "val",
            "test",
        ], "Kinetics 不支持 split '{}'".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # 训练/验证阶段每个视频只采一个片段；测试阶段每个视频采
        # NUM_ENSEMBLE_VIEWS 个片段，并对每个片段再做 NUM_SPATIAL_CROPS
        # 次空间裁剪。
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("正在构建 Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        构建视频加载器。
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "未找到 {} 目录".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "从 {} 加载 Kinetics split {} 失败".format(
            path_to_file, self._split_idx
        )
        logger.info(
            "正在从 {} 构建 Kinetics 数据加载列表（样本数：{}）".format(
                path_to_file, len(self._path_to_videos)
            )
        )

    def __getitem__(self, index):
        """
                给定视频索引，若视频能成功读取和解码，则返回帧、标签与索引；
                否则重复尝试随机替换成一个可解码的视频。
                参数：
                    index (int): PyTorch 采样器提供的视频索引。
                返回：
                    frames (tensor): 从视频采样得到的帧，维度为
                        说明：`channel` x `num frames` x `height` x `width`。
                    label (int): 当前视频标签。
                    index (int): 若原始视频可解码，则返回原索引；否则返回可解码替换
                        视频的索引。

        """
        short_cycle_idx = None
        # 使用 short cycle 时，输入索引会是一个 tuple。
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 表示随机采样。
            temporal_sample_index = -1
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
                # 缩小尺度等价于在采样网格中使用更大的“跨度”。
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index 取值为 [0, 1, 2]。宽大于高时对应左/中/右，
            # 否则对应上/中/下。
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # 测试阶段应为确定性流程，不执行抖动。
            # min_scale、max_scale 与 crop_size 应保持一致。
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "不支持 {} 模式".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # 尝试解码并采样视频片段；若失败，则重复寻找可解码的随机替代视频。
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "从 {} 加载视频失败，错误为 {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # 当前视频无法访问时，随机选一个替代视频。
            if video_container is None:
                logger.warning(
                    "加载视频元信息失败，索引 {}，路径 {}；第 {} 次尝试".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # 继续尝试另一个视频。
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # 解码视频。meta 信息用于选择性解码。
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # 若解码失败（格式错误、视频过短等），则换一个视频。
            if frames is None:
                logger.warning(
                    "解码视频失败，索引 {}，路径 {}；第 {} 次尝试".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # 继续尝试另一个视频。
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue


            label = self._labels[index]

            # 执行颜色归一化。
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # 将张量维度从 `T H W C` 调整为 `C T H W`。
            frames = frames.permute(3, 0, 1, 2)
            # 执行数据增强。
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )


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
        else:
            raise RuntimeError(
                "重试 {} 次后仍未能获取视频。".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        返回：
            (int): 数据集中视频的数量。
        """
        return len(self._path_to_videos)
