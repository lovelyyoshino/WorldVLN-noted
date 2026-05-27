#!/usr/bin/env python3

import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from fvcore.common.file_io import PathManager
from torch.utils.data.distributed import DistributedSampler

from . import transform as transform

logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    加载图像列表，并在读取失败时自动重试。

    参数：
        image_paths (list): 待加载图像的路径列表。
        retry (int, optional): 最大重试次数，默认 10。
        backend (str): `pytorch` 或 `cv2`。

    返回：
        imgs (list): 已加载的图像列表。
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("读取失败，将重试。")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("加载图像失败：{}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    在对应 clip 内生成采样帧索引序列。

    参数：
        center_idx (int): 当前 clip 的中心帧索引。
        half_len (int): clip 半长度。
        sample_rate (int): clip 内部的帧采样率。
        num_frames (int): 期望的采样帧总数。

    返回：
        seq (list): 当前 clip 中采样得到的帧索引列表。
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def pack_pathway_output(cfg, frames):
    """
        将输入帧整理成 pathway 张量列表。

        参数：
            frames (tensor): 从视频中采样得到的帧，维度为
                说明：`channel` x `num frames` x `height` x `width`。

        返回：
            frame_list (list): pathway 张量列表，每个张量维度均为
                说明：`channel` x `num frames` x `height` x `width`。

    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # 从快速通路中做时间降采样，生成慢速通路。
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "模型架构 {} 不在 {} 中".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
        对给定视频帧执行空间采样。

        当 `spatial_idx` 为 -1 时，执行随机缩放、随机裁剪和随机翻转；
        当 `spatial_idx` 为 0、1 或 2 时，执行确定性的均匀空间采样。

        参数：
            frames (tensor): 从视频中采样得到的帧，维度为
                说明：`num frames` x `height` x `width` x `channel`。
            spatial_idx (int): 若为 -1，则执行随机空间采样；若为 0、1、2，
                则在宽大于高时取左、中、右裁剪，在高大于宽时取上、中、下裁剪。
            min_scale (int): 缩放最小尺寸。
            max_scale (int): 缩放最大尺寸。
            crop_size (int): 裁剪后的高宽尺寸。
            inverse_uniform_sampling (bool): 若为 True，则在
                `[1 / max_scale, 1 / min_scale]` 上均匀采样后取倒数得到尺度；
                否则直接在 `[min_scale, max_scale]` 上均匀采样。

        返回：
            frames (tensor): 空间采样后的帧。

    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # 测试阶段是确定性的，不应执行随机抖动。
        # min_scale、max_scale 和 crop_size 应保持一致。
        # 保留的上游调试/兼容代码：assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames

def spatial_sampling_2crops(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
        对给定视频帧执行双裁剪版本的空间采样。

        参数和行为与 `spatial_sampling` 一致，只是在确定性裁剪阶段调用
        说明：`uniform_crop_2crops`。

        参数：
            frames (tensor): 从视频中采样得到的帧，维度为
                说明：`num frames` x `height` x `width` x `channel`。
            spatial_idx (int): 若为 -1，则执行随机空间采样；若为 0、1、2，
                则执行确定性的均匀空间裁剪。
            min_scale (int): 缩放最小尺寸。
            max_scale (int): 缩放最大尺寸。
            crop_size (int): 裁剪后的高宽尺寸。
            inverse_uniform_sampling (bool): 是否采用倒数均匀采样策略。

        返回：
            frames (tensor): 空间采样后的帧。

    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # 测试阶段是确定性的，不应执行随机抖动。
        # min_scale、max_scale 和 crop_size 应保持一致。
        # 保留的上游调试/兼容代码：assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop_2crops(frames, crop_size, spatial_idx)
    return frames


def as_binary_vector(labels, num_classes):
    """
    根据标签索引列表构造二值标签向量。

    参数：
        labels (list): 输入标签列表。
        num_classes (int): 标签向量的类别总数。

    返回：
        labels (numpy array): 生成的二值向量。
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def aggregate_labels(label_list):
    """
    合并一组标签列表。

    参数：
        labels (list): 输入标签列表。

    返回：
        labels (list): 将所有子列表合并去重后的结果。
    """
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def convert_to_video_level_labels(labels):
    """
    将视频所有帧的标注聚合成视频级标签。

    参数：
        labels (list): 输入标签列表。

    返回：
        labels (list): 与输入结构一致，但每帧标签都会替换成对应视频级标签。
    """
    for video_id in range(len(labels)):
        video_level_labels = aggregate_labels(labels[video_id])
        for i in range(len(labels[video_id])):
            labels[video_id][i] = video_level_labels
    return labels


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
        从 "frame list" 文件中读取图像路径和标签。

        文件每一行格式为：
        说明：`original_vido_id video_id frame_id path labels`

        参数：
            frame_list_file (string): frame list 文件路径。
            prefix (str): 图像路径前缀。
            return_list (bool): 若为 True，返回 list；否则返回 dict。

        返回：
            image_paths (list or dict): 每帧路径组成的列表或字典。
                若 `return_list` 为 False，则返回 dict 形式。
            labels (list or dict): 每帧标签组成的列表或字典。
                若 `return_list` 为 False，则返回 dict 形式。

    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with PathManager.open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()
            # 字段顺序：original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)


def tensor_normalize(tensor, mean, std):
    """
    对张量执行减均值、除标准差的归一化。

    参数：
        tensor (tensor): 待归一化张量。
        mean (tensor or list): 需要减去的均值。
        std (tensor or list): 需要除以的标准差。
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    在 multigrid 训练减少帧数时，随机增大采样率，使部分 clip 仍覆盖原始时间跨度。
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate


def revert_tensor_normalize(tensor, mean, std):
    """
    将归一化后的张量还原为原始尺度。

    参数：
        tensor (tensor): 待还原的张量。
        mean (tensor or list): 需要加回的均值。
        std (tensor or list): 需要乘回的标准差。
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def create_sampler(dataset, shuffle, cfg):
    """
    为给定数据集创建 sampler。

    参数：
        dataset (torch.utils.data.Dataset): 目标数据集。
        shuffle (bool): 若为 ``True``，每个 epoch 都重新打乱数据。
        cfg (CfgNode): 配置对象，细节可参考 `slowfast/config/defaults.py`。

    返回：
        sampler (Sampler): 创建出的采样器。
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    创建传给 PyTorch 数据加载器的 worker 初始化函数。

    参数：
        dataset (torch.utils.data.Dataset): 目标数据集。
    """
    return None
