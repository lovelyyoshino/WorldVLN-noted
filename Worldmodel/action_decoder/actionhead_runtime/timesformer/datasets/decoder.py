# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import random
import torch
import torchvision.io as io


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
        根据起止帧索引，在两者之间等间隔采样 `num_samples` 帧。

        参数：
            frames (tensor): 视频帧张量，维度为
                说明：`num video frames` x `channel` x `height` x `width`。
            start_idx (int): 起始帧索引。
            end_idx (int): 结束帧索引。
            num_samples (int): 采样帧数。

        返回：
            frames (tensor): 时间采样后的视频帧张量，维度为
                说明：`num clip frames` x `channel` x `height` x `width`。

    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    从总长度为 `video_size` 的视频中采样一个长度为 `clip_size` 的片段，并返回
    该片段首尾帧的索引。

    当 `clip_idx` 为 -1 时执行随机采样；否则将整段视频均匀划分为 `num_clips`
    个片段，并返回第 `clip_idx` 个片段的起止索引。

    参数：
        video_size (int): 视频总帧数。
        clip_size (int): 需要采样的片段长度。
        clip_idx (int): 若为 -1，则执行随机抖动采样；若大于 -1，则在均匀划分出的
            `num_clips` 个片段中选取第 `clip_idx` 个片段。
        num_clips (int): 测试时从给定视频中均匀采样的片段总数。

    返回：
        start_idx (int): 起始帧索引。
        end_idx (int): 结束帧索引。
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # 随机时间采样。
        start_idx = random.uniform(0, delta)
    else:
        # 按给定索引均匀采样片段。
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    使用 PyAV 解码视频流。

    参数：
        container (container): PyAV 容器对象。
        start_pts (int): 要读取视频帧的起始 PTS 时间戳。
        end_pts (int): 解码帧的结束 PTS 时间戳。
        stream (stream): PyAV 视频流对象。
        stream_name (dict): 流名字典。例如 `{"video": 0}` 表示索引为 0 的
            视频流。
        buffer_size (int): 超过 `end_pts` 后仍额外解码的帧数。

    返回：
        result (list): 解码出的帧列表。
        max_pts (int): 视频序列中的最大 PTS 时间戳。
    """
    # 流定位本身并不精确，因此向前多回退一小段 PTS 作为余量。
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def torchvision_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta,
    num_clips=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
):
    """
        使用 TorchVision 解码视频，并按需要做时间选择性解码。

        如果 `video_meta` 非空，则利用其中元信息只解码目标片段；若为空，则先解码
        整段视频并补全 `video_meta`。

        参数：
            video_handle (bytes): 视频文件的原始字节流。
            sampling_rate (int): 帧采样率，即相邻采样帧的间隔。
            num_frames (int): 需要采样的帧数。
            clip_idx (int): 若为 -1，则执行随机时间采样；若大于 -1，则将视频均匀
                划分为 `num_clips` 个片段，并选取第 `clip_idx` 个片段。
            video_meta (dict): 包含 VideoMetaData 的字典，细节见
                说明：`pytorch/vision/torchvision/io/_video_opt.py`。
            num_clips (int): 从给定视频中均匀采样的片段总数。
            target_fps (int): 输入视频 fps 可能不同，解码时会映射到目标 fps。
            modalities (tuple): 要解码的模态元组。目前只支持 `visual`。
            max_spatial_scale (int): 解码时短边允许的最大空间尺寸。

        返回：
            frames (tensor): 解码得到的视频帧。
            fps (float): 视频帧率。
            decode_all_video (bool): 若为 True，表示解码了整段视频。

    """
    # 将字节流转成张量。
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # `video_meta` 为空时，从原始视频中探测元信息。
    if len(video_meta) == 0:
        # 记录元信息，供后续选择性解码复用。
        meta = io._probe_video_from_memory(video_tensor)
        # 后续使用 `video_meta` 中的信息执行选择性解码。
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]
    if (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
    ):
        # 尝试选择性解码。
        decode_all_video = False
        clip_size = sampling_rate * num_frames / target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            fps * video_meta["video_duration"], clip_size, clip_idx, num_clips
        )
        # 将帧索引换算成 pts。
        pts_per_frame = video_meta["video_denominator"] / fps
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

    # 使用 TorchVision 解码原始视频。
    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream="visual" in modalities,
        video_width=0,
        video_height=0,
        video_min_dimension=max_spatial_scale,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )

    if v_frames.shape == torch.Size([0]):
        # 选择性解码失败时，退回整段解码。
        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
        )

    return v_frames, fps, decode_all_video


def pyav_decode(
    container, sampling_rate, num_frames, clip_idx, num_clips=10, target_fps=30, start=None, end=None
, duration=None, frames_length=None):
    """
    使用 PyAV 解码视频，并在可能时按目标 fps 执行选择性解码。

    若视频头包含足够的解码信息，则只解码目标时间片段；否则退回整段解码。

    参数：
        container (container): PyAV 容器对象。
        sampling_rate (int): 帧采样率，即相邻采样帧的间隔。
        num_frames (int): 需要采样的帧数。
        clip_idx (int): 若为 -1，则执行随机时间采样；若大于 -1，则将视频均匀
            划分为 `num_clips` 个片段，并选取第 `clip_idx` 个片段。
        num_clips (int): 从给定视频中均匀采样的片段总数。
        target_fps (int): 输入视频 fps 可能不同，采样前会转换到目标 fps。

    返回：
        frames (tensor): 解码得到的视频帧；若未找到视频流则返回 None。
        fps (float): 视频帧率。
        decode_all_video (bool): 若为 True，表示解码了整段视频。
    """
    # 先尝试从视频头读取解码信息；部分视频拿不到这些信息，此时 duration 为 None。
    fps = float(container.streams.video[0].average_rate)

    orig_duration = duration
    tb = float(container.streams.video[0].time_base)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration
    if duration is None and orig_duration is not None:
       duration = orig_duration / tb

    if duration is None:
        # 如果无法读取解码信息，则退回整段解码。
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # 执行选择性解码。
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            frames_length,
            sampling_rate * num_frames / target_fps * fps,
            clip_idx,
            num_clips,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    if start is not None and end is not None:
        decode_all_video = False

    frames = None
    # 如果存在视频流，则读取视频帧。
    if container.streams.video:
        if start is None and end is None:
            video_frames, max_pts = pyav_decode_stream(
                container,
                video_start_pts,
                video_end_pts,
                container.streams.video[0],
                {"video": 0},
            )
        else:
            timebase = duration / frames_length
            start_i = start
            end_i = end
            video_frames, max_pts = pyav_decode_stream(
                container,
                start_i,
                end_i,
                container.streams.video[0],
                {"video": 0},
            )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))

    return frames, fps, decode_all_video


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    start=None,
    end=None,
    duration=None,
    frames_length=None,
):
    """
        解码视频并执行时间采样。

        参数：
            container (container): PyAV 容器或 TorchVision 使用的视频句柄。
            sampling_rate (int): 帧采样率，即相邻采样帧的间隔。
            num_frames (int): 需要采样的帧数。
            clip_idx (int): 若为 -1，则执行随机时间采样；若大于 -1，则将视频均匀
                划分为 `num_clips` 个片段，并选取第 `clip_idx` 个片段。
            num_clips (int): 从给定视频中均匀采样的片段总数。
            video_meta (dict): 包含 VideoMetaData 的字典，细节见
                说明：`pytorch/vision/torchvision/io/_video_opt.py`。
            target_fps (int): 输入视频 fps 可能不同，采样前会转换到目标 fps。
            backend (str): 解码后端，支持 `pyav` 和 `torchvision`，默认 `pyav`。
            max_spatial_scale (int): 保持宽高比前提下，限制帧短边最大尺寸；仅
                `torchvision` 后端使用。

        返回：
            frames (tensor): 解码并采样后的视频帧。

    """
    # 当前支持两种解码器：1）PyAV，2）TorchVision。
    assert clip_idx >= -1, "无效的 clip_idx {}".format(clip_idx)
    try:
        if backend == "pyav":
            frames, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips,
                target_fps,
                start,
                end,
                duration,
                frames_length,
            )
        elif backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
            )
        else:
            raise NotImplementedError(
                "未知的视频解码后端 {}".format(backend)
            )
    except Exception as e:
        print("使用 {} 解码失败，异常信息：{}".format(backend, e))
        return None

    # 解码失败或没有有效帧时返回 None。
    if frames is None or frames.size(0) == 0:
        return None

    clip_sz = sampling_rate * num_frames / target_fps * fps
    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        clip_sz,
        clip_idx if decode_all_video else 0,
        num_clips if decode_all_video else 1,
    )
    # 对解码后的视频执行时间采样。
    frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    return frames
