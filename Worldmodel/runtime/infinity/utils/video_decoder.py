# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
import io
import math
import numpy as np
from typing import Optional, TypeVar, Union
import collections

try:
    import decord
except ImportError:
    _HAS_DECORD = False
else:
    _HAS_DECORD = True

if _HAS_DECORD:
    decord.bridge.set_bridge('native')

DecordDevice = TypeVar("DecordDevice")

if _HAS_DECORD:
    # 参考：https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
    class VideoReaderWrapper(decord.VideoReader):
        """包装 Decord VideoReader，保证批量读取后回到第 0 帧，减少状态残留带来的坑。"""

        def __init__(self, *args, **kwargs):
            """初始化底层 Decord reader，并显式把读指针归零。"""
            super().__init__(*args, **kwargs)
            self.seek(0)

        def __getitem__(self, key):
            """读取后重置指针，避免下次调用从中途继续读。"""
            frames = super().__getitem__(key)
            self.seek(0)
            return frames
else:
    class VideoReaderWrapper:  # type: ignore[too-many-ancestors]
        """在未安装 decord 时占位，真正使用时会在 EncodedVideoDecord.__init__ 中抛错。"""

        pass


class Video(ABC):
    """
    Video 抽象类提供“按时间窗取样视频帧”的统一接口。

    返回约定统一为 `(frames, relative_frame_indices)`：
    - `frames`：`numpy.ndarray`，通常形状是 `(T, H, W, C)`；
    - `relative_frame_indices`：`numpy.ndarray`，表示相对本 clip 第 0 帧的偏移，
      方便上层把采样帧对齐到局部时间轴。
    """

    @abstractmethod
    def __init__(
        self,
        file: Union[str, io.IOBase],
        video_name: Optional[str] = None,
        decode_audio: bool = True,
    ) -> None:
        """
        参数：
            file (BinaryIO)：包含编码视频的类文件对象，例如 `io.BytesIO` 或 `io.StringIO`。
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        返回：
            视频时长，单位秒。
        """
        pass

    @abstractmethod
    def get_clip(
        self, start_sec: float, end_sec: float, num_samples: int
    ):
        """
        按给定起止秒数从内部视频对象读取帧；视频时间轴总是从 0 秒开始。

        参数：
            start_sec (float)：clip 起始时间，单位秒。
            end_sec (float)：clip 结束时间，单位秒。
            num_samples (int)：在 `[start_sec, end_sec]` 内均匀抽取多少帧。
        返回：
            tuple：`(frames, relative_frame_indices)`。
        """
        pass

    def close(self):
        """释放底层视频句柄。"""
        pass


class EncodedVideoDecord(Video):
    """
    使用 Decord 作为后端，从编码视频中均匀抽帧。

    返回值不是带音频的字典，而是 `(frames_np, relative_frame_indices)`：
    - `frames_np`：RGB `numpy.ndarray`，形状通常为 `(T, H, W, C)`；
    - `relative_frame_indices`：相对于 clip 首帧的帧号偏移。
    """

    def __init__(
        self,
        file: Union[str, io.IOBase],
        video_name: Optional[str] = None,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
    ) -> None:
        """
        参数：
            file str：视频文件路径。
            video_name (str)：可选的视频名称。
            width: int，默认 -1。
                期望的视频输出宽度；如果为 `-1` 则保持不变。
            height: int，默认 -1。
                期望的视频输出高度；如果为 `-1` 则保持不变。
            num_threads: int，默认 0。
                解码线程数；如果为 `0` 则自动选择。
            fault_tol: int，默认 -1。
                fault_tol 控制损坏帧/恢复帧阈值，避免大量帧无法解码时静默返回重复帧。设 `N = recovered frames 数量`：`fault_tol < 0` 时不限制；`0 < fault_tol < 1.0` 且 `N > fault_tol * len(video)` 时抛出 `DECORDLimitReachedError`；`fault_tol > 1` 且 `N > fault_tol` 时也抛出该异常。
        """
        self._video_name = video_name
        if not _HAS_DECORD:
            raise ImportError(
                "使用 EncodedVideoDecord 需要先安装 decord。"
                "CPU 版可执行 `pip install decord`，GPU 版请参考 https://github.com/dmlc/decord"
            )
        try:
            self._av_reader = VideoReaderWrapper(
                uri=file,
                ctx=decord.cpu(0),
                width=width,
                height=height,
                num_threads=num_threads,
                fault_tol=fault_tol,
            )
        except Exception as e:
            raise RuntimeError(f"无法用 Decord 打开视频 {video_name}：{e}")

        self._fps = self._av_reader.get_avg_fps()
        self._duration = float(len(self._av_reader)) / float(self._fps)

    @property
    def name(self) -> Optional[str]:
        """
        返回：
            name：如果设置过视频名称，则返回该名称。
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        返回：
            duration：视频持续时长/结束时间，单位秒。
        """
        return self._duration

    def close(self):
        """释放 Decord reader。"""
        if self._av_reader is not None:
            del self._av_reader
            self._av_reader = None

    def get_clip(
        self, start_sec: float, end_sec: float, num_samples: int
    ):
        """
        在 `[start_sec, end_sec]` 上均匀抽取 `num_samples` 帧。

        返回：
        - `frames_np`：Decord 解码出的 RGB 帧，`numpy.ndarray`，形状通常是 `(T, H, W, C)`；
        - `relative_frame_indices`：相对第 0 个采样帧的偏移，例如 `[0, 4, 8, ...]`。
        """
        if start_sec > end_sec or start_sec > self._duration:
            raise RuntimeError(
                f"Decord 解码时间窗非法：video={self._video_name}, start={start_sec}, end={end_sec}, duration={self._duration}"
            )

        start_idx = math.ceil(self._fps * start_sec)
        end_idx = math.ceil(self._fps * end_sec)
        end_idx = min(end_idx, len(self._av_reader))
        # 可选写法：frame_idxs = list(range(start_idx, end_idx))

        frame_idxs = np.linspace(start_idx, end_idx - 1, num_samples, dtype=int)

        try:
            outputs = self._av_reader.get_batch(frame_idxs)
            return outputs.asnumpy(), frame_idxs - frame_idxs[0]
        except Exception as e:
            print(f"Decord 解码视频失败：{self._video_name}。{e}")
            raise e

    def get_frames(self, frame_idxs):
        """中文说明：按绝对帧号读取帧。"""
        frame_idxs = np.asarray(frame_idxs, dtype=np.int64)
        if frame_idxs.size == 0:
            raise ValueError("frame_idxs 不能为空")
        try:
            outputs = self._av_reader.get_batch(frame_idxs.tolist())
            return outputs.asnumpy(), frame_idxs - frame_idxs[0]
        except Exception as e:
            print(f"Decord 按帧号解码失败：{self._video_name}。{e}")
            raise e

try:
    import cv2
except ImportError:
    print("错误：导入 cv2 失败，可执行 `pip install opencv-python` 安装。")

class EncodedVideoOpencv():
    """
    使用 OpenCV 作为后端解码视频。

    与 Decord 版本保持同一返回约定：返回 `(frames_np, relative_frame_indices)`，
    其中 `frames_np` 的通道顺序是 OpenCV 默认的 BGR。
    """
    def __init__(
        self,
        file: Union[str, io.IOBase],
        video_name: Optional[str] = None,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
    ) -> None:
        """
        参数：
            file str：视频文件路径。
            video_name (str)：可选的视频名称。
            width / height / num_threads / fault_tol：
                当前 OpenCV 实现里这些参数只是为了和 Decord 版本对齐签名，
                还没有真正参与解码控制。
        """

        self._video_name = video_name
        self.cap = cv2.VideoCapture(file)
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._vlen = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = float(self._vlen) / float(self._fps)

    @property
    def name(self) -> Optional[str]:
        """
        返回：
            name：如果设置过视频名称，则返回该名称。
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        返回：
            duration：视频持续时长/结束时间，单位秒。
        """
        return self._duration

    def __del__(self):
        """对象析构时尝试释放 VideoCapture。"""
        self.close()

    def close(self):
        """释放 OpenCV VideoCapture。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_clip(
        self, start_sec: float, end_sec: float, num_samples: int
    ):
        """
        在 `[start_sec, end_sec]` 内均匀抽取 `num_samples` 帧。

        说明：
        - OpenCV 的 `VideoCapture` 是有状态的；这里每次都会重置到第 0 帧重新扫描，
          保证同一个对象多次调用 `get_clip()` 时返回稳定结果；
        - 返回的 `frames` 是 BGR `numpy.ndarray`，不是 RGB，也不包含音频。
        """
        if start_sec > end_sec or start_sec > self._duration:
            raise RuntimeError(
                f"OpenCV 解码时间窗非法：video={self._video_name}, start={start_sec}, end={end_sec}, duration={self._duration}"
            )
        start_idx = math.ceil(self._fps * start_sec)
        end_idx = math.ceil(self._fps * end_sec)
        end_idx = min(end_idx, self._vlen)
        frame_idxs = np.linspace(start_idx, end_idx - 1, num_samples, dtype=int)
        frame_idx2freq = collections.defaultdict(int)
        for frame_idx in frame_idxs:
            frame_idx2freq[frame_idx] += 1
        try:
            frames = []
            # OpenCV reader 会记住当前读到哪一帧，所以每次 clip 采样前都显式回到开头。
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(self._vlen):
                if i > frame_idxs[-1]:
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break
                if i in frame_idx2freq:
                    frames.extend([frame] * frame_idx2freq[i])
            frames = np.array(frames).astype(np.uint8)  # 图像通道顺序为 BGR。
            if len(frames) != num_samples:
                raise RuntimeError(f"OpenCV 实际解码到 {len(frames)} 帧，但请求的是 {num_samples} 帧")
            return frames, frame_idxs - frame_idxs[0]
        except Exception as e:
            print(f"OpenCV 解码视频失败：{self._video_name}。{e}")
            raise e

    def get_frames(self, frame_idxs):
        """中文说明：按绝对帧号读取 BGR 帧。"""
        frame_idxs = np.asarray(frame_idxs, dtype=np.int64)
        if frame_idxs.size == 0:
            raise ValueError("frame_idxs 不能为空")
        frame_idx2freq = collections.defaultdict(int)
        for idx in frame_idxs.tolist():
            frame_idx2freq[int(idx)] += 1
        max_idx = int(frame_idxs.max())
        try:
            frames = []
            # 确保从视频开头开始读取。
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(self._vlen):
                if i > max_idx:
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break
                if i in frame_idx2freq:
                    frames.extend([frame] * frame_idx2freq[i])
            frames = np.array(frames).astype(np.uint8)
            if len(frames) != int(frame_idxs.size):
                raise RuntimeError(f"OpenCV 实际解码到 {len(frames)} 帧，但请求的是 {int(frame_idxs.size)} 帧")
            return frames, frame_idxs - frame_idxs[0]
        except Exception as e:
            print(f"OpenCV 按帧号解码失败：{self._video_name}。{e}")
            raise e
