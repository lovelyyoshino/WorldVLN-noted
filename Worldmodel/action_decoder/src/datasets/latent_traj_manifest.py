import json
import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import signal

"""
WorldVLN action-decoder 训练流程使用的 manifest 数据集。

中文导读：
Stage A 和 Stage B 都从这里读取同一类样本。每个样本由三部分组成：
- `latent_path`：视频 VAE 编码后的 latent，通常是 Stage-2 VAE decode 需要的 `z_ext`；
- `traj_json_path`：专家轨迹或动作增量标签；
- `images_dir`：真实 RGB 帧目录，Stage A 用它构造 TimesFormer teacher tokens。

这层数据集不负责训练逻辑，只负责把文件系统中的多种格式统一成训练脚本可消费的
`z_ext`、`traj`、`frames_rgb`。
"""


@dataclass(frozen=True)
class LatentTrajManifestItem:
    """manifest 中的一行样本；这里保存的路径已经解析成文件系统绝对路径。"""

    latent_path: str
    traj_json_path: str
    images_dir: str


def _resolve_path(p: str, workspace_root: str) -> str:
    """把 manifest 里的相对路径解析到 workspace_root 下，绝对路径保持不变。"""
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(workspace_root, p))


class LatentTrajManifestDataset(Dataset):
    """
        从 manifest JSON 读取样本的数据集，manifest 里可以包含一个或多个 `items_*` 列表。

        每个样本会提供：
        列表项说明：- z_ext: (1,64,T_lat,16,16) float32
        - frames_rgb: (3,T,H,W) float32（可选，由 load_frames 控制）
        列表项说明：- traj: (T,6) float32

        中文导读：
        - Stage A 需要 `z_ext + frames_rgb`：用 RGB 帧通过 TimesFormer patch_embed 得到 teacher token，
          用 `z_ext` 通过 VAE decoder hook + Adapter 得到 student token。
        - Stage B 需要 `z_ext + traj`：用 latent 生成 token，再监督 TimesFormer 输出动作增量。
        - `require_T` 用来约束轨迹长度，默认常见为 49 帧；设为 None/0 时允许变长样本。

    """

    def __init__(
        self,
        manifest_json: str,
        items_key: str = "ALL",
        workspace_root: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        load_frames: bool = False,
        load_traj: bool = True,
        max_items: Optional[int] = None,
        require_T: Optional[int] = 49,
        io_timeout_s: float = 0.0,
        on_error: str = "raise",
    ):
        """
        读取 manifest 并构造样本索引。

        `items_key` 可以是单个 split、逗号分隔 split，或 `ALL` 合并所有 `items_*`；
        真正的 latent/trajectory/RGB 数据在 `__getitem__()` 中按需加载。
        """
        self.manifest_json = str(manifest_json)
        self.items_key = str(items_key)
        self.workspace_root = str(workspace_root).strip() if workspace_root else os.getcwd()
        self.transform = transform
        self.load_frames = bool(load_frames)
        self.load_traj = bool(load_traj)
        self.require_T = int(require_T) if require_T is not None else None
        self.io_timeout_s = float(io_timeout_s) if io_timeout_s is not None else 0.0
        self.on_error = str(on_error).strip().lower() if on_error is not None else "raise"
        if self.on_error not in ("raise", "empty"):
            raise ValueError(f"on_error 只能是 'raise' 或 'empty'，实际为 {on_error!r}")

        # Manifest 约定格式：
        # {
        # 代码/形状说明："items_train": [{"latent_path": "...", "traj_json_path": "...", "images_dir": "..."}],
        # 代码/形状说明："items_val": [...]
        # }
        # `items_key=ALL` 会让训练脚本合并所有 `items_*` split，便于快速实验。
        with open(self.manifest_json, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("manifest 顶层必须是 JSON dict")

        key_s = str(self.items_key).strip()
        if key_s.lower() == "all":
            keys = [k for k, v in data.items() if isinstance(k, str) and k.startswith("items_") and isinstance(v, list)]
            if len(keys) == 0:
                raise ValueError("manifest 中没有形如 items_* 且值为 list 的 split")
        else:
            keys = [k.strip() for k in key_s.split(",") if k.strip()]
            for k in keys:
                if k not in data:
                    raise ValueError(f"manifest 缺少 key={k}")
                if not isinstance(data.get(k), list):
                    raise ValueError(f"manifest[{k}] 必须是 list")

        raw_items: List[Dict[str, Any]] = []
        for k in keys:
            raw_items.extend(data.get(k, []))

        items: List[LatentTrajManifestItem] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            lp = it.get("latent_path")
            tp = it.get("traj_json_path")
            im = it.get("images_dir")
            if not (lp and tp and im):
                continue
            items.append(
                LatentTrajManifestItem(
                    # 发布版 manifest 里的路径可能是相对 repo 的；这里统一解析，训练代码就不用再猜基准目录。
                    latent_path=_resolve_path(lp, self.workspace_root),
                    traj_json_path=_resolve_path(tp, self.workspace_root),
                    images_dir=_resolve_path(im, self.workspace_root),
                )
            )

        if max_items is not None:
            items = items[: int(max_items)]
        if len(items) == 0:
            raise FileNotFoundError(f"manifest={self.manifest_json} key={self.items_key} 中没有可用样本")

        self.items = items

    @contextmanager
    def _io_timeout(self):
        """
        对慢网络文件系统做尽力而为的实际耗时超时保护。
        只有 Unix 且可用 signal 时才会生效，否则等价于“不做任何事”。
        """
        sec = float(self.io_timeout_s)
        if sec <= 0:
            yield
            return
        if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(_signum, _frame):
            """SIGALRM 处理器：把阻塞 I/O 转成 Python TimeoutError。"""
            raise TimeoutError(f"I/O 超时：超过 {sec:.1f}s")

        old = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, sec)
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old)
            except Exception:
                pass

    def __len__(self) -> int:
        """返回 manifest 中可用样本数量。"""
        return len(self.items)

    def _load_latents(self, path: str) -> torch.Tensor:
        """
        从原始 Tensor checkpoint 或 {"latents": Tensor} 中读取 latent tensor。

        中文导读：
        Stage-2 动作解码器期望的是 patchified `z_ext`，常见 shape 为
        `(1, 64, T_lat, 16, 16)`。这里不做语义转换，只保证 Tensor/float/contiguous。
        """
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "latents" in obj:
            z = obj["latents"]
        else:
            z = obj
        if not isinstance(z, torch.Tensor):
            raise ValueError(f"{path} 中的 latents 不是 Tensor")
        if z.ndim != 5:
            raise ValueError(f"{path} 中 latents 期望 ndim=5，实际 shape={tuple(z.shape)}")
        return z.float().contiguous()

    def _load_traj(self, path: str) -> np.ndarray:
        """
        读取轨迹或动作标签，并把常见保存格式统一成 `(T, 6)`。

        中文导读：
        训练动作头时，标签必须和帧时间线对齐。若输入是 `(T-1,6)` 的 per-step action6，
        这里会补一行 `delta[0]=0` 变成 `(T,6)`，这样第 i 行可理解为“到第 i 帧的增量”。
        公式写法是：`delta[0] = 0`；对 `t >= 1`，`delta[t] = action6[t-1]`。
        """
        with open(path, "r") as f:
            arr = json.load(f)

        # 常见格式：
        # 代码/形状说明：1) List[[x,y,z,roll,yaw,pitch], ...]  -> (T,6)
        # 2) Dict，把列表放在已知 key 下（preprocessed_logs / processed_logs / traj / poses / logs）
        # 3) IndoorUAV "processed_log.json" 格式：dict 里有长度为 (T-1,6) 的 action6，布局为：
        #    [dz(dyaw), dy, dx, tx, ty, tz]，其中 dz 可能是 rad 或 deg，取决于生成脚本版本。
        if isinstance(arr, dict):
            for k in ("preprocessed_logs", "processed_logs", "traj", "poses", "logs"):
                v = arr.get(k, None)
                if isinstance(v, list):
                    arr = v
                    break
            else:
                a6 = arr.get("action6", None)
                if isinstance(a6, list) and (len(a6) == 0 or isinstance(a6[0], (list, tuple))):
                    action6 = np.asarray(a6, dtype=np.float32)
                    if action6.ndim != 2 or action6.shape[1] < 6:
                        raise ValueError(f"{path} 中 action6 必须是 (T-1,6+)，实际 shape={action6.shape}")

                    # 尽量把 dz(dyaw) 转成 radians（与 IndoorUavF49Dataset 的做法保持一致）。
                    # 布局说明：历史工具把 action6 存成 [dz(dyaw), dy, dx, tx, ty, tz]，
                    # server API 暴露的是 [dx, dy, dz, droll, dyaw, dpitch]。
                    # 转换限定在 dataset 内部，让 Stage B 始终面对同一种标签布局。
                    unit = "auto"
                    meta = arr.get("meta") if isinstance(arr.get("meta"), dict) else {}
                    layout = meta.get("action6_layout")
                    layout_s = " ".join([str(x) for x in layout]).lower() if isinstance(layout, list) else str(layout or "").lower()
                    if "rad" in layout_s:
                        unit = "rad"
                    elif "deg" in layout_s:
                        unit = "deg"
                    else:
                        dz = action6[:, 0]
                        p95 = float(np.nanpercentile(np.abs(dz), 95)) if dz.size else 0.0
                        unit = "deg" if p95 > 1.0 else "rad"
                    if unit == "deg":
                        action6 = action6.copy()
                        action6[:, 0] = action6[:, 0] * (np.pi / 180.0)

                    # 把逐步 action6 (T-1,6) 转成 (T,6) 序列：delta[0]=0，delta[t]=action6[t-1]。
                    T = int(action6.shape[0]) + 1
                    delta = np.zeros((T, 6), dtype=np.float32)
                    delta[1:, :] = action6[:, :6].astype(np.float32)
                    return delta

        traj = np.asarray(arr, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] < 6:
            raise ValueError(f"{path} 中 traj 必须是 (T,6+)，实际 shape={traj.shape}")
        return traj[:, :6]

    def _frame_path(self, images_dir: str, idx: int) -> str:
        """按 `frame_000000.*` 命名查找 RGB 帧，兼容 png/jpg/jpeg/webp。"""
        p = os.path.join(images_dir, f"frame_{idx:06d}.png")
        if os.path.exists(p):
            return p
        for ext in (".jpg", ".jpeg", ".webp"):
            p2 = os.path.join(images_dir, f"frame_{idx:06d}{ext}")
            if os.path.exists(p2):
                return p2
        raise FileNotFoundError(f"找不到帧文件：{images_dir} idx={idx}")

    def _load_frames(self, images_dir: str, T: int) -> torch.Tensor:
        """
        为 Stage A teacher-token 蒸馏读取 RGB 帧。

        中文导读：
        Stage A 的 teacher 不是 latent，而是真实 RGB 帧经过 TimesFormer 的 patch_embed。
        返回 shape 为 `(3,T,H,W)`，collate 后会变成 `(B,3,T,H,W)`。
        """
        if self.transform is None:
            raise ValueError("load_frames=True 时必须提供 transform")
        frames: List[torch.Tensor] = []
        for i in range(int(T)):
            img = Image.open(self._frame_path(images_dir, i)).convert("RGB")
            x = self.transform(img)
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 3:
                raise ValueError("transform(img) 必须返回 shape 为 (3,H,W) 的 Tensor")
            frames.append(x.unsqueeze(0))
        x = torch.cat(frames, dim=0)  # (T,3,H,W)
        return x.transpose(0, 1).contiguous()  # (3,T,H,W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个训练样本；需要时可用空样本作为兜底。

        中文导读：
        网络盘或大规模数据集偶尔会出现慢 I/O/坏样本。`on_error="empty"` 会返回
        时间长度为 0 的 latent，让训练脚本在 collate 或 step 中跳过，而不是让整轮任务崩掉。
        """
        it = self.items[int(idx)]
        try:
            with self._io_timeout():
                z_ext = self._load_latents(it.latent_path)  # (1,64,T_lat,16,16)
            traj = None
            T = None
            if self.load_traj:
                with self._io_timeout():
                    traj = self._load_traj(it.traj_json_path)  # (T,6)
                T = int(traj.shape[0])
                if self.require_T is not None and T != int(self.require_T):
                    raise ValueError(f"require_T={self.require_T} but got T={T} for {it.traj_json_path}")
        except Exception as e:
            if self.on_error == "empty":
                # 返回空 latent，让训练代码可以安全跳过这个样本。
                z_ext = torch.empty((1, 64, 0, 16, 16), dtype=torch.float32)
                traj = np.zeros((0, 6), dtype=np.float32) if self.load_traj else None
                T = 0
                err_s = f"{type(e).__name__}: {e}"
                out: Dict[str, Any] = {
                    "z_ext": z_ext,
                    "meta": {
                        "latent_path": it.latent_path,
                        "traj_json_path": it.traj_json_path,
                        "images_dir": it.images_dir,
                        "error": err_s[:500],
                    },
                }
                if self.load_traj:
                    out["traj"] = traj
                return out
            raise

        out: Dict[str, Any] = {
            "z_ext": z_ext,
            "meta": {
                "latent_path": it.latent_path,
                "traj_json_path": it.traj_json_path,
                "images_dir": it.images_dir,
            },
        }
        if self.load_traj:
            out["traj"] = traj
        if self.load_frames:
            if T is None:
                raise ValueError("load_frames=True 需要同时设置 load_traj=True，以便获得时间长度 T")
            out["frames_rgb"] = self._load_frames(it.images_dir, T=T)
        return out
