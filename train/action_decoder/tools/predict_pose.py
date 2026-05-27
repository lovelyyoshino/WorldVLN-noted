"""
面向 UAV-Flow 风格 route 目录的 Stage-2 latent-to-action 批量推理脚本。

中文导读：
这个脚本用于离线验证 Stage-2 动作头，不经过在线 FastAPI 服务。它直接读取每条 route 的
`latents.pt` 和 GT 初始位姿，用训练好的 adapter+TimesFormer checkpoint 预测每个相邻帧的
6D delta action，再把 delta 积分成绝对轨迹。

阅读主线：
1. `main()` 加载 checkpoint、VAE、adapter、TimesFormer，并发现 route 列表。
2. `infer_one_route()` 处理单条 route：latent -> VAE decoder hook -> tokens -> 滑窗动作。
3. `_decode_tokens_full_T()` 解释为什么这里调用 VAE decode 但真正使用的是 decoder 中间特征。
4. `integrate_trajectory_se3()` 把相对动作按 SE(3) 连乘成绝对 `[roll,yaw,pitch,x,y,z]`。

每个 route 目录的输入：
- latents.pt（Tensor shape 为 (1,64,T_lat,16,16)）
- preprocessed_logs.json（长度为 T 的列表，每行 [x,y,z,roll,yaw,pitch]；角度默认
  是 degrees，平移单位由 --translation_divisor 控制）

可以显式传 --ckpt，也可以把 --stage2_root 指到包含多个 run 子目录的目录，
脚本会在其中寻找 checkpoint_last.pth。

每个 route 的输出：
- deltas.npy: (T,6)，每步为 [dz,dy,dx,tx,ty,tz]（rad + meters），delta[0]=0
- window_deltas.npy: (N,3,6)，每个滑窗的 3 个动作预测
- trajectory.npy / trajectory.json: (T,6) 绝对 [roll,yaw,pitch,x,y,z]（rad + meters）
- pred_path.json / pred_actions.json: 兼容旧格式的输出，包含 actions6 和积分后的轨迹
- metrics.json（可选）：做单位转换后与 preprocessed_logs 对齐计算 RMSE
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.path.abspath(os.path.join(_TOOLS_DIR, ".."))
_OPEN_ROOT = os.path.abspath(os.path.join(_TRAIN_ROOT, "..", ".."))
_ARCH_ROOT = os.path.join(_OPEN_ROOT, "Worldmodel", "action_decoder", "src")
if _ARCH_ROOT not in sys.path:
    sys.path.insert(0, _ARCH_ROOT)

from datasets.utils import euler_to_rotation, rotation_to_euler  # noqa: E402
from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter  # noqa: E402
from timesformer.models.vit import VisionTransformer  # noqa: E402

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:  # noqa: E402
    from tools.train_stage2_latent2action_ddp import _add_infinitystar_to_syspath, load_infinitystar_vae
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    def _add_infinitystar_to_syspath(inf_root: Optional[str], proj_root: str):
        """兜底导入工具：把 InfinityStar runtime 路径加入 sys.path。"""
        if inf_root and os.path.isdir(inf_root):
            p = os.path.abspath(inf_root)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
        open_root = os.path.abspath(os.path.join(proj_root, "..", "..", ".."))
        candidates = [
            os.environ.get("INFINITYSTAR_ROOT", ""),
            os.environ.get("INFINITYSTAR_HOME", ""),
            os.path.join(open_root, "Worldmodel", "runtime"),
            os.path.join(open_root, "Worldmodel"),
        ]
        for cand in candidates:
            if cand and os.path.isdir(cand):
                p = os.path.abspath(cand)
                if p not in sys.path:
                    sys.path.insert(0, p)
                return

    def load_infinitystar_vae(
        vae_path: str,
        vae_type: int,
        device: torch.device,
        infinitystar_root: Optional[str],
        proj_root: str,
        semantic_scale_dim: int,
        detail_scale_dim: int,
        use_learnable_dim_proj: int,
        detail_scale_min_tokens: int,
        use_feat_proj: int,
        semantic_scales: int,
    ):
        """兜底 VAE 加载器：按 Stage-2 checkpoint 记录的 VAE 参数构建 video VAE。"""
        _add_infinitystar_to_syspath(infinitystar_root, proj_root=proj_root)
        from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import (  # type: ignore
            video_vae_model,
        )

        global_args = SimpleNamespace(
            semantic_scale_dim=int(semantic_scale_dim),
            detail_scale_dim=int(detail_scale_dim),
            use_learnable_dim_proj=int(use_learnable_dim_proj),
            detail_scale_min_tokens=int(detail_scale_min_tokens),
            use_feat_proj=int(use_feat_proj),
            semantic_scales=int(semantic_scales),
        )
        vae = video_vae_model(
            vqgan_ckpt=str(vae_path),
            schedule_mode="dynamic",
            codebook_dim=int(vae_type),
            global_args=global_args,
            test_mode=True,
        ).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae


def build_tsformer() -> VisionTransformer:
    """
    构建 Stage-2 latent-to-action 使用的 4 帧 TimesFormer 动作头。

    关键原因：
    Stage-2 训练时每个窗口看 4 帧，因此会预测 3 个相邻帧转移；
    每个转移是 6 维动作，所以 head 输出维度固定为 `3 * 6 = 18`。

    注意：
    这里的结构必须和 Stage-2 训练脚本中的 TimesFormer 完全一致，否则 checkpoint
    即使能 `strict=False` 加载，动作头语义也可能错位。
    """
    # 必须和 Stage-2 训练脚本中的 TimesFormer 结构保持一致，否则 checkpoint 无法正确加载。
    from functools import partial

    return VisionTransformer(
        img_size=(192, 640),
        num_classes=18,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=4,
        attention_type="divided_space_time",
    )


def _find_latest_stage2_checkpoint(stage2_root: str) -> str:
    """当用户没有显式传 --ckpt 时，从 run 目录里找最新的 checkpoint_last.pth。"""
    root = os.path.abspath(stage2_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"找不到 stage2_root 目录：{root}")

    best_p = None
    best_m = -1.0
    for d in os.listdir(root):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        p = os.path.join(dd, "checkpoint_last.pth")
        if not os.path.exists(p):
            continue
        try:
            m = os.path.getmtime(p)
        except Exception:
            continue
        if m > best_m:
            best_m = m
            best_p = p

    if best_p is None:
        raise FileNotFoundError(f"在 {root} 下没有找到 checkpoint_last.pth")
    return best_p


def _load_latents(path: str) -> torch.Tensor:
    """读取 route 的 latent；兼容原始 Tensor 和 {"latents": Tensor} 两种保存格式。"""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "latents" in obj:
        z = obj["latents"]
    else:
        z = obj
    if not isinstance(z, torch.Tensor) or z.ndim != 5:
        raise ValueError(f"latents 必须是 5D Tensor，在 {path} 收到的是 {type(z)}，形状={getattr(z,'shape',None)}")
    return z.float().contiguous()

def _safe_torch_load(path: str):
    """
    兼容 PyTorch 2.6 `weights_only=True` 默认行为的 checkpoint 加载函数。

    中文说明：
    新版 PyTorch 会默认把 `torch.load()` 当成“只加载权重 tensor”，这对只含 state_dict
    的纯权重文件很好，但某些 Stage-2 checkpoint 里还带有 `numpy` 元数据
    （例如 `label_stats`），此时就需要回退到 `weights_only=False`。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # PyTorch >= 2.6 默认 torch.load(..., weights_only=True)，
        # 可能拒绝带 numpy 元数据的 checkpoint（例如 label_stats）。
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_preprocessed_traj(path: str) -> np.ndarray:
    """读取 route 绝对轨迹日志，返回前 6 列 `[x,y,z,roll,yaw,pitch]`。"""
    with open(path, "r") as f:
        arr = json.load(f)
    traj = np.asarray(arr, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 6:
        raise ValueError(f"预处理后的轨迹必须是 (T,6+) 形状，在 {path} 收到的是 {traj.shape}")
    return traj[:, :6]


def _find_traj_json(route_dir: str) -> str:
    """
    重新生成或重采样的 route 可能使用不同文件名。
    如果旧文件名存在，则优先使用旧文件名。

    中文说明：不同预处理脚本可能写 `preprocessed_logs.json` 或 `processed_logs.json`；
    二者都表示每帧绝对位姿，后续会裁成前 6 维。
    """
    cand = [
        os.path.join(route_dir, "preprocessed_logs.json"),
        os.path.join(route_dir, "processed_logs.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"route 目录缺少轨迹 json：{route_dir}；已尝试 {cand}")


def _sample_all_window_starts(T: int, window_size: int, stride: int, device: torch.device) -> torch.Tensor:
    """生成 4 帧 TimesFormer 窗口的起点；stride=1 时覆盖所有相邻窗口。"""
    max_start = int(T - window_size)
    if max_start < 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.arange(0, max_start + 1, int(stride), device=device, dtype=torch.long)


def _gather_window_tokens(tokens_tnd: torch.Tensor, starts: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    按滑窗起点收集 TimesFormer 所需的 patch tokens。

    输入：
    - `tokens_tnd`：形状 `(T,N,D)`，表示整条 route 的逐帧 patch token；
    - `starts`：形状 `(K,)`，表示 K 个滑窗的起始帧下标；
    - `window_size`：窗口长度，Stage-2 默认是 4。

    输出：
    - `patch_tokens`：形状 `(K*window_size, N, D)`。

    中文说明：
    TimesFormer 的 `forward_features_from_patch_tokens()` 不直接接收
    `(K,window_size,N,D)`，而是要求先把 K 个窗口拍平为 `(K*window_size,N,D)`，
    然后通过参数 `B=K,T=window_size,W=40` 再还原时空结构。

    """
    T, N, D = tokens_tnd.shape
    K = int(starts.shape[0])
    flat = tokens_tnd.view(T, N * D)
    t_idx = torch.arange(window_size, device=starts.device, dtype=torch.long).view(1, window_size)
    idx = starts.view(K, 1) + t_idx  # 代码/形状说明：(K,window_size)
    idx2 = idx.view(K * window_size, 1).expand(K * window_size, N * D)
    g = flat.gather(0, idx2).view(K * window_size, N, D)
    return g.contiguous()


def _R_from_rpy(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """按 ZYX 顺序从 roll/yaw/pitch 构造旋转矩阵。"""
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    """把旋转矩阵转回 `[roll,yaw,pitch]`。"""
    zyx = rotation_to_euler(R, seq="zyx")  # 代码/形状说明：[yaw,pitch,roll]
    yaw, pitch, roll = float(zyx[0]), float(zyx[1]), float(zyx[2])
    return np.asarray([roll, yaw, pitch], dtype=np.float32)


def integrate_trajectory_se3(deltas_zyx: np.ndarray, init_rpy_rad: np.ndarray, init_pos_m: np.ndarray) -> np.ndarray:
    """
    将每帧相对动作积分为绝对轨迹。

    `deltas_zyx[i] = [dz,dy,dx,tx,ty,tz]`，其中旋转是上一帧坐标系下的 ZYX Euler 增量，
    平移 `t_rel` 也在上一帧坐标系下。因此积分时先 `p = p + R @ t_rel`，再 `R = R @ R_rel`。

    初学者公式：
    - `R_rel = Rz(dz) @ Ry(dy) @ Rx(dx)`，表示从上一帧到当前帧的小旋转；
    - `p_i = p_{i-1} + R_{i-1} @ [tx,ty,tz]`，先把局部平移转到世界坐标再累加；
    - `R_i = R_{i-1} @ R_rel`，把相对旋转接到当前绝对姿态后面；
    - 输出 `traj[i] = [Euler(R_i), p_i]`，即 `[roll,yaw,pitch,x,y,z]`。
    """
    t = int(deltas_zyx.shape[0])
    traj = np.zeros((t, 6), dtype=np.float32)

    roll0, yaw0, pitch0 = float(init_rpy_rad[0]), float(init_rpy_rad[1]), float(init_rpy_rad[2])
    R = _R_from_rpy(roll0, yaw0, pitch0)
    p = init_pos_m.astype(np.float32).copy()

    traj[0, 0:3] = np.asarray([roll0, yaw0, pitch0], dtype=np.float32)
    traj[0, 3:6] = p

    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas_zyx[i, 0:3]]
        t_rel = deltas_zyx[i, 3:6].astype(np.float32)
        R_rel = np.asarray(euler_to_rotation(z=dz, y=dy, x=dx, isRadian=True, seq="zyx"), dtype=np.float32)
        p = p + (R @ t_rel)
        R = R @ R_rel
        traj[i, 0:3] = _rpy_from_R(R)
        traj[i, 3:6] = p
    return traj


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """计算均方根误差，用于可选 metrics.json。"""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _decode_tokens_full_T(
    *,
    vae: nn.Module,
    adapter: nn.Module,
    z_ext: torch.Tensor,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, int]:
    """
    返回：
      `tokens_tnd`：形状 `(T,N,D)`。，保持在同一个 device 上
      T

    中文说明：
    Stage-2 动作头需要 TimesFormer patch token，不需要 RGB 重建结果。
    所以这里给 VAE decoder 注册 forward hook，捕获最后一个 up_block 的 `(B,96,t,H,W)`
    中间特征，再交给 `Vae96ToTSformerEmbedAdapter` 变成 `(T,N,D)`。
    """
    if z_ext.ndim != 5 or int(z_ext.shape[0]) != 1:
        raise ValueError(f"期望 z_ext 形状是 (1,64,T_lat,16,16)，收到的是 {tuple(z_ext.shape)}")

    tokens_slices: List[torch.Tensor] = []
    T_acc = 0

    def hook(_module, _inp, out):
        """VAE decoder hook：收集每个时间切片的 up_block feature 并转成 tokens。"""
        nonlocal tokens_slices, T_acc
        # VAE 可能按时间切片 decode；hook 每触发一次就收集一个 t_slice，最后按时间拼回完整 T。
        hs = out[0] if isinstance(out, (tuple, list)) else out  # 代码/形状说明：(B,96,t_slice,H,W)
        if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
            raise RuntimeError("VAE 最后一个 up block 的 hook 输出不是 5D Tensor")
        Bh = int(hs.shape[0])
        if Bh != 1:
            raise RuntimeError(f"VAE hook 收到非预期 batch 维度：hs={tuple(hs.shape)}")
        t_slice = int(hs.shape[2])
        tok, _t2, _w2 = adapter(hs)  # 代码/形状说明：(Bh*t_slice,N,D)
        tok = tok.view(Bh, t_slice, tok.shape[1], tok.shape[2]).contiguous()  # 代码/形状说明：(1,t_slice,N,D)
        tokens_slices.append(tok[0])
        T_acc += t_slice

    handle = vae.decoder.up_blocks[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(amp_enabled) and torch.cuda.is_available()):
                _ = vae.decode(z_ext, return_dict=False)[0]
    finally:
        handle.remove()

    if len(tokens_slices) == 0 or T_acc <= 0:
        return torch.empty((0, 0, 0), device=z_ext.device, dtype=torch.float32), 0
    tokens_tnd = torch.cat(tokens_slices, dim=0).contiguous()  # (T,N,D)
    return tokens_tnd, int(tokens_tnd.shape[0])


def infer_one_route(
    *,
    route: str,
    route_dir: str,
    ckpt_path: str,
    tsformer: VisionTransformer,
    adapter: nn.Module,
    vae: nn.Module,
    device: torch.device,
    out_root: str,
    stride: int,
    translation_divisor: float,
    angles_in_degrees: bool,
    amp: bool,
    compute_metrics: bool,
    label_stats: Optional[Dict[str, np.ndarray]] = None,
):
    """
    对一条 route 做离线 Stage-2 推理并写出多种兼容格式。

    输入：
    - `latents.pt`: 世界模型或预处理得到的 latent 序列；
    - `preprocessed_logs.json`: 只用第 0 帧作为积分初始位姿，指标计算时才用完整 GT。

    输出：
    - `pred_actions.json`: 下游评估优先读取的相对动作 `[dz,dy,dx,tx,ty,tz]`；
    - `pred_path.json`: 已积分的绝对轨迹 `[roll,yaw,pitch,x,y,z]`；
    - `trajectory_m_deg.*` / `actions6_m_deg.*`: 便于和 GT `[x,y,z,roll,yaw,pitch]` 人工对齐。

    推荐阅读顺序：
    1. 先看 `z_ext -> tokens_tnd`，理解 latent 为什么还能恢复动作；
    2. 再看 4 帧滑窗如何得到 `window_deltas`；
    3. 最后看 `integrate_trajectory_se3()`，理解相对动作如何累计成整条轨迹。
    """
    lat_path = os.path.join(route_dir, "latents.pt")
    if not os.path.exists(lat_path):
        return False
    try:
        traj_path = _find_traj_json(route_dir)
    except FileNotFoundError:
        return False

    z_ext = _load_latents(lat_path).to(device)
    traj_abs = _load_preprocessed_traj(traj_path)  # 代码/形状说明：(T,6) = [x,y,z,roll,yaw,pitch] (deg)
    T = int(traj_abs.shape[0])
    if T < 4:
        return False

    tokens_tnd, T_dec = _decode_tokens_full_T(vae=vae, adapter=adapter, z_ext=z_ext, amp_enabled=bool(amp))
    T_use = min(int(T), int(T_dec))
    if T_use < 4:
        return False
    tokens_tnd = tokens_tnd[:T_use]
    traj_abs = traj_abs[:T_use]

    window_size = 4
    starts = _sample_all_window_starts(T=T_use, window_size=window_size, stride=int(stride), device=device)
    if starts.numel() == 0:
        return False

    patch_tokens = _gather_window_tokens(tokens_tnd, starts=starts, window_size=window_size)
    K = int(starts.shape[0])
    W_grid = 40
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=bool(amp) and torch.cuda.is_available()):
            # 每个 4 帧窗口输出 18 维，即 3 个相邻动作增量 * 每个增量 6 维。
            feat = tsformer.forward_features_from_patch_tokens(patch_tokens, B=K, T=window_size, W=W_grid)
            pred = tsformer.head(feat)  # (K,18)

    pred_f = pred.detach().float()  # (K,18)
    if isinstance(label_stats, dict) and all(k in label_stats for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
        # 反标准化回 (rad, meters)，保证后续评估单位一致。
        # 角度组公式：x_angle = x_norm_angle * std_angles + mean_angles。
        # 平移组公式：x_trans = x_norm_trans * std_t + mean_t。
        ma = torch.as_tensor(label_stats["mean_angles"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        sa = torch.as_tensor(label_stats["std_angles"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        mt = torch.as_tensor(label_stats["mean_t"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        st = torch.as_tensor(label_stats["std_t"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        p = pred_f.view(K, 3, 6)
        p[:, :, 0:3] = p[:, :, 0:3] * sa + ma
        p[:, :, 3:6] = p[:, :, 3:6] * st + mt
        window_deltas = p.cpu().numpy().astype(np.float32)
    else:
        window_deltas = pred_f.cpu().numpy().reshape(K, 3, 6).astype(np.float32)  # (K,3,6)

    # 聚合滑窗预测到逐帧动作：
    # 一个时间点可能被多个窗口预测到，取平均可以降低窗口边界噪声；第 0 帧没有“上一帧 -> 当前帧”动作。
    # 公式：delta[t] = sum(所有覆盖 t 的窗口预测) / count[t]；deltas 先全 0，所以 delta[0]=0。
    # 例子：窗口 [0,1,2,3] 预测的是帧 1/2/3 的动作，窗口 [1,2,3,4] 又会再次预测帧 2/3/4，
    # 因此帧 2、3 会有多个候选值，需要做平均。
    deltas = np.zeros((T_use, 6), dtype=np.float32)
    acc = np.zeros((T_use, 6), dtype=np.float32)
    cnt = np.zeros((T_use,), dtype=np.int32)
    starts_np = starts.detach().cpu().numpy().astype(np.int32).tolist()
    for i, s in enumerate(starts_np):
        for j in range(1, window_size):
            t = int(s + j)
            if 0 <= t < T_use:
                acc[t] += window_deltas[i, j - 1]
                cnt[t] += 1
    mask = cnt > 0
    deltas[mask] = acc[mask] / cnt[mask, None]

    # 初始位姿来自 GT 第 0 帧；之后整条轨迹完全由预测 delta 积分得到。
    # 代码/形状说明：traj_abs[0] layout = [x,y,z,roll,yaw,pitch]。
    init_xyz = traj_abs[0, 0:3].astype(np.float32)
    if float(translation_divisor) != 1.0:
        init_xyz = init_xyz / float(translation_divisor)
    init_rpy = traj_abs[0, 3:6].astype(np.float32)
    if bool(angles_in_degrees):
        init_rpy = init_rpy * (np.pi / 180.0)

    traj_pred = integrate_trajectory_se3(deltas_zyx=deltas, init_rpy_rad=init_rpy, init_pos_m=init_xyz)

    out_one = os.path.join(out_root, route)
    os.makedirs(out_one, exist_ok=True)
    np.save(os.path.join(out_one, "deltas.npy"), deltas.astype(np.float32))
    np.save(os.path.join(out_one, "window_deltas.npy"), window_deltas.astype(np.float32))
    np.save(os.path.join(out_one, "trajectory.npy"), traj_pred.astype(np.float32))
    with open(os.path.join(out_one, "trajectory.json"), "w") as f:
        json.dump(traj_pred.tolist(), f)

    # 写 batch_infer 风格 JSON，方便 eval_endpoints.py 和旧实验脚本复用。
    actions6 = deltas[1:].astype(np.float32)  # (T-1,6)
    pred_actions_json = os.path.join(out_one, "pred_actions.json")
    pred_path_json = os.path.join(out_one, "pred_path.json")
    with open(pred_actions_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "route": route,
                "route_dir": route_dir,
                "latents_pt": lat_path,
                "preprocessed_logs_json": traj_path,
                "ckpt": ckpt_path,
                "actions6_layout": ["rz(dz)_rad", "ry(dy)_rad", "rx(dx)_rad", "tx_m", "ty_m", "tz_m"],
                "actions6": actions6.tolist(),
                "window_size": 4,
                "stride": int(stride),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(pred_path_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "route": route,
                "route_dir": route_dir,
                "latents_pt": lat_path,
                "preprocessed_logs_json": traj_path,
                "ckpt": ckpt_path,
                "start_pose_abs": {
                    "x": float(init_xyz[0]),
                    "y": float(init_xyz[1]),
                    "z": float(init_xyz[2]),
                    "roll_rad": float(init_rpy[0]),
                    "yaw_rad": float(init_rpy[1]),
                    "pitch_rad": float(init_rpy[2]),
                },
                "poses_layout": ["roll_rad", "yaw_rad", "pitch_rad", "x_m", "y_m", "z_m"],
                "poses": traj_pred.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 额外输出“物理顺序 + 物理单位”版本，方便和 GT preprocessed_logs.json 对齐：
    # - GT pose layout: [x,y,z, roll,yaw,pitch]，应用 --translation_divisor 后单位是 (m,deg)
    # - 当前 traj_pred layout: [roll,yaw,pitch,x,y,z]，单位是 (rad,m)
    traj_xyz_m = traj_pred[:, 3:6].astype(np.float32)
    traj_rpy_deg = (traj_pred[:, 0:3] * (180.0 / np.pi)).astype(np.float32)
    traj_m_deg = np.concatenate([traj_xyz_m, traj_rpy_deg], axis=1).astype(np.float32)  # 代码/形状说明：(T,6) [x,y,z,roll,yaw,pitch]
    np.save(os.path.join(out_one, "trajectory_m_deg.npy"), traj_m_deg)
    with open(os.path.join(out_one, "trajectory_m_deg.json"), "w", encoding="utf-8") as f:
        json.dump(traj_m_deg.tolist(), f, ensure_ascii=False)

    # actions6 布局转换：
    # 训练/推理每步 delta 布局为 [dz,dy,dx, tx,ty,tz] (rad,m)，其中 dz=dyaw, dy=dpitch, dx=droll。
    # 这里转成 [tx_m,ty_m,tz_m, droll_deg, dyaw_deg, dpitch_deg]，匹配“xyz 后接 rpy”的物理顺序。
    a = actions6.astype(np.float32)
    trans_m = a[:, 3:6].astype(np.float32)  # [tx,ty,tz] 单位 meters，仍在上一帧坐标系下。
    droll_deg = (a[:, 2] * (180.0 / np.pi)).astype(np.float32)
    dyaw_deg = (a[:, 0] * (180.0 / np.pi)).astype(np.float32)
    dpitch_deg = (a[:, 1] * (180.0 / np.pi)).astype(np.float32)
    actions6_m_deg = np.concatenate(
        [trans_m[:, 0:3], droll_deg[:, None], dyaw_deg[:, None], dpitch_deg[:, None]],
        axis=1,
    ).astype(np.float32)
    np.save(os.path.join(out_one, "actions6_m_deg.npy"), actions6_m_deg)
    with open(os.path.join(out_one, "actions6_m_deg.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "route": route,
                "actions6_layout": ["x_m", "y_m", "z_m", "roll_deg", "yaw_deg", "pitch_deg"],
                "actions6": actions6_m_deg.tolist(),
                "note": "平移分量仍然位于上一帧坐标系中；这里只做单位和字段顺序转换，没有改参考坐标系。",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(out_one, "infer_meta.json"), "w") as f:
        json.dump(
            {
                "route": route,
                "ckpt_path": ckpt_path,
                "created_at": datetime.now().isoformat(),
                "T_use": int(T_use),
                "stride": int(stride),
                "translation_divisor": float(translation_divisor),
                "angles_in_degrees": bool(angles_in_degrees),
            },
            f,
            indent=2,
        )

    if compute_metrics:
        ref_xyz = traj_abs[:, 0:3].astype(np.float32)
        if float(translation_divisor) != 1.0:
            ref_xyz = ref_xyz / float(translation_divisor)
        ref_rpy = traj_abs[:, 3:6].astype(np.float32)
        if bool(angles_in_degrees):
            ref_rpy = ref_rpy * (np.pi / 180.0)
        ref_traj = np.concatenate([ref_rpy, ref_xyz], axis=1).astype(np.float32)  # (T,6) [rpy,xyz]
        metrics = {
            "len": int(T_use),
            "rmse_xyz_m": rmse(traj_pred[:, 3:6], ref_traj[:, 3:6]),
            "rmse_rpy_rad": rmse(traj_pred[:, 0:3], ref_traj[:, 0:3]),
        }
        with open(os.path.join(out_one, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return True


def main():
    """
    CLI 入口：加载模型和 route 列表，逐条调用 `infer_one_route()` 写预测结果。

    小白使用建议：
    - 第一次阅读时，先只跑 `--first_n 1` 看单条 route 输出目录里会生成哪些文件；
    - 然后对照 `pred_actions.json`、`trajectory.json`、`metrics.json` 回到代码里看每一步来源。

    整个入口可分成三段读：
    1. checkpoint/模型准备：决定 TSFormer、Adapter、InfinityStar VAE 从哪里加载；
    2. route 列表准备：决定这次到底处理哪些 route；
    3. 单条 route 推理：进入 `infer_one_route()`，真正执行 `latents -> tokens -> deltas -> trajectory`。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage2_root",
        type=str,
        default="./checkpoints/stage2_latent2action",
        help="Stage-2 checkpoint 根目录。仅当 --ckpt 为空时才会使用，并在其中自动寻找最新 checkpoint_last.pth。",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help=(
            "显式指定 Stage-2 checkpoint 文件路径。支持 checkpoint_last.pth、"
            "checkpoint_pre_fulltrain_e*.pth、stage2_latent2action_combined.pt 等格式。"
            "如果留空，脚本会在 --stage2_root 下自动查找最新的 checkpoint_last.pth。"
        ),
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default="./data/uavflow_latents",
        help="UAV-Flow 风格的 route 根目录。每个 route 子目录至少应包含 latents.pt 和 preprocessed_logs.json。",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/stage2_latent2action",
        help="推理输出根目录；每条 route 会在这里生成一个同名子目录。",
    )
    ap.add_argument("--device", type=str, default="cuda:0", help="推理设备，例如 cuda:0 或 cpu；CUDA 不可用时会自动回退到 cpu。")
    ap.add_argument("--stride", type=int, default=1, help="4 帧滑窗的步长；1 表示尽量覆盖所有相邻窗口。")
    ap.add_argument("--translation_divisor", type=float, default=1.0, help="GT 平移量的单位缩放因子；例如原始单位是厘米时可设为 100。")
    ap.add_argument("--angles_in_degrees", action="store_true", default=True, help="表示 GT 角度是 degrees，脚本会转换为 radians 后积分。")
    ap.add_argument("--amp", action="store_true", default=True, help="启用 CUDA AMP 混合精度推理；仅在 CUDA 可用时生效。")
    ap.add_argument("--compute_metrics", action="store_true", default=True, help="推理后计算 metrics.json，用于和 GT 轨迹做快速 RMSE 对齐检查。")
    ap.add_argument("--first_n", type=int, default=0, help="只对前 N 条 route 做推理；0 表示全部执行。")
    ap.add_argument(
        "--routes",
        type=str,
        default="",
        help="可选：只推理指定的 route 目录名，多个名称用英文逗号分隔。",
    )
    # 为空或 0 时，优先从 Stage-2 checkpoint 元数据读取（推荐）。
    ap.add_argument(
        "--infinitystar_vae_path",
        type=str,
        default="",
        help="可选覆盖项：手动指定 InfinityStar VAE 权重路径；留空时优先从 Stage-2 checkpoint 元数据读取。",
    )
    ap.add_argument(
        "--infinitystar_vae_type",
        type=int,
        default=0,
        help="可选覆盖项：手动指定 VAE codebook 通道数；0 表示从 checkpoint 元数据读取，读不到时默认 64。",
    )
    ap.add_argument("--infinitystar_root", type=str, default="", help="可选：手动指定 InfinityStar runtime 根目录；留空时按环境变量和项目默认路径查找。")
    ap.add_argument("--tqdm", action="store_true", default=True, help="如果本机安装了 tqdm，则显示推理进度条。")
    args = ap.parse_args()

    # 第 1 段：checkpoint/模型准备。
    # checkpoint 支持训练期 checkpoint_last.pth，也支持整理后的 stage2_latent2action_combined.pt。
    ckpt_path = str(args.ckpt).strip() or _find_latest_stage2_checkpoint(str(args.stage2_root))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 ckpt：{ckpt_path}")

    os.makedirs(str(args.out_dir), exist_ok=True)
    device = torch.device(str(args.device) if (torch.cuda.is_available() and str(args.device).startswith("cuda")) else "cpu")

    ckpt = _safe_torch_load(ckpt_path)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint 必须是 dict（组合后的 Stage-2 checkpoint）")

    tsformer = build_tsformer().to(device).eval()
    adapter = Vae96ToTSformerEmbedAdapter().to(device).eval()

    # 初始化 InfinityStar VAE。adapter 的输入来自这个 VAE decoder 的中间层 feature，
    # 所以 VAE 结构参数必须和 Stage-2 训练 checkpoint 记录的一致。
    inf_root = str(args.infinitystar_root).strip() or None
    _add_infinitystar_to_syspath(inf_root, proj_root=_ARCH_ROOT)
    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    vae_path = (
        str(args.infinitystar_vae_path).strip()
        or str(ckpt.get("infinitystar_vae_path", "")).strip()
        or str(ckpt_args.get("infinitystar_vae_path", "")).strip()
    )
    if not vae_path:
        raise ValueError(
            "必须提供 InfinityStar VAE 路径。请显式传入 --infinitystar_vae_path，"
            "或把 infinitystar_vae_path 写进 Stage-2 checkpoint 元数据。"
        )
    vae_type = int(args.infinitystar_vae_type) if int(args.infinitystar_vae_type) > 0 else int(
        ckpt.get("infinitystar_vae_type", ckpt_args.get("infinitystar_vae_type", 64))
    )
    print(f"[配置] InfinityStar VAE 配置：path={vae_path} type={vae_type}", flush=True)
    vae = load_infinitystar_vae(
        vae_path=str(vae_path),
        vae_type=int(vae_type),
        device=device,
        infinitystar_root=inf_root,
        proj_root=_ARCH_ROOT,
        semantic_scale_dim=int(ckpt_args.get("semantic_scale_dim", 16)),
        detail_scale_dim=int(ckpt_args.get("detail_scale_dim", 64)),
        use_learnable_dim_proj=int(ckpt_args.get("use_learnable_dim_proj", 0)),
        detail_scale_min_tokens=int(ckpt_args.get("detail_scale_min_tokens", 350)),
        use_feat_proj=int(ckpt_args.get("use_feat_proj", 2)),
        semantic_scales=int(ckpt_args.get("semantic_scales", 11)),
    )

    # 支持的格式：
    # 代码/形状说明：- checkpoint_last.pth: {model_state_dict, adapter_state_dict, vae_state_dict, ...}
    # - checkpoint_pre_fulltrain_e*.pth: 同上
    # 代码/形状说明：- stage2_latent2action_combined.pt: {tsformer_state_dict, adapter_state_dict, vae_state_dict, model_state_dict(alias), ...}
    ts_sd = ckpt.get("model_state_dict") or ckpt.get("tsformer_state_dict")
    ad_sd = ckpt.get("adapter_state_dict") or ckpt.get("state_dict")  # 兜底。
    vae_sd = ckpt.get("vae_state_dict", {})
    if not isinstance(ts_sd, dict) or not isinstance(ad_sd, dict):
        raise ValueError("checkpoint 缺少 model_state_dict 或 adapter_state_dict")
    missing, unexpected = tsformer.load_state_dict(ts_sd, strict=False)
    if missing or unexpected:
        print(f"[警告] TSFormer 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}", flush=True)
    missing, unexpected = adapter.load_state_dict(ad_sd, strict=False)
    if missing or unexpected:
        print(f"[警告] Adapter 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}", flush=True)
    if isinstance(vae_sd, dict) and len(vae_sd) > 0:
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
        if missing or unexpected:
            print(f"[警告] VAE 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}", flush=True)

    # 可选 label stats：如果 Stage-2 训练时对动作标签做过标准化，这里必须反标准化回 rad/m。
    label_stats = None
    ls = ckpt.get("label_stats")
    if isinstance(ls, dict) and all(k in ls for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
        try:
            label_stats = {
                "mean_angles": np.asarray(ls["mean_angles"], dtype=np.float32).reshape(3),
                "std_angles": np.asarray(ls["std_angles"], dtype=np.float32).reshape(3),
                "mean_t": np.asarray(ls["mean_t"], dtype=np.float32).reshape(3),
                "std_t": np.asarray(ls["std_t"], dtype=np.float32).reshape(3),
            }
            src = ckpt.get("label_stats_source") or "checkpoint"
            print(f"[配置] label_stats 已从 {src} 加载", flush=True)
        except Exception as e:
            print(f"[警告] 无法从 checkpoint 解析 label_stats：{e}", flush=True)
            label_stats = None
    if label_stats is None:
        # 向后兼容的兜底逻辑：尝试在 Stage-2 训练参数记录的 TSFormer 预训练 checkpoint 旁边找 run_config.json。
        try:
            args0 = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
            ts_pre = str(args0.get("tsformer_pretrained", "")).strip()
            if ts_pre:
                run_cfg = os.path.join(os.path.dirname(os.path.abspath(ts_pre)), "run_config.json")
                if os.path.isfile(run_cfg):
                    rc = json.loads(open(run_cfg, "r", encoding="utf-8").read())
                    ls2 = rc.get("label_stats") if isinstance(rc, dict) else None
                    if isinstance(ls2, dict) and all(k in ls2 for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
                        label_stats = {
                            "mean_angles": np.asarray(ls2["mean_angles"], dtype=np.float32).reshape(3),
                            "std_angles": np.asarray(ls2["std_angles"], dtype=np.float32).reshape(3),
                            "mean_t": np.asarray(ls2["mean_t"], dtype=np.float32).reshape(3),
                            "std_t": np.asarray(ls2["std_t"], dtype=np.float32).reshape(3),
                        }
                        print(f"[配置] label_stats 已从 tsformer_pretrained 旁边的 run_config.json 加载：{run_cfg}", flush=True)
        except Exception as e:
            print(f"[警告] label_stats 兜底加载失败：{e}", flush=True)

    # 第 2 段：route 列表准备。
    # data_root 下每个子目录通常对应一条 route，内部至少要有 latents.pt 和 GT pose json。
    routes = [d for d in os.listdir(str(args.data_root)) if os.path.isdir(os.path.join(str(args.data_root), d))]
    routes.sort()
    only = str(args.routes).strip()
    if only:
        # --routes 用于只跑少量指定 route 做快速冒烟检查；保持用户给定顺序并去重。
        wanted = [x.strip() for x in only.split(",") if x.strip()]
        # 保持顺序并去重。
        seen = set()
        wanted2 = []
        for w in wanted:
            if w not in seen:
                wanted2.append(w)
                seen.add(w)
        missing = [w for w in wanted2 if w not in set(routes)]
        if missing:
            raise FileNotFoundError(f"--routes 中包含 data_root 下不存在的目录：{missing}")
        routes = wanted2
    if int(args.first_n) > 0:
        routes = routes[: int(args.first_n)]

    ok = 0
    skipped = 0
    it = routes
    if bool(args.tqdm) and tqdm is not None:
        it = tqdm(routes, desc="推理 routes", dynamic_ncols=True)
    # 第 3 段：逐条 route 推理。
    # 每条 route 最终会在 out_dir/<route>/ 下写出 pred_actions.json、pred_path.json、trajectory*.json 等结果。
    for r in it:
        try:
            did = infer_one_route(
                route=r,
                route_dir=os.path.join(str(args.data_root), r),
                ckpt_path=ckpt_path,
                tsformer=tsformer,
                adapter=adapter,
                vae=vae,
                device=device,
                out_root=str(args.out_dir),
                stride=int(args.stride),
                translation_divisor=float(args.translation_divisor),
                angles_in_degrees=bool(args.angles_in_degrees),
                amp=bool(args.amp),
                compute_metrics=bool(args.compute_metrics),
                label_stats=label_stats,
            )
            ok += int(bool(did))
            if not did:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"[警告] 跳过 route={r}，错误={e}", flush=True)

    print(json.dumps({"ok": ok, "skipped": skipped, "ckpt": ckpt_path, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
