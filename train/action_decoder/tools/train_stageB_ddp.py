"""
使用 InfinityStar VAE + Adapter + TSformer 进行 Stage-2（latent -> action/VO）训练。

主要目标：
- 支持变长轨迹（require_T 可以关闭）。
- 支持两种 batch 组装策略：
  - crop：按 batch 内最短长度裁剪后堆叠（快，但会改变长度）
  - per_sample：保留每个样本原始长度，在一个 step 内逐样本训练（不裁剪、不 padding）
- tqdm 和 train.log 只在 rank0 使用。

中文导读：
Stage B 才是真正的 latent-to-action 训练。它复用 Stage A 得到的 Adapter，把
`latents.pt` 转成 TimesFormer patch tokens，并用专家轨迹中的相邻位姿增量监督动作头。
因此训练链路是：
公式/形状说明：`latent segment -> VAE decoder feature -> Adapter tokens -> TimesFormer -> 6D delta action`。
重点关注 `_normalize_delta_bt6()`、`build_tsformer()`、VAE hook/adapter 调用，以及窗口
输出如何和专家动作标签对齐。
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

# 即使通过绝对路径启动，也要确保 action-decoder 架构代码可 import。
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.path.abspath(os.path.join(_TOOL_DIR, ".."))
_OPEN_ROOT = os.path.abspath(os.path.join(_TRAIN_ROOT, "..", ".."))
_ARCH_ROOT = os.path.join(_OPEN_ROOT, "Worldmodel", "action_decoder", "src")
if _ARCH_ROOT not in sys.path:
    sys.path.insert(0, _ARCH_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.latent_traj_manifest import LatentTrajManifestDataset
from datasets.utils import euler_to_rotation, rotation_to_euler
from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter
from timesformer.models.vit import VisionTransformer

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


_PROJ_ROOT = _ARCH_ROOT
_RANK0_LOG_FH = None


def seed_everything(seed: int):
    """设置 Python/NumPy/Torch/CUDA 随机种子，降低多卡训练随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    """判断当前进程是否处于 torch.distributed 训练。"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """返回当前进程 rank；单进程时为 0。"""
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    """返回 DDP world size；单进程时为 1。"""
    return dist.get_world_size() if is_dist() else 1


def rank0_print(*args, **kwargs):
    """只在 rank0 打印，并写入可选 train.log。"""
    if get_rank() == 0:
        global _RANK0_LOG_FH
        print(*args, **kwargs, flush=True)
        if _RANK0_LOG_FH is not None:
            msg = " ".join(str(a) for a in args)
            _RANK0_LOG_FH.write(msg + "\n")
            _RANK0_LOG_FH.flush()


def ddp_setup() -> int:
    """按 torchrun 环境变量初始化 DDP，并返回 local_rank。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """跨 rank 平均一个标量 tensor，用于日志指标。"""
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= get_world_size()
    return y


def _allreduce_grads(param_groups):
    """在所有 DDP rank 之间平均可训练参数的梯度。"""
    if not is_dist():
        return
    ws = float(get_world_size())
    for pg in param_groups:
        for p in pg["params"]:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(ws)


def _read_json(path: str):
    """读取 UTF-8 JSON；主要用于 label_stats/run_config。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_load_label_stats(*, label_stats_json: str, tsformer_pretrained: str) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
    """
    加载 TSformer 训练时使用的标签归一化统计量（mean/std）。

    返回：
      返回 `(stats, source_path)`。
      - stats keys: mean_angles,std_angles,mean_t,std_t，都是形状为 (3,) 的 np.ndarray
      - 找不到时 stats 为 None
    """
    cand = str(label_stats_json).strip()
    if cand:
        p = os.path.abspath(cand)
        if os.path.isfile(p):
            obj = _read_json(p)
            if isinstance(obj, dict) and "label_stats" in obj and isinstance(obj["label_stats"], dict):
                obj = obj["label_stats"]
            if not isinstance(obj, dict):
                raise ValueError(f"label_stats_json 必须是 dict，或是包含 label_stats 的 run_config：{p}")
            out: Dict[str, np.ndarray] = {}
            for k in ("mean_angles", "std_angles", "mean_t", "std_t"):
                if k not in obj:
                    raise ValueError(f"label_stats_json 缺少字段 key={k}：{p}")
                out[k] = np.asarray(obj[k], dtype=np.float32).reshape(3)
            return out, p

    # 尝试读取预训练 checkpoint 旁边的 run_config.json。
    base = os.path.dirname(os.path.abspath(str(tsformer_pretrained)))
    p2 = os.path.join(base, "run_config.json")
    if os.path.isfile(p2):
        obj = _read_json(p2)
        if isinstance(obj, dict) and "label_stats" in obj and isinstance(obj["label_stats"], dict):
            ls = obj["label_stats"]
            out = {
                "mean_angles": np.asarray(ls["mean_angles"], dtype=np.float32).reshape(3),
                "std_angles": np.asarray(ls["std_angles"], dtype=np.float32).reshape(3),
                "mean_t": np.asarray(ls["mean_t"], dtype=np.float32).reshape(3),
                "std_t": np.asarray(ls["std_t"], dtype=np.float32).reshape(3),
            }
            return out, p2

    return None, ""


def _normalize_delta_bt6(delta_bt6: torch.Tensor, stats_t: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    delta_bt6: (B,T,6)，单位 (rad, meters)。
    返回与 UavflowSimDataset 一致的归一化 (B,T,6)。

    初学者公式：
    - 角度组 `x_angle = [dz,dy,dx]`：`x_norm_angle = (x_angle - mean_angles) / std_angles`。
    - 平移组 `x_trans = [tx,ty,tz]`：`x_norm_trans = (x_trans - mean_t) / std_t`。
    对应反归一化是相反方向：角度 `x_angle = x_norm_angle * std_angles + mean_angles`，
    平移 `x_trans = x_norm_trans * std_t + mean_t`。
    """
    if delta_bt6.ndim != 3 or int(delta_bt6.shape[-1]) < 6:
        raise ValueError(f"期望 delta_bt6 形状是 (B,T,6)，收到的是 {tuple(delta_bt6.shape)}")
    mean_a = stats_t["mean_angles"].view(1, 1, 3)
    std_a = stats_t["std_angles"].view(1, 1, 3)
    mean_t = stats_t["mean_t"].view(1, 1, 3)
    std_t = stats_t["std_t"].view(1, 1, 3)
    out = delta_bt6.clone()
    # 角度标签单独用角度统计量归一化，避免 radians 的尺度和 meters 的尺度混在一起。
    out[..., 0:3] = (out[..., 0:3] - mean_a) / std_a
    # 平移标签单独用平移统计量归一化，让 tx/ty/tz 保持自己的均值和标准差。
    out[..., 3:6] = (out[..., 3:6] - mean_t) / std_t
    return out


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """续训后把 optimizer state（优化器状态）里的 tensor 搬到当前训练 device。"""
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def build_tsformer() -> VisionTransformer:
    """构建 Stage B 动作头 TimesFormer，输出 18 维即 3 个 6D delta。"""
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


def _resolve_ckpt_state_dict(ckpt_obj):
    """兼容原始 state_dict 和包含 `model_state_dict` 的 checkpoint。"""
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_tsformer(model: nn.Module, ckpt_path: str):
    """加载 Stage B 初始化 TimesFormer 权重，strict=False 兼容不同 head/adapter 分支。"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _resolve_ckpt_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        rank0_print(f"[警告] TSFormer 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}")


def _load_adapter_state_dict(adapter_ckpt_path: str):
    """从 Stage A checkpoint 中取出 Adapter state_dict，兼容多种保存格式。"""
    try:
        obj = torch.load(adapter_ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(adapter_ckpt_path, map_location="cpu")

    if isinstance(obj, dict) and "adapter_state_dict" in obj:
        return obj["adapter_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and any(k.startswith("patch.") or k.startswith("proj.") for k in obj.keys()):
        return obj
    raise ValueError(f"不支持的 Adapter checkpoint 格式：{adapter_ckpt_path}")


def _add_infinitystar_to_syspath(inf_root: Optional[str], proj_root: str):
    """把 InfinityStar runtime 路径加入 sys.path，优先使用显式参数或环境变量。"""
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
    """加载 Stage B 使用的 InfinityStar VAE；可选在后期解冻部分 decoder 参数。"""
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


def traj_abs_to_delta(traj: np.ndarray, angles_in_degrees: bool, translation_divisor: float) -> np.ndarray:
    """
    traj: (T,6) = [x, y, z, roll, yaw, pitch]；angles_in_degrees=True 时角度单位是 degrees。
    返回 delta: (T,6)，其中 delta[t] 表示 (t-1 -> t) 的运动，delta[0]=0。
    每步布局为 [dz, dy, dx, tx, ty, tz]，ZYX Euler 角单位为 radians。
    """
    traj = np.asarray(traj, dtype=np.float32)
    T = int(traj.shape[0])
    out = np.zeros((T, 6), dtype=np.float32)
    if T <= 1:
        return out

    pos = traj[:, 0:3].copy() / float(translation_divisor)
    rpy = traj[:, 3:6].copy()
    if angles_in_degrees:
        rpy = rpy * (np.pi / 180.0)

    # 对角度（radians）做 unwrap，避免跨 +/-pi 时出现跳变。
    for i in range(3):
        rpy[:, i] = np.unwrap(rpy[:, i])

    Rs = []
    for t in range(T):
        # 本项目原始/preprocessed logs 使用 [roll, yaw, pitch]。
        roll, yaw, pitch = float(rpy[t, 0]), float(rpy[t, 1]), float(rpy[t, 2])
        R = np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)
        Rs.append(R)

    for t in range(1, T):
        R_prev = Rs[t - 1]
        R_cur = Rs[t]
        # 相对旋转公式：R_rel = R_prev^T @ R_cur，把“世界姿态变化”转成上一帧坐标系下的动作。
        R_rel = R_prev.T @ R_cur
        zyx = rotation_to_euler(R_rel, seq="zyx")  # 代码/形状说明：[z,y,x] radians
        out[t, 0:3] = np.asarray(zyx, dtype=np.float32)
        # 相对平移公式：p_rel = R_prev^T @ (p_cur - p_prev)，所以 tx/ty/tz 也在上一帧坐标系下。
        p_rel = R_prev.T @ (pos[t] - pos[t - 1])
        out[t, 3:6] = p_rel.astype(np.float32)
    return out


def delta_to_delta(
    delta_like: np.ndarray,
    *,
    translation_divisor: float,
    dyaw_unit: str = "auto",
) -> np.ndarray:
    """
    把类似 delta 的序列转换成规范的 (T,6) delta，单位为 (rad, meters)。

    支持的输入：
    - (T,6):  [dz,dy,dx, tx,ty,tz]，delta[0] 可能已经是 0
    - (T-1,6): 逐步 delta，会补上 delta[0]=0
    - (T,4) 或 (T-1,4): [tx,ty,tz, dyaw]（只有 yaw），roll/pitch 置 0

    注意：
    - 旋转布局始终是 [dz,dy,dx] = [dyaw, dpitch, droll]，单位 radians。
    - 平移会除以 translation_divisor（例如 100 表示 cm->m）；如果已经是 meters 就设为 1。
    - dyaw_unit: auto/deg/rad，其中 auto 使用幅值启发式判断。
    """
    x = np.asarray(delta_like, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"delta_like 必须是 2D 数组，收到的形状={x.shape}")
    Tm = int(x.shape[0])
    C = int(x.shape[1])
    if C not in (4, 6):
        raise ValueError(f"delta_like 必须有 4 列或 6 列，收到的形状={x.shape}")

    # 构建形状为 (T-1,6) 的逐步 deltas。
    if C == 6:
        step = x[:, :6].astype(np.float32)
    else:
        # 代码/形状说明：[tx,ty,tz, dyaw]
        step = np.zeros((Tm, 6), dtype=np.float32)
        step[:, 0] = x[:, 3]  # 中文说明：dz=dyaw
        step[:, 3:6] = x[:, 0:3]

    # 判断输入是带 delta[0]=0 的 (T,6)，还是 (T-1,6)。
    # 启发式：如果第一行全 0 或近似 0，就认为它已经是 (T,6)。
    if Tm >= 1 and float(np.max(np.abs(step[0]))) < 1e-8:
        delta = step
    else:
        delta = np.zeros((Tm + 1, 6), dtype=np.float32)
        delta[1:, :] = step

    # dyaw 单位转换。
    unit = str(dyaw_unit).lower().strip() if dyaw_unit else "auto"
    if unit not in ("auto", "deg", "rad"):
        raise ValueError(f"dyaw_unit 必须是 auto/deg/rad 之一，收到 {dyaw_unit!r}")
    if unit == "auto":
        dz = delta[1:, 0]
        p95 = float(np.nanpercentile(np.abs(dz), 95)) if dz.size else 0.0
        unit = "deg" if p95 > 1.0 else "rad"
    if unit == "deg":
        delta[:, 0] = delta[:, 0] * (np.pi / 180.0)

    # 平移单位转换。
    div = float(translation_divisor)
    if not np.isfinite(div) or div <= 0:
        raise ValueError(f"translation_divisor 必须是有限且大于 0 的数，收到 {translation_divisor}")
    if div != 1.0:
        delta[:, 3:6] = delta[:, 3:6] / div
    return delta.astype(np.float32)


def traj_to_delta(
    traj_or_delta: np.ndarray,
    *,
    traj_mode: str,
    angles_in_degrees: bool,
    translation_divisor: float,
    delta_dyaw_unit: str,
) -> np.ndarray:
    """
    把已加载的 trajectory json 内容转换成规范 delta (T,6)，单位为 (rad, meters)。
    """
    mode = str(traj_mode).lower().strip()
    if mode == "abs_pose":
        return traj_abs_to_delta(traj_or_delta, angles_in_degrees=angles_in_degrees, translation_divisor=translation_divisor)
    if mode == "delta":
        return delta_to_delta(traj_or_delta, translation_divisor=translation_divisor, dyaw_unit=delta_dyaw_unit)
    raise ValueError(f"未知 traj_mode：{traj_mode}，期望是 abs_pose 或 delta")


def compute_loss_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    rot_weight: float,
    trans_xy_weight: float,
    trans_z_weight: float,
    trans_vertical_index: int,
) -> torch.Tensor:
    """计算动作回归损失，并分别加权旋转、水平平移和垂直平移。"""
    b = int(target.shape[0])
    t = int(window_size - 1)
    pred = pred.view(b, t, 6)
    target = target.view(b, t, 6)
    pred_r, pred_t = pred[:, :, :3], pred[:, :, 3:]
    tgt_r, tgt_t = target[:, :, :3], target[:, :, 3:]

    loss = torch.zeros((), device=pred.device, dtype=torch.float32)
    if float(rot_weight) > 0:
        loss = loss + float(rot_weight) * F.mse_loss(pred_r, tgt_r)

    vi = int(trans_vertical_index)
    horiz = [0, 1, 2]
    horiz.remove(vi)
    loss_h = F.mse_loss(pred_t[:, :, horiz], tgt_t[:, :, horiz])
    loss_v = F.mse_loss(pred_t[:, :, vi], tgt_t[:, :, vi])
    loss = loss + float(trans_xy_weight) * loss_h + float(trans_z_weight) * loss_v
    return loss


def _linear_schedule(epoch: int, warmup_or_decay_epochs: int, start: float, end: float) -> float:
    """按 epoch 做线性预热/衰减，返回当前权重或学习率系数。"""
    if warmup_or_decay_epochs <= 0:
        return float(end)
    e = max(1, int(epoch))
    if warmup_or_decay_epochs == 1:
        return float(end)
    t = float(e - 1) / float(warmup_or_decay_epochs - 1)
    t = min(1.0, max(0.0, t))
    return float(start + t * (end - start))


def _sample_all_window_starts(B: int, T: int, window_size: int, stride: int, device: torch.device) -> torch.Tensor:
    """为 batch 中每条样本生成所有 TimesFormer 滑窗起点。"""
    max_start = int(T - window_size)
    if max_start < 0:
        raise ValueError(f"时间长度 T={T} 小于 window_size={window_size}")
    idx = torch.arange(0, max_start + 1, int(stride), device=device, dtype=torch.long)
    return idx.view(1, -1).repeat(B, 1).contiguous()


def _gather_window_tokens(tokens_btnd: torch.Tensor, starts_bk: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    按窗口起点从 `(B,T,N,D)` token 序列中取出 TimesFormer 输入。

    输入输出形状：
    - `tokens_btnd`: `(B,T,N,D)`，整条 clip 的逐帧 patch token；
    - `starts_bk`: `(B,K)`，每条样本的 K 个窗口起点；
    - 返回 `(B*K*window_size, N, D)`。

    中文说明：
    TimesFormer 的 `forward_features_from_patch_tokens()` 不直接接收
    `(B,K,window_size,N,D)`，而是要求先把所有窗口拍平成一长串 token，
    再通过传入的 `B`、`T` 参数恢复时空结构。
    """
    B, T, N, D = tokens_btnd.shape
    K = int(starts_bk.shape[1])
    flat = tokens_btnd.view(B, T, N * D)
    t_idx = torch.arange(window_size, device=starts_bk.device, dtype=torch.long).view(1, 1, window_size)
    idx = starts_bk.unsqueeze(-1) + t_idx  # 代码/形状说明：(B,K,window_size)
    idx2 = idx.view(B, K * window_size, 1).expand(B, K * window_size, N * D)
    g = flat.gather(1, idx2).view(B * K, window_size, N, D)
    return g.reshape(B * K * window_size, N, D).contiguous()


def _gather_window_targets(delta_bt6: torch.Tensor, starts_bk: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    按窗口起点收集 3 个相邻动作标签，并展平成 `(B*K,18)`。

    关键关系：
    4 帧窗口对应 3 个相邻帧转移，每个转移 6 维，因此目标维度固定为 `3 * 6 = 18`。
    """
    B, T, _ = delta_bt6.shape
    K = int(starts_bk.shape[1])
    t_idx = torch.arange(1, window_size, device=starts_bk.device, dtype=torch.long).view(1, 1, window_size - 1)
    idx = starts_bk.unsqueeze(-1) + t_idx
    idx2 = idx.view(B, K * (window_size - 1), 1).expand(B, K * (window_size - 1), 6)
    g = delta_bt6.gather(1, idx2).view(B * K, window_size - 1, 6)
    return g.reshape(B * K, (window_size - 1) * 6).contiguous()


def _decode_tokens_full_T(
    vae_decode_module,
    adapter,
    z_ext: torch.Tensor,
    allow_adapter_grad: bool,
    allow_vae_grad: bool,
    amp_enabled: bool,
    expected_T: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    返回：
    - `tokens`：形状 `(B,T,N,D)`；
    - `T`：decode 后实际得到的时间长度。

    中文导读：
    这里不关心最终 RGB，而是借助 VAE decoder hook 截取中间 feature，
    再通过 Adapter 把这些 feature 变成 TimesFormer token。

    梯度流说明：
    - `allow_adapter_grad=True`：Adapter 参与反向传播；
    - `allow_vae_grad=True`：VAE decoder 对应部分也接收梯度；
    - 两者都关：这里退化成冻结的特征提取器，只训练后面的动作头。
    """
    B = int(z_ext.shape[0])
    m = vae_decode_module.module if isinstance(vae_decode_module, DDP) else vae_decode_module
    vae = m.vae

    uses_batch_slicing = bool(getattr(vae, "use_slicing", False)) and B > 1

    def _run_decode(z_sub: torch.Tensor, expected_B: int) -> Tuple[torch.Tensor, int]:
        """执行一次 VAE decode，并通过 hook 捕获 feature 后转成 Adapter tokens。"""
        tokens_slices = []
        start = 0

        def hook(_module, _inp, out):
            """VAE decoder hook：将 `(B,96,t,H,W)` 中间特征送入 Adapter。"""
            nonlocal start, tokens_slices
            hs = out[0] if isinstance(out, (tuple, list)) else out  # 代码/形状说明：(B,96,t_slice,H,W)
            if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
                raise RuntimeError("up_block_3 hook 输出必须是 5D Tensor")
            Bh = int(hs.shape[0])
            if Bh != int(expected_B):
                raise RuntimeError(f"VAE hook 中 batch 维度不符合预期：hs={tuple(hs.shape)} expected_B={expected_B}")
            t_slice = int(hs.shape[2])
            ctx = torch.enable_grad() if allow_adapter_grad else torch.no_grad()
            with ctx:
                tok, _t2, _w2 = adapter(hs)  # 代码/形状说明：(Bh*t_slice,N,D)
                tok = tok.view(Bh, t_slice, tok.shape[1], tok.shape[2]).contiguous()
            tokens_slices.append(tok)
            start += t_slice

        handle = m.vae.decoder.up_blocks[-1].register_forward_hook(hook)
        try:
            grad_ctx = contextlib.nullcontext() if allow_vae_grad else torch.no_grad()
            amp_ctx = autocast(enabled=bool(amp_enabled) and torch.cuda.is_available())
            try:
                with grad_ctx, amp_ctx:
                    _ = vae_decode_module(z_sub)
            except RuntimeError as e:
                msg = str(e)
                if "torch.cat(): expected a non-empty list of Tensors" in msg:
                    rank0_print(f"[警告] VAE decode 得到空 dec 列表，跳过该片段。z_sub={tuple(z_sub.shape)} err={msg}")
                    empty = torch.empty((int(expected_B), 0, 480, 384), device=z_sub.device, dtype=torch.float32)
                    return empty, 0
                raise
        finally:
            handle.remove()

        T = int(start)
        if len(tokens_slices) == 0 or T <= 0:
            empty = torch.empty((int(expected_B), 0, 480, 384), device=z_sub.device, dtype=torch.float32)
            return empty, 0
        out = torch.cat(tokens_slices, dim=1)
        return out, T

    if uses_batch_slicing:
        outs = []
        Ts = []
        for b_idx in range(B):
            out_b, T_b = _run_decode(z_ext[b_idx : b_idx + 1], expected_B=1)
            outs.append(out_b)
            Ts.append(int(T_b))
        if len(set(Ts)) != 1 and max(Ts) > 0:
            rank0_print(f"[警告] VAE slicing 后 batch 内各样本解码帧数 T 不一致：{Ts}")
        out = torch.cat(outs, dim=0)
        T = int(out.shape[1])
    else:
        out, T = _run_decode(z_ext, expected_B=B)

    if expected_T is not None and int(T) != int(expected_T):
        rank0_print(f"[警告] 解码得到 T={T}，但轨迹标签期望 expected_T={expected_T}")
    return out, int(T)


def collate_fn(samples, mode: str = "crop"):
    """
    Stage B batch 组装函数：对齐 latent、轨迹标签和 meta，支持整样本或裁剪模式。

    两种模式：
    - `per_sample`：不裁剪、不 padding。训练循环里逐条样本处理，最保真；
    - `crop`：裁到 batch 内最短长度后堆叠，速度更快，但会丢掉长轨迹尾部。
    """
    mode = str(mode).strip().lower()
    if mode == "per_sample":
        out_samples = []
        for s in samples:
            z = s["z_ext"]
            if not isinstance(z, torch.Tensor):
                z = torch.as_tensor(z)
            z = z.float().contiguous()

            t = s["traj"]
            if isinstance(t, torch.Tensor):
                tt = t.float().contiguous()
            else:
                tt = torch.from_numpy(np.asarray(t, dtype=np.float32)).contiguous()
            out_samples.append({"z_ext": z, "traj": tt, "meta": s.get("meta", {})})
        return {"samples": out_samples, "meta": [s.get("meta", {}) for s in samples]}

    # crop 模式：先裁到 batch 内最短长度，再堆叠。
    z_list = []
    t_lat_list = []
    for s in samples:
        z = s["z_ext"]
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        z = z.float().contiguous()
        z_list.append(z)
        t_lat_list.append(int(z.shape[2]))
    t_lat_min = int(min(t_lat_list)) if t_lat_list else 0
    if t_lat_min <= 0:
        raise ValueError(f"batch 中 latent 时间长度非法：{t_lat_list}")

    traj_list = []
    t_traj_list = []
    for s in samples:
        t = s["traj"]
        if isinstance(t, torch.Tensor):
            tt = t.float().contiguous()
        else:
            tt = torch.from_numpy(np.asarray(t, dtype=np.float32)).contiguous()
        traj_list.append(tt)
        t_traj_list.append(int(tt.shape[0]))
    t_traj_min = int(min(t_traj_list)) if t_traj_list else 0
    if t_traj_min <= 0:
        raise ValueError(f"batch 中轨迹时间长度非法：{t_traj_list}")

    z_ext = torch.cat([z[:, :, :t_lat_min].contiguous() for z in z_list], dim=0).contiguous()
    traj = torch.stack([t[:t_traj_min].contiguous() for t in traj_list], dim=0).contiguous()
    meta = [s.get("meta", {}) for s in samples]
    return {"z_ext": z_ext, "traj": traj, "meta": meta}


def main():
    """
    Stage B 训练入口。

    中文导读：
    这个阶段把 Stage A 对齐好的 Adapter 接到 TimesFormer 动作头上，训练目标从
    token distillation 变成动作监督：
    主数据流：`z_ext -> VAE decoder feature -> Adapter tokens -> TimesFormer -> delta action`。
    读代码时先看 label_stats、adapter checkpoint、trajectory delta 转换，再看训练循环。
    """
    ap = argparse.ArgumentParser(description="Stage B latent-to-action 训练入口：把 VAE/Adapter 提取的 token 送入 TimesFormer，监督学习逐步 delta action。")
    ap.add_argument("--manifest_json", type=str, required=True, help="训练样本 manifest 路径；其中每条样本应指向 latent、trajectory 和可选图像目录。")
    ap.add_argument("--items_key", type=str, default="ALL", help="要使用的 manifest key；可传逗号分隔的多个 key，或用 ALL 表示读取所有 items_* 列表。")
    ap.add_argument(
        "--workspace_root",
        type=str,
        default="",
        help=(
            "解析 manifest_json 内相对路径时使用的根目录。"
            "如果留空，默认使用当前开源包根目录（即包含 Worldmodel/ 和 train/ 的目录）。"
            "当 manifest 里的路径是相对于其它目录编写时，需要显式设置它。"
        ),
    )
    ap.add_argument("--max_items", type=int, default=0, help="最多读取多少条样本；0 表示不限制，适合完整训练。")
    # 匹配旧版 stage2 launcher 默认值：不强制固定 T。
    ap.add_argument("--require_T", type=int, default=0, help="若 >0，则只保留轨迹长度等于该值的样本；0 表示不强制固定长度。")

    ap.add_argument("--tsformer_pretrained", type=str, required=True, help="Stage B 初始化用的 TimesFormer checkpoint，通常来自参考动作头或已有 Stage B 训练。")
    ap.add_argument("--adapter_ckpt", type=str, required=True, help="Stage A 训练出的 Adapter checkpoint，用于初始化 VAE feature -> TimesFormer token 桥。")
    ap.add_argument("--resume", type=str, default="", help="可选续训 checkpoint；留空表示从 `--tsformer_pretrained` 和 `--adapter_ckpt` 初始化新训练。")
    ap.add_argument(
        "--label_stats_json",
        type=str,
        default="",
        help="可选：指定 TSFormer 训练时保存的 run_config.json，或只包含 label_stats 的 json。用于把 GT 动作标签归一化到 checkpoint 期望的输出分布。",
    )

    ap.add_argument("--infinitystar_vae_path", type=str, required=True, help="InfinityStar VAE checkpoint 路径；Stage B 通过 VAE decoder hook 提取中间 feature。")
    ap.add_argument("--infinitystar_vae_type", type=int, default=64, help="VAE 类型编号，必须和 checkpoint 对应结构一致。")
    ap.add_argument("--infinitystar_root", type=str, default="", help="InfinityStar 代码根目录；留空时按仓库默认位置推断。")

    # 这些参数必须和 VAE checkpoint 的结构一致。
    ap.add_argument("--semantic_scale_dim", type=int, default=16, help="VAE 语义尺度 embedding 维度，必须和 VAE checkpoint 配置一致。")
    ap.add_argument("--detail_scale_dim", type=int, default=64, help="VAE 细节尺度 embedding 维度，必须和 VAE checkpoint 配置一致。")
    ap.add_argument("--use_learnable_dim_proj", type=int, default=0, help="VAE 是否使用可学习维度投影；必须和 checkpoint 配置一致。")
    ap.add_argument("--detail_scale_min_tokens", type=int, default=350, help="VAE 细节尺度最少 token 数；必须和 checkpoint 配置一致。")
    ap.add_argument("--use_feat_proj", type=int, default=2, help="VAE 特征投影模式；必须和 checkpoint 配置一致。")
    ap.add_argument("--semantic_scales", type=int, default=11, help="VAE 语义尺度数量；必须和 checkpoint 配置一致。")

    ap.add_argument("--out_dir", type=str, required=True, help="输出目录，用于保存 Stage B checkpoint、组合导出文件和训练日志。")
    # 匹配旧版 stage2 launcher 默认值。
    ap.add_argument("--epochs", type=int, default=60, help="训练 epoch 数。")
    ap.add_argument("--global_batch_size", type=int, default=8, help="全局 batch size；若 >0，会按 world_size 自动换算单卡 batch_size。")
    ap.add_argument("--batch_size", type=int, default=1, help="单卡 batch size；当 global_batch_size>0 时会被覆盖。")
    ap.add_argument("--window_stride", type=int, default=1, help="4 帧 TimesFormer 滑窗的起点步长；1 表示覆盖所有相邻窗口。")
    ap.add_argument("--windows_chunk", type=int, default=24, help="每次送入 TimesFormer 的窗口块大小；越小越省显存但更慢。")

    ap.add_argument("--num_workers", type=int, default=16, help="DataLoader worker 数量。")
    ap.add_argument("--persistent_workers", action="store_true", default=True, help="是否让 DataLoader workers 跨 epoch 保持常驻，减少反复创建进程的开销。")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="每个 DataLoader worker 预取多少个 batch。")
    ap.add_argument("--collate_mode", type=str, default="per_sample", choices=["crop", "per_sample"], help="batch 组装方式：crop 会裁到最短序列；per_sample 会逐样本处理变长序列。")

    ap.add_argument("--lr", type=float, default=1e-5, help="TimesFormer backbone 的基础学习率。")
    ap.add_argument("--head_lr_mult", type=float, default=10.0, help="动作 head 相对基础学习率的倍率。")
    ap.add_argument("--adapter_lr_mult", type=float, default=2.0, help="Adapter 相对基础学习率的倍率。")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 权重衰减。")
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=3,
        help="梯度累积步数。每 N 次 DataLoader 迭代才执行一次 optimizer.step()；单个样本内部不会再做 microbatch。",
    )
    ap.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪的最大范数；设为 0 表示关闭。")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子；实际每个 rank 会加上 rank id。")
    ap.add_argument("--save_every", type=int, default=2, help="每隔多少个 epoch 保存一次 checkpoint。")
    ap.add_argument("--log_every", type=int, default=20, help="每隔多少个 step 打印一次训练日志。")
    ap.add_argument("--tqdm", action="store_true", default=False, help="是否在 rank0 显示 tqdm 进度条。")
    ap.add_argument("--log_file", type=str, default="train.log", help="rank0 文件日志名；留空则不写文件日志。")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备，例如 cuda 或 cpu。")
    ap.add_argument("--amp", action="store_true", default=True, help="是否启用 AMP 混合精度。")

    # loss 配置。
    ap.add_argument("--translation_divisor", type=float, default=100.0, help="把输入轨迹平移量除以该值后转成 meters；默认把 cm 转成 m。")
    ap.add_argument("--angles_in_degrees", action="store_true", default=True, help="输入绝对位姿角度是否是 degree；开启后会转成 radians 训练。")
    ap.add_argument(
        "--traj_mode",
        type=str,
        default="abs_pose",
        choices=["abs_pose", "delta"],
        help="如何解释 traj_json_path。abs_pose 表示 `(T,6)` 的累计绝对位姿，需要先转成逐步 delta；delta 表示文件本身就是逐步动作，可为 `(T-1,6)/(T,6)` 或只含 yaw 的 `(T-1,4)/(T,4)`。",
    )
    ap.add_argument(
        "--delta_dyaw_unit",
        type=str,
        default="auto",
        choices=["auto", "deg", "rad"],
        help="当 traj_mode=delta 时，如何解释 dz(dyaw) 的单位。auto 会用数值范围做启发式判断。",
    )
    ap.add_argument("--rot_loss_weight_start", type=float, default=0.05, help="旋转 loss 初始权重。")
    ap.add_argument("--rot_loss_weight_max", type=float, default=0.25, help="旋转 loss 预热后的最大权重。")
    ap.add_argument("--rot_warmup_epochs", type=int, default=15, help="旋转 loss 从 start 增长到 max 的 epoch 数。")
    ap.add_argument("--trans_xy_weight", type=float, default=1.0, help="水平平移 x/y loss 权重。")
    ap.add_argument("--trans_z_weight", type=float, default=2.0, help="垂直平移 z loss 默认权重。")
    ap.add_argument("--trans_z_weight_start", type=float, default=None, help="可选：垂直平移 z loss 的起始权重。")
    ap.add_argument("--trans_z_weight_end", type=float, default=None, help="可选：垂直平移 z loss 的结束权重。")
    ap.add_argument("--trans_z_decay_epochs", type=int, default=12, help="垂直平移 z loss 权重变化持续的 epoch 数。")
    ap.add_argument("--trans_vertical_index", type=int, default=2, help="动作 6 维中哪个平移分量视为垂直方向，默认 2 表示 tz。")

    # 训练开关。
    ap.add_argument("--train_adapter", action="store_true", default=True, help="是否训练 Adapter；关闭时 Adapter 仅作为冻结特征桥。")
    ap.add_argument("--freeze_adapter_epochs", type=int, default=6, help="前 N 个 epoch 冻结 Adapter，让 TimesFormer 先适应它输出的 token 分布。")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0, help="前 N 个 epoch 冻结 TimesFormer backbone，只训练 head/其它可训练模块。")
    ap.add_argument("--train_vae_after_epoch", type=int, default=20, help="从哪个 epoch 之后开始允许训练 VAE decoder；默认晚期才解冻。")
    ap.add_argument("--vae_lr_mult", type=float, default=0.0, help="VAE 学习率相对基础 lr 的倍率；0 表示即使到解冻阶段也不训练 VAE。")
    ap.add_argument("--vae_disable_slicing", action="store_true", default=False, help="禁用 VAE slicing；更吃显存，但时间维 hook 更直观。")
    ap.add_argument("--vae_disable_tiling", action="store_true", default=False, help="禁用 VAE tiling；通常用于排查边界或兼容问题。")
    ap.add_argument("--vae_num_sample_frames_batch_size", type=int, default=0, help="覆盖 VAE 内部一次 decode 的帧批大小；0 表示使用模型默认值。")

    # 导出配置。
    ap.add_argument("--export_combined", action="store_true", default=True, help="训练后是否导出组合 checkpoint，便于推理脚本直接加载。")
    ap.add_argument("--export_name", type=str, default="stage2_latent2action_combined.pt", help="组合 checkpoint 文件名。")

    args = ap.parse_args()

    local_rank = ddp_setup()
    device = torch.device(str(args.device))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 需要 torch.cuda.is_available() 为 True，但当前不可用")
        device = torch.device(f"cuda:{local_rank}")

    seed_everything(int(args.seed) + get_rank())
    if not str(args.out_dir).strip():
        raise ValueError("--out_dir 不能为空")
    os.makedirs(str(args.out_dir), exist_ok=True)

    global _RANK0_LOG_FH
    if get_rank() == 0:
        lf = str(args.log_file).strip()
        if lf:
            _RANK0_LOG_FH = open(os.path.join(str(args.out_dir), lf), "a", encoding="utf-8")
            rank0_print(f"[日志] rank0 日志写入 {os.path.join(str(args.out_dir), lf)}")
        try:
            args_dump = json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            args_dump = str(vars(args))
        rank0_print(
            "[配置]"
            f" rank={get_rank()}"
            f" world_size={get_world_size()}"
            f" device={device}"
            f" cuda={torch.cuda.is_available()}"
            f" amp={bool(args.amp)}"
        )
        rank0_print("[配置] 训练参数 args:\n" + args_dump)

    # 全局 batch size 覆盖。
    if int(args.global_batch_size) > 0:
        ws = int(get_world_size())
        if int(args.global_batch_size) % ws != 0:
            raise ValueError(f"global_batch_size={args.global_batch_size} 必须能被 world_size={ws} 整除")
        args.batch_size = int(args.global_batch_size) // ws
        if int(args.batch_size) < 1:
            raise ValueError("计算得到的单卡 batch_size 小于 1")
        rank0_print(f"[batch] global_batch_size={args.global_batch_size} world_size={ws} per_gpu_batch={args.batch_size}")

    # TimesFormer 动作头：
    # 初始化自参考视频/VO checkpoint，后续用 Adapter 产生的 patch tokens 代替 RGB patch_embed。
    tsformer = build_tsformer().to(device)
    load_tsformer(tsformer, str(args.tsformer_pretrained))

    # 标签归一化必须和 checkpoint head 的输出分布一致。
    # 若 checkpoint 来自归一化标签训练，但这里忘记加载 label_stats，动作尺度会明显错位。
    label_stats_np, label_stats_src = _try_load_label_stats(
        label_stats_json=str(args.label_stats_json),
        tsformer_pretrained=str(args.tsformer_pretrained),
    )
    label_stats_t = None
    if label_stats_np is not None:
        label_stats_t = {
            k: torch.from_numpy(v).to(device=device, dtype=torch.float32)
            for k, v in label_stats_np.items()
        }
        rank0_print(f"[配置] label_stats 来源：{label_stats_src}")
    else:
        label_stats_src = ""
        rank0_print(
            "[警告] 未找到 label_stats；将直接在未归一化的目标空间（rad + m）训练。"
            "如果你的 TSFormer checkpoint 是按归一化标签训练得到的，请显式传入 --label_stats_json，"
            "或把 run_config.json 放在 --tsformer_pretrained 旁边。"
        )

    # Adapter 从 Stage A 蒸馏结果初始化。这里可以继续训练它，
    # 但它的初始职责是保持 VAE feature -> TimesFormer token 的对齐关系。
    adapter = Vae96ToTSformerEmbedAdapter().to(device)
    adapter_sd = _load_adapter_state_dict(str(args.adapter_ckpt))
    missing, unexpected = adapter.load_state_dict(adapter_sd, strict=False)
    if len(missing) or len(unexpected):
        rank0_print(f"[警告] Adapter 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}")
    adapter.requires_grad_(bool(args.train_adapter))
    adapter.eval()

    # VAE decoder 为 Adapter 提供中间 feature map。
    # Stage B 可以在训练后期按需解冻部分 VAE decoder 组件。
    inf_root = str(args.infinitystar_root).strip() or None
    vae_model = load_infinitystar_vae(
        vae_path=str(args.infinitystar_vae_path),
        vae_type=int(args.infinitystar_vae_type),
        device=device,
        infinitystar_root=inf_root,
        proj_root=_PROJ_ROOT,
        semantic_scale_dim=int(args.semantic_scale_dim),
        detail_scale_dim=int(args.detail_scale_dim),
        use_learnable_dim_proj=int(args.use_learnable_dim_proj),
        detail_scale_min_tokens=int(args.detail_scale_min_tokens),
        use_feat_proj=int(args.use_feat_proj),
        semantic_scales=int(args.semantic_scales),
    )

    if bool(args.vae_disable_slicing) and hasattr(vae_model, "disable_slicing"):
        try:
            vae_model.disable_slicing()
            rank0_print("[配置] vae.disable_slicing()")
        except Exception as e:
            rank0_print(f"[警告] 调用 vae.disable_slicing() 失败：{e}")
    if bool(args.vae_disable_tiling) and hasattr(vae_model, "disable_tiling"):
        try:
            vae_model.disable_tiling()
            rank0_print("[配置] vae.disable_tiling()")
        except Exception as e:
            rank0_print(f"[警告] 调用 vae.disable_tiling() 失败：{e}")
    if int(args.vae_num_sample_frames_batch_size) > 0 and hasattr(vae_model, "num_sample_frames_batch_size"):
        try:
            vae_model.num_sample_frames_batch_size = int(args.vae_num_sample_frames_batch_size)
            rank0_print(f"[配置] 设置 vae.num_sample_frames_batch_size={int(args.vae_num_sample_frames_batch_size)}")
        except Exception as e:
            rank0_print(f"[警告] 设置 vae.num_sample_frames_batch_size 失败：{e}")

    will_train_vae = bool(int(args.train_vae_after_epoch) < int(args.epochs))

    class _VaeDecodeOnly(nn.Module):
        """只暴露 VAE decode 的薄封装，便于在训练里单独冻结/解冻。"""

        def __init__(self, vae):
            """保存底层 VAE 模型。"""
            super().__init__()
            self.vae = vae

        def forward(self, z_ext: torch.Tensor) -> torch.Tensor:
            """触发 decode；真正的训练输入来自 hook 捕获的中间特征。"""
            return self.vae.decode(z_ext, return_dict=False)[0]

    vae_decode = _VaeDecodeOnly(vae_model)

    def _vae_set_trainable(trainable: bool):
        """统一切换 VAE 中需要训练的子模块参数开关。"""
        for name in ("proj_up", "post_quant_conv", "decoder", "scale_learnable_parameters"):
            sub = getattr(vae_model, name, None)
            if sub is None:
                continue
            try:
                for p in sub.parameters():
                    p.requires_grad_(bool(trainable))
            except Exception:
                # scale_learnable_parameters 可能不是 module。
                pass

    if is_dist() and will_train_vae:
        # DDP 构造时至少需要一个可训练参数。
        _vae_set_trainable(True)
        vae_decode = DDP(vae_decode, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
        _vae_set_trainable(False)

    # Optimizer 参数组：
    # - backbone_params: TimesFormer backbone，使用保守 LR；
    # - head_params: action regression head，通常使用更大 LR；
    # - adapter params: 可选地继续做对齐训练；
    # - VAE params: 先以 LR=0 放入 optimizer，直到配置的解冻 epoch。
    backbone_params = []
    head_params = []
    for n, p in tsformer.named_parameters():
        if n.startswith("head."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    params = [{"params": backbone_params, "lr": float(args.lr)}, {"params": head_params, "lr": float(args.lr) * float(args.head_lr_mult)}]
    if bool(args.train_adapter):
        params.append(
            {"params": [p for p in adapter.parameters() if p.requires_grad], "lr": float(args.lr) * float(args.adapter_lr_mult)}
        )

    m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
    vae_params = [p for p in m_vae.vae.parameters()]
    vae_pg_idx = len(params)
    params.append({"params": vae_params, "lr": 0.0})

    optimizer = torch.optim.AdamW(params, weight_decay=float(args.weight_decay), betas=(0.9, 0.95))
    scaler = GradScaler(enabled=bool(args.amp) and torch.cuda.is_available())

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    resume_path = str(args.resume).strip()
    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(resume_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            tsformer.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "adapter_state_dict" in ckpt:
                adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
            if "vae_state_dict" in ckpt:
                try:
                    m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                    missing, unexpected = m_vae.vae.load_state_dict(ckpt["vae_state_dict"], strict=False)
                    if len(missing) or len(unexpected):
                        rank0_print(
                            f"[警告] 续训 VAE 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}"
                        )
                except Exception as e:
                    rank0_print(f"[警告] 续训 VAE 状态加载失败：{e}")
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _optimizer_state_to_device(optimizer, device)
            if "scaler_state_dict" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            best_val = float(ckpt.get("best_val", best_val))
            rank0_print(f"[续训] 从 {resume_path} 恢复：epoch={start_epoch} step={global_step}")

    # 注意：这里故意不把 tsformer 或 adapter 包进 DDP。
    # TSformer forward 使用 .forward_features_from_patch_tokens()，会绕过 DDP forward hooks，
    # 导致梯度同步静默失效。
    # Adapter 在每个 step 中可能多次 backward（per_sample 模式下每个样本一次），
    # 如果交给 DDP 会触发错误的重复 all-reduce。
    # 因此在 optimizer.step() 前通过 _allreduce_grads() 手动 all-reduce 所有可训练梯度，
    # 不管单个 optimizer step 里有多少次 backward，都能保证梯度平均正确。
    # 这也是本文件最容易看混的一段：TimesFormer 的“窗口级前向”与 Adapter/VAE 的
    # “整段 token 级反向”被拆成了两段，再在 step 前统一做梯度同步。
    #
    # ------------------------------------------------------------------
    # 通俗版（“班长收作业”的比喻）：
    # ------------------------------------------------------------------
    # DDP 就像一个自动批改的“班长”，它的同步机制依赖两个固定假设：
    #   1) 班长只在大门口收作业：DDP 的梯度同步 hook 注册在 model.forward() 上，
    #      每次调用 model(x)，hook 才知道“这次前向用了哪些参数”，反向时才会
    #      把多卡上对应参数的梯度做 all-reduce 平均。
    #   2) 一次发卷、一次收卷：DDP 默认一个 step 里 forward 一次、backward 一次。
    #
    # 但本文件两条都违反了：
    #
    # 问题 1：TSformer 走了“后门”
    #   TSformer 的前向不是 tsformer(x)，而是 m.forward_features_from_patch_tokens(...)
    #   （见下方训练循环）。相当于学生绕过班长的桌子，从后窗爬进教室做题：
    #     - 题做了（梯度算了）
    #     - 班长没看见（DDP forward hook 没触发）
    #     - 结果：N 张卡各练各的，参数会慢慢漂移成 N 个不同模型
    #   最坑的是：它不会报错。loss 也在下降，但多卡间的模型其实已经不一致。
    #   这就是上面说的“梯度同步静默失效”。
    #
    # 问题 2：Adapter 一节课交了好几次作业
    #   per_sample 模式下，一个 DataLoader batch 可能装着多条不等长的轨迹，
    #   循环里逐条样本独立跑 VAE -> Adapter -> TSformer -> backward。
    #   也就是说一个 optimizer step 内会调用多次 backward。
    #   如果把 Adapter 交给 DDP：
    #     - 第 1 次 backward 触发一次 all-reduce
    #     - 第 2 次 backward 又触发一次 all-reduce（把已经平均过的梯度再平均，
    #       或者直接报 "Expected to mark a variable ready only once"）
    #     - 第 3 次……同上
    #   梯度被反复平均，等价于学习率被悄悄缩成原来的 1/N。
    #
    # 解法：自己当班长
    #   干脆不让 DDP 管 tsformer 和 adapter，改成：
    #   “下课铃响之前（optimizer.step() 之前），不管你今天做了几道题（多少次 backward），
    #    把每个人草稿纸上累积的答案（p.grad）拿出来，统一对一次答案就完事了。”
    #   对应下面 _allreduce_grads(optimizer.param_groups) 这一行：
    #     - 手动对每个可训练参数的 p.grad 做一次 all_reduce(SUM) + 除以 world_size
    #     - 不依赖 forward 路径，不依赖 backward 次数
    #     - 一个 step 同步一次，干净
    #
    # 为什么说“最容易看混”：因为前向和反向被故意拆成了两段
    #   - 窗口级前向/反向：TSformer 对若干个 4 帧窗口跑 forward+backward，
    #     梯度只打到 patch_tokens 这个“中间缓存”上。
    #   - 整段 token 级反向：把窗口梯度按时间轴 scatter_add 回完整 token 序列，
    #     再一次性 tokens_btnd.backward(grad_tokens) 把梯度灌回 Adapter/VAE。
    #   - step 级同步：所有累积梯度在 optimizer.step() 前统一 all-reduce 一次。
    #   所以读代码时容易以为“VAE/Adapter 没收到梯度”，实际上是先攒着，最后一把灌回去。
    #   配合手动 all-reduce，效果等价于一次普通 DDP 训练，但避开了上面两个雷。

    # 仓库根目录（包含 Worldmodel/、infer/、train/ 等目录）。
    vln_uav_root = os.path.abspath(os.path.join(_PROJ_ROOT, "..", "..", ".."))
    ws_root = str(args.workspace_root).strip() or vln_uav_root
    max_items = int(args.max_items) if int(args.max_items) > 0 else None
    ds = LatentTrajManifestDataset(
        manifest_json=str(args.manifest_json),
        items_key=str(args.items_key),
        workspace_root=ws_root,
        transform=None,
        load_frames=False,
        max_items=max_items,
        require_T=int(args.require_T) if int(args.require_T) > 0 else None,
    )
    sampler = DistributedSampler(ds, shuffle=True, drop_last=False) if is_dist() else None
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=partial(collate_fn, mode=str(args.collate_mode)),
        persistent_workers=bool(args.persistent_workers) if int(args.num_workers) > 0 else False,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
    )
    if get_rank() == 0:
        rank0_print(
            "[配置]"
            f" dataset_len={len(ds)}"
            f" batches_per_epoch={len(dl)}"
            f" batch_size_per_rank={int(args.batch_size)}"
            f" global_batch_size={int(args.batch_size) * get_world_size()}"
            f" num_workers={int(args.num_workers)}"
        )

    window_size = 4
    W_grid = 40

    for epoch in range(start_epoch, int(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)

        rot_w = _linear_schedule(
            epoch=epoch + 1,
            warmup_or_decay_epochs=int(args.rot_warmup_epochs),
            start=float(args.rot_loss_weight_start),
            end=float(args.rot_loss_weight_max),
        )
        if args.trans_z_weight_start is not None and args.trans_z_weight_end is not None:
            trans_z_w = _linear_schedule(
                epoch=epoch + 1,
                warmup_or_decay_epochs=int(args.trans_z_decay_epochs),
                start=float(args.trans_z_weight_start),
                end=float(args.trans_z_weight_end),
            )
        else:
            trans_z_w = float(args.trans_z_weight)

        train_vae_now = (epoch + 1) > int(args.train_vae_after_epoch)
        _vae_set_trainable(train_vae_now)
        optimizer.param_groups[vae_pg_idx]["lr"] = float(args.lr) * float(args.vae_lr_mult) if train_vae_now else 0.0

        freeze_backbone = int(args.freeze_backbone_epochs) > 0 and (epoch < int(args.freeze_backbone_epochs))
        if freeze_backbone:
            for n, p in tsformer.named_parameters():
                p.requires_grad = n.startswith("head.")
        else:
            for p in tsformer.parameters():
                p.requires_grad = True

        freeze_adapter_now = int(args.freeze_adapter_epochs) > 0 and (epoch < int(args.freeze_adapter_epochs))
        adapter_trainable_now = bool(args.train_adapter) and (not freeze_adapter_now)
        adapter.requires_grad_(adapter_trainable_now)

        tsformer.train()
        if adapter_trainable_now:
            adapter.train()
        else:
            adapter.eval()

        if get_rank() == 0 and epoch == 0:
            rank0_print(f"[阶段] freeze_adapter={freeze_adapter_now} adapter_trainable={adapter_trainable_now} freeze_backbone={freeze_backbone} train_vae={train_vae_now}")
        if get_rank() == 0 and (freeze_adapter_now != (int(args.freeze_adapter_epochs) > 0 and (epoch - 1 < int(args.freeze_adapter_epochs)))):
            rank0_print(f"[阶段] epoch={epoch+1} 正在解冻 adapter：freeze_adapter={freeze_adapter_now} adapter_trainable={adapter_trainable_now}")

        running = torch.zeros((), device=device)
        nb = 0
        t0 = time.time()
        use_tqdm = bool(args.tqdm) and (get_rank() == 0) and (tqdm is not None)
        pbar = tqdm(total=len(dl), desc=f"stage2 第 {epoch+1}/{int(args.epochs)} 个 epoch", dynamic_ncols=True, leave=True) if use_tqdm else None

        t_after_step = time.time()
        grad_accum_steps = max(1, int(args.grad_accum_steps))
        accum_i = 0
        for batch in dl:
            nb += 1
            global_step += 1
            data_s = time.time() - t_after_step
            t_step0 = time.time()
            if accum_i == 0:
                optimizer.zero_grad(set_to_none=True)
            m = tsformer
            accum_scale = 1.0 / float(grad_accum_steps)

            if isinstance(batch, dict) and "samples" in batch:
                samples = batch["samples"]
                denom = float(max(1, len(samples)))
                used = 0
                loss_sum = torch.zeros((), device=device, dtype=torch.float32)

                for s in samples:
                    # per_sample 模式：一个 DataLoader batch 里可能有多条不等长轨迹，
                    # 这里逐条处理，避免为了堆叠而裁剪或填充。
                    z_ext = s["z_ext"].to(device, non_blocking=True)  # (1,64,T_lat,16,16)
                    traj_t = s["traj"]
                    traj_np = traj_t.detach().cpu().numpy() if isinstance(traj_t, torch.Tensor) else np.asarray(traj_t, dtype=np.float32)
                    if int(traj_np.shape[0]) < window_size:
                        continue

                    d = traj_to_delta(
                        traj_np,
                        traj_mode=str(args.traj_mode),
                        angles_in_degrees=bool(args.angles_in_degrees),
                        translation_divisor=float(args.translation_divisor),
                        delta_dyaw_unit=str(args.delta_dyaw_unit),
                    )
                    delta_bt6 = torch.from_numpy(d).unsqueeze(0).to(device, non_blocking=True)  # (1,T,6) rad+m
                    if label_stats_t is not None:
                        delta_bt6 = _normalize_delta_bt6(delta_bt6, label_stats_t)
                    T_traj = int(delta_bt6.shape[1])

                    allow_adapter_grad = adapter_trainable_now
                    tokens_btnd, T_dec = _decode_tokens_full_T(
                        vae_decode_module=vae_decode,
                        adapter=adapter,
                        z_ext=z_ext,
                        allow_adapter_grad=allow_adapter_grad,
                        allow_vae_grad=bool(train_vae_now),
                        amp_enabled=bool(args.amp),
                        expected_T=T_traj,
                    )
                    T_use = min(int(T_dec), int(T_traj))
                    if T_use < window_size:
                        continue
                    tokens_btnd = tokens_btnd[:, :T_use]
                    delta_bt6 = delta_bt6[:, :T_use]

                    starts_bk_all = _sample_all_window_starts(
                        B=1, T=T_use, window_size=window_size, stride=int(args.window_stride), device=device
                    )
                    K = int(starts_bk_all.shape[1])
                    if K <= 0:
                        continue

                    chunk = int(args.windows_chunk)
                    if chunk <= 0 or chunk >= K:
                        chunk = K

                    total_loss = torch.zeros((), device=device, dtype=torch.float32)
                    tokens_det = tokens_btnd.detach()
                    grad_tokens = torch.zeros_like(tokens_det, dtype=torch.float32)
                    done = 0
                    for s0 in range(0, K, chunk):
                        # 再把一个样本内部的所有滑窗切成小块，是为了控制显存：
                        # token 先 detach 成“中间特征缓存”，TimesFormer 只对当前 chunk 回传梯度。
                        s1 = min(K, s0 + chunk)
                        starts_bk = starts_bk_all[:, s0:s1]
                        kc = int(starts_bk.shape[1])
                        Bwin = int(kc)

                        patch_tokens_val = _gather_window_tokens(tokens_det, starts_bk, window_size=window_size)
                        patch_tokens = patch_tokens_val.detach().requires_grad_(True)
                        targets = _gather_window_targets(delta_bt6, starts_bk, window_size=window_size)

                        with autocast(enabled=bool(args.amp) and torch.cuda.is_available()):
                            feat = m.forward_features_from_patch_tokens(patch_tokens, B=Bwin, T=window_size, W=W_grid)
                            pred = m.head(feat)
                            loss_chunk = compute_loss_mse(
                                pred=pred,
                                target=targets,
                                window_size=window_size,
                                rot_weight=float(rot_w),
                                trans_xy_weight=float(args.trans_xy_weight),
                                trans_z_weight=float(trans_z_w),
                                trans_vertical_index=int(args.trans_vertical_index),
                            )
                        w = float(kc) / float(K)
                        scaler.scale(((loss_chunk * w) / denom) * accum_scale).backward()
                        g_patch = patch_tokens.grad.detach()
                        g_patch = g_patch.view(1, kc, window_size, g_patch.shape[1], g_patch.shape[2])
                        t_idx = torch.arange(window_size, device=device, dtype=torch.long).view(1, 1, window_size)
                        idx = starts_bk.unsqueeze(-1) + t_idx
                        for dt in range(window_size):
                            # 一个时间点可能被多个窗口覆盖，所以这里要把窗口梯度 scatter 回
                            # 原始时间轴并做累加，而不是简单覆盖。
                            t = idx[:, :, dt]
                            gi = g_patch[:, :, dt]
                            grad_tokens.scatter_add_(1, t[:, :, None, None].expand(1, kc, gi.shape[2], gi.shape[3]), gi.float())
                        total_loss = total_loss + loss_chunk.detach().float() * w
                        done += kc

                    if done != K:
                        raise RuntimeError(f"窗口覆盖数量不一致：done={done} K={K}")

                    if adapter_trainable_now or bool(train_vae_now):
                        # 第一步 backward 只会把梯度打到 `patch_tokens`；
                        # 这里再把整段 token 的累积梯度回传给 Adapter / VAE。
                        tokens_btnd.backward(grad_tokens.to(dtype=tokens_btnd.dtype))
                    loss_sum = loss_sum + total_loss.detach()
                    used += 1

                if used <= 0:
                    t_after_step = time.time()
                    continue
                loss = loss_sum / float(used)
            else:
                z_ext = batch["z_ext"].to(device, non_blocking=True)  # (B,64,T_lat,16,16)
                traj_b = batch["traj"]
                traj_np = traj_b.detach().cpu().numpy() if isinstance(traj_b, torch.Tensor) else np.asarray(traj_b, dtype=np.float32)

                deltas = []
                for b in range(traj_np.shape[0]):
                    d = traj_to_delta(
                        traj_np[b],
                        traj_mode=str(args.traj_mode),
                        angles_in_degrees=bool(args.angles_in_degrees),
                        translation_divisor=float(args.translation_divisor),
                        delta_dyaw_unit=str(args.delta_dyaw_unit),
                    )
                    deltas.append(torch.from_numpy(d).unsqueeze(0))
                delta_bt6 = torch.cat(deltas, dim=0).to(device, non_blocking=True)  # (B,T,6) rad+m
                if label_stats_t is not None:
                    delta_bt6 = _normalize_delta_bt6(delta_bt6, label_stats_t)

                B = int(z_ext.shape[0])
                T_traj = int(delta_bt6.shape[1])

                allow_adapter_grad = adapter_trainable_now
                tokens_btnd, T_dec = _decode_tokens_full_T(
                    vae_decode_module=vae_decode,
                    adapter=adapter,
                    z_ext=z_ext,
                    allow_adapter_grad=allow_adapter_grad,
                    allow_vae_grad=bool(train_vae_now),
                    amp_enabled=bool(args.amp),
                    expected_T=T_traj,
                )
                T_use = min(int(T_dec), int(T_traj))
                if T_use < window_size:
                    t_after_step = time.time()
                    continue
                tokens_btnd = tokens_btnd[:, :T_use]
                delta_bt6 = delta_bt6[:, :T_use]

                starts_bk_all = _sample_all_window_starts(
                    B=B, T=T_use, window_size=window_size, stride=int(args.window_stride), device=device
                )
                K = int(starts_bk_all.shape[1])
                if K <= 0:
                    t_after_step = time.time()
                    continue

                chunk = int(args.windows_chunk)
                if chunk <= 0 or chunk >= K:
                    chunk = K

                total_loss = torch.zeros((), device=device, dtype=torch.float32)
                tokens_det = tokens_btnd.detach()
                grad_tokens = torch.zeros_like(tokens_det, dtype=torch.float32)
                done = 0
                for s0 in range(0, K, chunk):
                    # crop 模式下 batch 已被裁成等长；这里继续按窗口块切分，
                    # 只是为了控制 `B*K` 过大时的显存峰值。
                    s1 = min(K, s0 + chunk)
                    starts_bk = starts_bk_all[:, s0:s1]
                    kc = int(starts_bk.shape[1])
                    Bwin = int(B * kc)

                    patch_tokens_val = _gather_window_tokens(tokens_det, starts_bk, window_size=window_size)
                    patch_tokens = patch_tokens_val.detach().requires_grad_(True)
                    targets = _gather_window_targets(delta_bt6, starts_bk, window_size=window_size)

                    with autocast(enabled=bool(args.amp) and torch.cuda.is_available()):
                        feat = m.forward_features_from_patch_tokens(patch_tokens, B=Bwin, T=window_size, W=W_grid)
                        pred = m.head(feat)
                        loss_chunk = compute_loss_mse(
                            pred=pred,
                            target=targets,
                            window_size=window_size,
                            rot_weight=float(rot_w),
                            trans_xy_weight=float(args.trans_xy_weight),
                            trans_z_weight=float(trans_z_w),
                            trans_vertical_index=int(args.trans_vertical_index),
                        )
                    w = float(kc) / float(K)
                    scaler.scale((loss_chunk * w) * accum_scale).backward()
                    g_patch = patch_tokens.grad.detach()  # 代码/形状说明：(Bwin*window,N,D)
                    g_patch = g_patch.view(B, kc, window_size, g_patch.shape[1], g_patch.shape[2])
                    t_idx = torch.arange(window_size, device=device, dtype=torch.long).view(1, 1, window_size)
                    idx = starts_bk.unsqueeze(-1) + t_idx  # 代码/形状说明：(B,kc,window)
                    for dt in range(window_size):
                        # 把 chunk 内窗口梯度累计回整条 token 时间线。
                        t = idx[:, :, dt]  # (B,kc)
                        gi = g_patch[:, :, dt]  # (B,kc,N,D)
                        grad_tokens.scatter_add_(1, t[:, :, None, None].expand(B, kc, gi.shape[2], gi.shape[3]), gi.float())
                    total_loss = total_loss + loss_chunk.detach().float() * w
                    done += kc

                if done != K:
                    raise RuntimeError(f"窗口覆盖数量不一致：done={done} K={K}")

                if adapter_trainable_now or bool(train_vae_now):
                    # crop 模式与 per_sample 模式相同：先算窗口 loss，再把累积梯度回传给整段 token。
                    tokens_btnd.backward(grad_tokens.to(dtype=tokens_btnd.dtype))
                loss = total_loss

            running += loss.detach()
            step_s = time.time() - t_step0
            t_after_step = time.time()

            accum_i += 1
            do_step = (accum_i >= grad_accum_steps)
            if do_step:
                _allreduce_grads(optimizer.param_groups)
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    all_params = [p for pg in optimizer.param_groups for p in pg["params"] if p.grad is not None]
                    if all_params:
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                accum_i = 0

            if pbar is not None:
                avg = (running / max(1, nb)).detach().item()
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.detach().item():.6f}",
                    avg=f"{avg:.6f}",
                    rot_w=f"{rot_w:.4f}",
                    trans_z_w=f"{trans_z_w:.4f}",
                    data_s=f"{data_s:.2f}",
                    step_s=f"{step_s:.2f}",
                    step=global_step,
                )

            if get_rank() == 0 and (global_step % int(args.log_every) == 0):
                avg = (running / max(1, nb)).detach().item()
                dt = time.time() - t0
                rank0_print(
                    f"[stage2] epoch={epoch+1}/{args.epochs} step={global_step} "
                    f"loss={loss.detach().item():.6f} avg={avg:.6f} rot_w={rot_w:.4f} trans_z_w={trans_z_w:.4f} dt={dt:.1f}s"
                )
                t0 = time.time()

        if pbar is not None:
            pbar.close()

        if accum_i > 0:
            _allreduce_grads(optimizer.param_groups)
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                all_params = [p for pg in optimizer.param_groups for p in pg["params"] if p.grad is not None]
                if all_params:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            accum_i = 0

        train_loss = (running / max(1, nb)).detach()
        train_loss = reduce_mean(train_loss).item()
        if get_rank() == 0:
            rank0_print(f"[epoch] {epoch+1} train_loss={train_loss:.6f}")
            if adapter_trainable_now and (epoch + 1) <= int(args.freeze_adapter_epochs) + 3:
                adapter_grad_norm = 0.0
                adapter_grad_count = 0
                for p in adapter.parameters():
                    if p.grad is not None:
                        adapter_grad_norm += p.grad.detach().float().norm().item() ** 2
                        adapter_grad_count += 1
                adapter_grad_norm = adapter_grad_norm ** 0.5
                rank0_print(
                    f"[grad-diag] adapter: {adapter_grad_count}/{sum(1 for _ in adapter.parameters())} 个参数张量有梯度, "
                    f"grad_norm={adapter_grad_norm:.6f}"
                )

            # 进入 VAE 训练阶段前，先保存最后一个 checkpoint。
            # 当 (epoch+1) > train_vae_after_epoch 时 train_vae_now 才会变 True，
            # 所以最后一个“完整训练前”epoch 正好是 train_vae_after_epoch。
            if int(args.train_vae_after_epoch) < 10**9 and (epoch + 1) == int(args.train_vae_after_epoch):
                to_save = tsformer
                adp_save = adapter
                m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                vae_sd_full = m_vae.vae.state_dict()
                vae_sd = {
                    k: v.detach().cpu()
                    for k, v in vae_sd_full.items()
                    if k.startswith(("proj_up.", "post_quant_conv.", "decoder.", "scale_learnable_parameters"))
                }
                state_pre = {
                    "format": "stage2_latent2action_checkpoint_pre_fulltrain_v1",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    "adapter_state_dict": {k: v.detach().cpu() for k, v in adp_save.state_dict().items()},
                    "vae_state_dict": vae_sd,
                    "label_stats": label_stats_np,
                    "label_stats_source": label_stats_src,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                    "time": datetime.now().isoformat(),
                }
                path_pre = os.path.join(str(args.out_dir), f"checkpoint_pre_fulltrain_e{epoch+1}.pth")
                torch.save(state_pre, path_pre)
                rank0_print(f"[保存] 完整训练前 checkpoint：{path_pre}")

            if (epoch + 1) % int(args.save_every) == 0:
                to_save = tsformer
                adp_save = adapter
                m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                vae_sd_full = m_vae.vae.state_dict()
                vae_sd = {
                    k: v.detach().cpu()
                    for k, v in vae_sd_full.items()
                    if k.startswith(("proj_up.", "post_quant_conv.", "decoder.", "scale_learnable_parameters"))
                }
                state = {
                    "format": "stage2_latent2action_checkpoint_combined_v1",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    "adapter_state_dict": {k: v.detach().cpu() for k, v in adp_save.state_dict().items()},
                    "vae_state_dict": vae_sd,
                    "label_stats": label_stats_np,
                    "label_stats_source": label_stats_src,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_val": best_val,
                    "args": vars(args),
                    "time": datetime.now().isoformat(),
                }
                torch.save(state, os.path.join(str(args.out_dir), "checkpoint_last.pth"))
                torch.save(state, os.path.join(str(args.out_dir), f"checkpoint_e{epoch+1}.pth"))

            if bool(args.export_combined) and ((epoch + 1) % int(args.save_every) == 0):
                to_save = tsformer
                adp_save = adapter
                m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                vae_sd_full = m_vae.vae.state_dict()
                vae_sd = {
                    k: v.detach().cpu()
                    for k, v in vae_sd_full.items()
                    if k.startswith(("proj_up.", "post_quant_conv.", "decoder.", "scale_learnable_parameters"))
                }
                export = {
                    "format": "stage2_latent2action_combined_v2_resumable",
                    "created_at": datetime.now().isoformat(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "infinitystar_vae_path": str(args.infinitystar_vae_path),
                    "infinitystar_vae_type": int(args.infinitystar_vae_type),
                    "tsformer_pretrained": str(args.tsformer_pretrained),
                    "adapter_ckpt": str(args.adapter_ckpt),
                    "model_config": {"num_frames": 4, "embed_dim": 384, "patch_size": 16, "W_grid": W_grid},
                    "vae_state_dict": vae_sd,
                    "label_stats": label_stats_np,
                    "label_stats_source": label_stats_src,
                    "adapter_state_dict": {k: v.detach().cpu() for k, v in adp_save.state_dict().items()},
                    "tsformer_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    # 为 --resume 兼容保留的别名；resume 逻辑期望 model_state_dict。
                    "model_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                }
                torch.save(export, os.path.join(str(args.out_dir), str(args.export_name)))

    if _RANK0_LOG_FH is not None and get_rank() == 0:
        _RANK0_LOG_FH.close()


if __name__ == "__main__":
    main()
