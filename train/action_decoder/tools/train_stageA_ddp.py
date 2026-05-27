"""
Stage-1：蒸馏一个 Adapter，把 InfinityStar VAE decoder 最后一个 `up_block`
（当前默认结构里通常对应 `up_block_3`）的特征映射到 TSformer PatchEmbed token。

教师分支来源（当前版本）：
- teacher tokens 来自原始 PNG 帧（manifest images_dir 中的 frames_rgb），
  再经过 TSformer `patch_embed`。

学生分支来源：
- student tokens 来自 InfinityStar VAE decoder 最后一个 `up_block` 的 feature（hook 捕获），再经过 Adapter。

说明：Loss：
- 复合蒸馏损失（cosine + 可选 distribution stats + 可选 MSE）。

DDP 注意事项：
- InfinityStar VAE 可能启用 batch slicing（use_slicing=True），导致 B>1 时 hook 看到的 batch 仍是 1。
  slicing 开启时需要按样本 decode 来对齐。
- tqdm 和 train.log 只在 rank0 使用，避免多 rank 重复刷屏。

中文导读：
Stage A 不是直接训练动作输出，而是先把“VAE decoder 中间特征”对齐到“TimesFormer
原生 patch_embed token 空间”。这样 Stage B 才能在不依赖真实 RGB patch_embed 的情况下，
从世界模型 latent 走到动作头输入。读代码时重点看三件事：
1. teacher token 来自真实 PNG 帧的 TimesFormer patch_embed；
2. student token 来自 latent 经 VAE decoder hook 捕获的最后一个 up block 特征；
3. `compute_distill_loss()` 用方向、均值、方差等约束让两种 token 分布对齐。
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
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 即使通过绝对路径启动，也要确保 action-decoder 架构代码可 import。
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.path.abspath(os.path.join(_TOOL_DIR, ".."))
_OPEN_ROOT = os.path.abspath(os.path.join(_TRAIN_ROOT, "..", ".."))
_ARCH_ROOT = os.path.join(_OPEN_ROOT, "Worldmodel", "action_decoder", "src")
if _ARCH_ROOT not in sys.path:
    sys.path.insert(0, _ARCH_ROOT)

from datasets.latent_traj_manifest import LatentTrajManifestDataset  # noqa: E402
from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter  # noqa: E402
from timesformer.models.vit import VisionTransformer  # noqa: E402

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


KITTI_MEAN = torch.tensor([0.34721234, 0.36705238, 0.36066107], dtype=torch.float32).view(1, 3, 1, 1, 1)
KITTI_STD = torch.tensor([0.30737526, 0.31515116, 0.32020183], dtype=torch.float32).view(1, 3, 1, 1, 1)
_KITTI_MEAN_3 = (0.34721234, 0.36705238, 0.36066107)
_KITTI_STD_3 = (0.30737526, 0.31515116, 0.32020183)


def compute_distill_loss(
    tok_s: torch.Tensor,
    tok_t: torch.Tensor,
    w_cos: float = 1.0,
    w_mean: float = 0.1,
    w_std: float = 0.1,
    w_mse: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """复合蒸馏损失：cosine 方向 + 分布统计量 + 可选 MSE。

        初学者公式：
            公式/形状说明：L_total = w_cos*L_cos + w_mse*L_mse + w_mean*L_mean + w_std*L_std。
            L_cos = mean(1 - cos(student, teacher))，教 student token 的方向接近 teacher；
            L_mse = mean((student - teacher)^2)，教 token 的逐元素数值接近；
            L_mean = MSE(mean(student), mean(teacher))，教整批 token 的分布中心接近；
            L_std = MSE(std(student), std(teacher))，教整批 token 的分布宽窄接近。

        参数：
            tok_s, tok_t: shape 相同的 student/teacher tokens，最后一维是 D。
            w_cos:  (1 - cosine_similarity) 的权重，用来对齐向量方向。
            w_mean: 每个维度 mean 的 MSE 权重，用来对齐分布中心。
            w_std:  每个维度 std 的 MSE 权重，用来对齐分布离散程度。
            w_mse:  原始 MSE 权重，对数值大小敏感，可选。
        返回：
            (带梯度的总 loss, {component_name: float_value})

    """
    D = tok_s.shape[-1]
    s_flat = tok_s.reshape(-1, D)
    t_flat = tok_t.reshape(-1, D)

    parts: dict = {}
    total = s_flat.new_zeros(())

    if w_cos > 0:
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
        l_cos = (1.0 - cos_sim).mean()
        parts["cos"] = float(l_cos.detach())
        total = total + w_cos * l_cos

    if w_mean > 0:
        l_mean = F.mse_loss(s_flat.mean(dim=0), t_flat.mean(dim=0))
        parts["mean"] = float(l_mean.detach())
        total = total + w_mean * l_mean

    if w_std > 0 and s_flat.shape[0] > 1:
        l_std = F.mse_loss(s_flat.std(dim=0), t_flat.std(dim=0))
        parts["std"] = float(l_std.detach())
        total = total + w_std * l_std

    if w_mse > 0:
        l_mse = F.mse_loss(s_flat, t_flat)
        parts["mse"] = float(l_mse.detach())
        total = total + w_mse * l_mse

    parts["total"] = float(total.detach())
    return total, parts


def _linear(a: float, b: float, t01: float) -> float:
    """线性插值工具，用于 cosine/MSE loss 权重调度。"""
    t01 = float(max(0.0, min(1.0, t01)))
    return float(a + (b - a) * t01)


def _loss_weights_for_epoch(epoch_num_1idx: int, args) -> Tuple[float, float]:
    """
    根据 1-indexed epoch 编号返回 (w_cos, w_mse)。

    schedule 也按 1-indexed epoch 定义，这样和用户习惯的轮数一致。
    例子：如果 `hold=40, ramp_start=45, ramp_end=55`，那么第 1~40 轮保持起始权重，
    第 45~55 轮线性切换，第 56 轮开始固定使用结束权重。
    """
    if str(getattr(args, "loss_schedule", "none")).strip().lower() in ("", "none"):
        return float(args.loss_cosine_w), float(args.loss_mse_w)

    # 分段线性 schedule（兼容旧实验）。
    hold = int(getattr(args, "loss_hold_epochs", 40))
    ramp_s = int(getattr(args, "loss_ramp_start_epoch", 45))
    ramp_e = int(getattr(args, "loss_ramp_end_epoch", 55))
    cos_a = float(getattr(args, "loss_cosine_w_start", 1.0))
    cos_b = float(getattr(args, "loss_cosine_w_end", 0.1))
    mse_a = float(getattr(args, "loss_mse_w_start", 0.1))
    mse_b = float(getattr(args, "loss_mse_w_end", 1.0))

    if epoch_num_1idx <= hold:
        return cos_a, mse_a
    if ramp_s <= epoch_num_1idx <= ramp_e:
        denom = max(1, (ramp_e - ramp_s))
        t01 = float(epoch_num_1idx - ramp_s) / float(denom)
        return _linear(cos_a, cos_b, t01), _linear(mse_a, mse_b, t01)
    if epoch_num_1idx > ramp_e:
        return cos_b, mse_b

    # hold 和 ramp start 之间的空档（例如 epoch 41-44）保持起始权重。
    return cos_a, mse_a


_RANK0_LOG_FH = None
_RANK0_LOG_PATH = None
_RANK0_LOG_BROKEN = False


def seed_everything(seed: int):
    """设置 Python、NumPy、Torch 和 CUDA 随机种子，保证多卡训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    """判断当前进程是否已经初始化 torch.distributed。"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """返回当前 DDP rank；非分布式运行时视作 rank0。"""
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    """返回 DDP world size；单进程运行时为 1。"""
    return dist.get_world_size() if is_dist() else 1


def rank0_print(*args, **kwargs):
    """只在 rank0 打印，并同步写入可选 train.log。"""
    if get_rank() == 0:
        global _RANK0_LOG_FH
        global _RANK0_LOG_BROKEN
        print(*args, **kwargs, flush=True)
        if _RANK0_LOG_FH is not None:
            try:
                msg = " ".join(str(a) for a in args)
                _RANK0_LOG_FH.write((msg + "\n").encode("utf-8", errors="replace"))
            except OSError as e:
                _RANK0_LOG_BROKEN = True
                try:
                    _RANK0_LOG_FH.close()
                except Exception:
                    pass
                _RANK0_LOG_FH = None
                print(f"[警告] train.log 写入失败，已关闭文件日志。err={e}", flush=True)


def rank0_log(msg: str):
    """
    只在 rank0 往 train.log 写一行，不额外刷 stdout。
    """
    if get_rank() == 0:
        global _RANK0_LOG_FH, _RANK0_LOG_PATH, _RANK0_LOG_BROKEN
        if _RANK0_LOG_BROKEN:
            return
        if _RANK0_LOG_PATH is None:
            return

        # 延迟打开或重开文件，尽量避开不稳定文件系统的偶发问题。
        if _RANK0_LOG_FH is None:
            try:
                _RANK0_LOG_FH = open(str(_RANK0_LOG_PATH), "ab", buffering=0)
            except OSError as e:
                _RANK0_LOG_BROKEN = True
                print(f"[警告] train.log 打开失败，已关闭文件日志。err={e}", flush=True)
                return
        try:
            _RANK0_LOG_FH.write((str(msg) + "\n").encode("utf-8", errors="replace"))
        except OSError:
            try:
                _RANK0_LOG_FH.close()
            except Exception:
                pass
            _RANK0_LOG_FH = None
            try:
                _RANK0_LOG_FH = open(str(_RANK0_LOG_PATH), "ab", buffering=0)
                _RANK0_LOG_FH.write((str(msg) + "\n").encode("utf-8", errors="replace"))
            except OSError as e:
                _RANK0_LOG_BROKEN = True
                try:
                    if _RANK0_LOG_FH is not None:
                        _RANK0_LOG_FH.close()
                except Exception:
                    pass
                _RANK0_LOG_FH = None
                print(f"[警告] train.log 写入失败，已关闭文件日志。err={e}", flush=True)


def _atomic_torch_save(obj, path: str) -> None:
    """
    原子写入 torch checkpoint，避免任务在保存中被抢占或崩溃时留下损坏的 .pt 文件。
    """
    path = str(path)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ddp_setup() -> int:
    """从 torchrun 环境变量初始化进程组，并返回 local_rank。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """跨 DDP rank 求平均，用于日志指标而不是反向传播。"""
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= get_world_size()
    return y


def build_tsformer() -> VisionTransformer:
    """
    构建 Stage A 的 frozen teacher TimesFormer。

    数据流位置：
    - teacher 分支：`frames_rgb -> TimesFormer patch_embed/forward_features -> teacher tokens`；
    - student 分支：`z_ext -> VAE decoder up_block_3 feature -> Adapter -> student tokens`。

    关键约束：
    这里的 `img_size=(192,640)`、`num_frames=4`、`embed_dim=384` 必须和 teacher checkpoint
    训练时一致；否则即使用 `strict=False` 加载成功，teacher token 空间也会发生语义错位。
    """
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
    """加载 teacher TimesFormer 权重；strict=False 兼容不同实验分支的 head 差异。"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _resolve_ckpt_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        rank0_print(f"[警告] TSFormer 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}")


def _add_infinitystar_to_syspath(inf_root: Optional[str], proj_root: str):
    """把 InfinityStar runtime 加入 sys.path，优先使用显式参数或环境变量。"""
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
    """
    加载冻结的 InfinityStar video VAE；Stage A 只使用 decoder 中间 feature 训练 Adapter。

    输入来自命令行：
    - `vae_path`：Video VAE 权重；
    - `vae_type`：codebook/quant bits 维度，常见为 64；
    - `semantic_scale_dim/detail_scale_dim/...`：必须和 VAE checkpoint 训练配置一致。

    输出：
    - 返回 eval 模式的 `vae`，所有参数 `requires_grad=False`；
    - 训练循环会给 `vae.decoder.up_blocks[-1]` 注册 hook，抓取 `(B,96,t,H,W)` 特征。
    """
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


def _preprocess_decoded_for_tsformer(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    x: (B,3,T,H,W) float，取值范围可以是 [-1,1] 或 [0,1]。
    返回归一化后的 (B,3,T,192,640)。

    中文导读：
    VAE decode 出来的图像更像“中间表征还原图”，而 teacher 分支的 TSFormer 是按
    KITTI 风格 RGB 统计量训练的，所以这里要把 student 侧的图像也变到相同输入域：
    1. 若输入是 `[-1,1]`，先映射回 `[0,1]`；
    2. resize 到 TSFormer 训练时使用的 `(192, 640)`；
    3. 用 KITTI mean/std 归一化，保证 teacher/student 看见的像素分布尽量一致。
    """
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError(f"decode 后帧张量必须是 (B,3,T,H,W)，收到的是 {tuple(x.shape)}")
    x = x.to(device=device, dtype=torch.float32)
    if float(x.min()) < -1e-3:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    B, C, T, H, W = x.shape
    xt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (BT,3,H,W)
    xt = F.interpolate(xt, size=(192, 640), mode="bilinear", align_corners=False)
    xt = xt.view(B, T, C, 192, 640).permute(0, 2, 1, 3, 4).contiguous()

    mean = KITTI_MEAN.to(device=device)
    std = KITTI_STD.to(device=device)
    xt = (xt - mean) / std
    return xt


def _build_png_transform():
    """
    构建逐帧 transform，使其匹配 TSformer 训练预处理：
    先 Resize 到 (192,640)，再 ToTensor+Normalize(KITTI mean/std)。

    它和 `_preprocess_decoded_for_tsformer()` 的分工是：
    - 这个函数处理 teacher 分支的 PNG 文件读取；
    - `_preprocess_decoded_for_tsformer()` 处理 student 分支从 VAE decode 出来的张量。
    两边最后都要落到 TSFormer 熟悉的同一像素统计域。
    """
    try:
        from torchvision import transforms  # type: ignore

        return transforms.Compose(
            [
                transforms.Resize((192, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_KITTI_MEAN_3, std=_KITTI_STD_3),
            ]
        )
    except Exception as e:
        raise RuntimeError(f"PNG teacher 分支需要 torchvision 做逐帧 transform，但当前环境不可用：{e}")


def _expected_frames_from_latent_chunk(latent_chunk_len: int) -> int:
    """根据 video VAE 时间压缩关系，把 latent chunk 长度换算成 RGB 帧数。"""
    # 类 Wan 的时间上采样关系：T_frames = 4*(T_lat-1)+1。
    L = int(latent_chunk_len)
    if L < 2:
        return 1
    return 4 * (L - 1) + 1


def _decode_teacher_student_tokens(
    vae_decode_module: nn.Module,
    adapter: nn.Module,
    tsformer: VisionTransformer,
    z_sub: torch.Tensor,
    frames_teacher: torch.Tensor,
    amp_enabled: bool,
    adapter_frames_chunk: int = 0,
    adapter_use_checkpoint: bool = False,
    expected_T: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    返回：(tokens_student, tokens_teacher, T_frames)
      - `tokens_student`：来自 `z_sub -> VAE decoder 最后一个 up block -> Adapter`，形状 `(B,T,N,D)`；
      - `tokens_teacher`：来自 `frames_teacher -> TSFormer patch_embed`，形状 `(B,T,N,D)`；
      - `T_frames`：对齐后的时间长度 `T`。

    关键输入/形状：
    - `z_sub`: `(B, C_lat, T_lat, H_lat, W_lat)`，是当前窗口的 latent；
    - hook 输出 `hs`: `(B, C_feat, T_frame_like, H_feat, W_feat)`，其中时间维已经被 VAE 上采样；
    - Adapter 输出 token：先展平成 `(B*T, N, D)`，再还原成 `(B,T,N,D)`。

    为什么 `expected_T` 可能对不上：
    - VAE 的时间上采样遵循近似 `4*(T_lat-1)+1` 的规则，但不同实现可能因边界裁剪、
      slicing 或空片段而少一两帧；
    - teacher 的 PNG 帧数量也可能因为样本缺帧、裁窗方式不同而和 student 不完全一致。
    因此这里最终按二者较短的一侧对齐，保证 loss 在共同时间段上计算。
    """
    m = vae_decode_module.module if isinstance(vae_decode_module, DDP) else vae_decode_module
    vae = m.vae
    B = int(z_sub.shape[0])

    # 重要：
    # VAE decode 放在 `no_grad` 下，节省显存并避免构建 VAE 计算图。
    # 不要在 VAE hook 内部的 `no_grad` 作用域里运行 `adapter()`；有些环境会导致
    # adapter 梯度没有填充，从而让 AdamW optimizer state（优化器状态）一直为空。
    # 正确做法是先捕获 feature slice，再在启用梯度的上下文中运行 `adapter()`。
    feat_slices = []
    start = 0

    def hook(_module, _inp, out):
        """VAE decoder hook：收集中间 feature，稍后再带梯度通过 adapter。"""
        nonlocal start, feat_slices
        hs = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
            raise RuntimeError("VAE 最后一个 up block 的 hook 输出不是 5D Tensor")
        t_slice = int(hs.shape[2])
        # 保存 detach 后的 feature slice，稍后再交给 adapter 使用。
        feat_slices.append(hs.detach())
        start += t_slice

    handle = vae.decoder.up_blocks[-1].register_forward_hook(hook)
    try:
        grad_ctx = torch.no_grad()
        amp_ctx = autocast(enabled=bool(amp_enabled) and torch.cuda.is_available())
        try:
            with grad_ctx, amp_ctx:
                _ = vae_decode_module(z_sub)  # 只为触发 hook 而 decode；输出会丢弃，teacher 来自 PNG。
        except RuntimeError as e:
            msg = str(e)
            if "torch.cat(): expected a non-empty list of Tensors" in msg:
                rank0_print(f"[警告] VAE decode 得到空 dec 列表，跳过该片段。z_sub={tuple(z_sub.shape)} err={msg}")
                empty = torch.empty((B, 0, 480, 384), device=z_sub.device, dtype=torch.float32)
                return empty, empty, 0
            raise
    finally:
        handle.remove()

    if len(feat_slices) == 0 or int(start) <= 0:
        empty = torch.empty((B, 0, 480, 384), device=z_sub.device, dtype=torch.float32)
        return empty, empty, 0

    # 在启用梯度的上下文中运行 adapter，得到 student tokens。
    # 重要（显存）：
    # Adapter 会把输入 reshape 成 (B*T,C,H,W) 后逐帧处理。一次喂完整 49 帧会保留很大的
    # (B*T) 激活用于反传，容易 OOM。因此这里按时间切块运行 adapter；因为 adapter 内部
    # 没有时间混合，所以数学上等价。
    chunk = int(adapter_frames_chunk)
    if chunk <= 0:
        chunk = 8  # 完整 decode 时的保守默认值。
    tok_slices = []
    amp_ctx_s = autocast(enabled=bool(amp_enabled) and torch.cuda.is_available())
    with amp_ctx_s:
        for hs in feat_slices:
            Bh = int(hs.shape[0])
            t_slice = int(hs.shape[2])
            for t0 in range(0, t_slice, chunk):
                t1 = min(t0 + chunk, t_slice)
                hs_sub = hs[:, :, t0:t1].contiguous()
                if bool(adapter_use_checkpoint):
                    def _adp(x: torch.Tensor) -> torch.Tensor:
                        """checkpoint 包装：只返回 adapter token，丢弃辅助 shape 输出。"""
                        y, _t2, _w2 = adapter(x)
                        return y
                    # 这里必须使用 use_reentrant=False，因为 hs_sub 已 detach（requires_grad=False）。
                    # 默认 reentrant checkpoint 会因没有输入需要梯度而不建图，导致 loss 没有 grad_fn。
                    tok = checkpoint(_adp, hs_sub, use_reentrant=False)  # (Bh*(t1-t0),N,D)
                else:
                    tok, _t2, _w2 = adapter(hs_sub)  # (Bh*(t1-t0),N,D)
                tok = tok.view(Bh, (t1 - t0), tok.shape[1], tok.shape[2]).contiguous()
                tok_slices.append(tok)
    tok_s = torch.cat(tok_slices, dim=1)  # (B,T,N,D)
    T_s = int(tok_s.shape[1])
    if expected_T is not None and T_s != int(expected_T):
        rank0_print(f"[警告] student 解码帧数 T={T_s}，但按 latent_chunk_len 推导应为 expected_T={expected_T}")

    # teacher 来自原始 PNG 帧。
    x_in = frames_teacher.to(device=z_sub.device, dtype=torch.float32, non_blocking=True)
    if x_in.ndim != 5 or int(x_in.shape[1]) != 3:
        raise ValueError(f"frames_teacher 必须是 (B,3,T,H,W)，收到的是 {tuple(x_in.shape)}")
    with torch.no_grad():
        tok_t, T_t, _W = tsformer.patch_embed(x_in)  # (B*T,N,D)
    tok_t = tok_t.view(B, int(T_t), tok_t.shape[1], tok_t.shape[2]).contiguous()
    if expected_T is not None and int(T_t) != int(expected_T):
        rank0_print(f"[警告] teacher 帧数 T={int(T_t)}，但按 latent_chunk_len 推导应为 expected_T={expected_T}")

    # 长度不一致时按较短的一侧对齐。
    T = min(int(tok_s.shape[1]), int(tok_t.shape[1]))
    if T <= 0:
        empty = torch.empty((B, 0, 480, 384), device=z_sub.device, dtype=torch.float32)
        return empty, empty, 0
    if int(tok_s.shape[1]) != T:
        tok_s = tok_s[:, :T].contiguous()
    if int(tok_t.shape[1]) != T:
        tok_t = tok_t[:, :T].contiguous()
    return tok_s, tok_t, int(T)


def collate_fn(samples, mode: str = "crop", latent_chunk_len: int = 3):
    """Stage A batch 组装函数：对齐 latent chunk 和 teacher RGB 帧，过滤坏样本。"""
    mode = str(mode).strip().lower()
    L = int(latent_chunk_len)
    if mode == "per_sample":
        out_samples = []
        skipped_no_frames = 0
        skipped_no_latent = 0
        for s in samples:
            z = s.get("z_ext", None)
            if z is None:
                skipped_no_latent += 1
                continue
            if not isinstance(z, torch.Tensor):
                z = torch.as_tensor(z)
            fr = s.get("frames_rgb", None)
            if fr is None:
                # LatentTrajManifestDataset 使用 on_error="empty" 时，
                # 可能会故意返回缺少 frames_rgb 的部分样本。
                # 这里跳过这些样本，让训练继续运行。
                skipped_no_frames += 1
                continue
            if not isinstance(fr, torch.Tensor):
                fr = torch.as_tensor(fr)
            out_samples.append({"z_ext": z.float().contiguous(), "frames_rgb": fr.float().contiguous(), "meta": s.get("meta", {})})
        return {
            "samples": out_samples,
            "meta": [s.get("meta", {}) for s in samples],
            "skipped_no_frames": int(skipped_no_frames),
            "skipped_no_latent": int(skipped_no_latent),
            "total_in": int(len(samples)),
            "total_out": int(len(out_samples)),
        }

    z_list = []
    t_list = []
    for s in samples:
        z = s["z_ext"]
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        z = z.float().contiguous()
        z_list.append(z)
        t_list.append(int(z.shape[2]))

    t_min = min(t_list) if len(t_list) else 0
    # 只要有一个样本太短，按最短长度裁剪就会让整个 batch 不可用。
    # 因此退回 per-sample 输出，让训练循环只跳过短样本。
    if t_min < L:
        out_samples = []
        skipped_no_frames = 0
        for z, s in zip(z_list, samples):
            out = {"z_ext": z, "meta": s.get("meta", {})}
            fr = s.get("frames_rgb", None)
            if fr is not None:
                if not isinstance(fr, torch.Tensor):
                    fr = torch.as_tensor(fr)
                out["frames_rgb"] = fr.float().contiguous()
            else:
                skipped_no_frames += 1
            out_samples.append(out)
        return {
            "samples": out_samples,
            "meta": [s.get("meta", {}) for s in samples],
            "skipped_no_frames": int(skipped_no_frames),
            "total_in": int(len(samples)),
            "total_out": int(len(out_samples)),
        }
    if t_min <= 0:
        raise ValueError(f"batch 中存在非法 latent 时间长度：{t_list}")
    # 把 frames 堆叠起来并裁到最短 T；要求所有样本都有 frames_rgb。
    fr_list = []
    ft_list = []
    for s in samples:
        fr = s.get("frames_rgb", None)
        if fr is None:
            raise ValueError("crop collate 需要每个样本都有 frames_rgb；请确认 dataset 设置了 load_frames=True")
        if not isinstance(fr, torch.Tensor):
            fr = torch.as_tensor(fr)
        fr = fr.float().contiguous()
        fr_list.append(fr)
        ft_list.append(int(fr.shape[1]))
    t_frame_min = int(min(ft_list)) if ft_list else 0
    if t_frame_min <= 0:
        raise ValueError(f"batch 中存在非法 RGB 帧长度：{ft_list}")

    z_ext = torch.cat([z[:, :, :t_min].contiguous() for z in z_list], dim=0).contiguous()  # (B,64,t_min,16,16)
    frames_rgb = torch.stack([fr[:, :t_frame_min].contiguous() for fr in fr_list], dim=0).contiguous()  # (B,3,T,192,640)
    meta = [s.get("meta", {}) for s in samples]
    return {"z_ext": z_ext, "frames_rgb": frames_rgb, "meta": meta}


def main():
    """
    Stage A 训练入口。

    中文导读：
    这个 `main()` 可以按四段读：
    1. 解析 manifest、VAE、TimesFormer、loss、DDP 等配置；
    2. 冻结 TimesFormer 作为 teacher，训练 Adapter 作为 student；
    3. 数据集同时加载 latent 和 RGB 帧，形成 student/teacher 两条 token 路径；
    4. 保存 adapter checkpoint，供 Stage B 的 latent-to-action 训练复用。
    """
    ap = argparse.ArgumentParser(
        description=(
            "Stage A 蒸馏训练：把 InfinityStar VAE decoder 最后一个 `up block` 的特征"
            "映射到 TimesFormer patch_embed token 空间。初学者建议先看本文件顶部总注释，再看 "
            "`compute_student_teacher_tokens()` 和这里的参数区。"
        )
    )
    ap.add_argument(
        "--manifest_json",
        type=str,
        required=True,
        help="训练 manifest 的 JSON 路径，里面列出 latent、RGB 帧路径以及样本元数据。",
    )
    ap.add_argument(
        "--items_key",
        type=str,
        default="ALL",
        help="要读取 manifest 里的哪个 items_* 列表；可逗号分隔多个 key，`ALL` 表示全部使用。",
    )
    ap.add_argument(
        "--workspace_root",
        type=str,
        default="",
        help=(
            "解析 manifest 内相对路径时使用的根目录。留空时默认使用当前开源包根目录"
            "（包含 `Worldmodel/` 与 `train/` 的目录）；如果 manifest 里的路径是相对"
            "于别的根目录生成的，就在这里显式指定。"
        ),
    )
    ap.add_argument("--max_items", type=int, default=0, help="最多读取多少条样本；0 表示不截断，常用于小规模调试。")
    # 匹配旧版 stage1 launcher 默认值：不强制固定帧长。
    ap.add_argument("--require_T", type=int, default=0, help="是否强制样本满足固定帧长 T；0 表示保持旧版默认，不强制。")

    ap.add_argument("--tsformer_ckpt", type=str, required=True, help="冻结的 teacher TimesFormer checkpoint 路径。")
    ap.add_argument("--infinitystar_vae_path", type=str, required=True, help="冻结的 InfinityStar VAE checkpoint 路径。")
    ap.add_argument("--infinitystar_vae_type", type=int, default=64, help="VAE 类型编号，必须和 checkpoint 对应的网络结构一致。")
    ap.add_argument("--infinitystar_root", type=str, default="", help="InfinityStar runtime 根目录；留空时按仓库默认位置自动推断。")

    # 这些参数必须和 VAE checkpoint 的结构一致。
    ap.add_argument("--semantic_scale_dim", type=int, default=16, help="VAE 语义尺度通道维度，必须与 checkpoint 中训练时配置对齐。")
    ap.add_argument("--detail_scale_dim", type=int, default=64, help="VAE 细节尺度通道维度，必须与 checkpoint 中训练时配置对齐。")
    ap.add_argument("--use_learnable_dim_proj", type=int, default=0, help="是否启用 VAE 内部可学习维度投影开关，需与原模型结构一致。")
    ap.add_argument("--detail_scale_min_tokens", type=int, default=350, help="细节尺度最少 token 数，用来构造与 checkpoint 匹配的 VAE 结构。")
    ap.add_argument("--use_feat_proj", type=int, default=2, help="VAE 特征投影模式编号，必须与训练该 VAE 时的配置一致。")
    ap.add_argument("--semantic_scales", type=int, default=11, help="VAE 使用的语义尺度数量，必须与 checkpoint 对齐。")

    ap.add_argument("--out_dir", type=str, required=True, help="输出目录，用于保存 checkpoint、导出文件和训练日志。")
    # 匹配旧版 stage1 launcher 默认值。
    ap.add_argument("--epochs", type=int, default=100, help="训练 epoch 数。")
    ap.add_argument("--global_batch_size", type=int, default=32, help="全局 batch size；大于 0 时会按 world size 自动换算每卡 batch_size。")
    ap.add_argument("--batch_size", type=int, default=4, help="单卡 batch size；若设置了 `--global_batch_size`，这里会被自动覆盖。")
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 数量。")
    ap.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用持久化 DataLoader worker；当 `num_workers > 0` 时通常建议开启，能减少 epoch 之间反复重启进程。",
    )
    ap.add_argument("--prefetch_factor", type=int, default=2, help="每个 DataLoader worker 预取多少个 batch；仅在 `num_workers > 0` 时生效。")
    ap.add_argument(
        "--collate_mode",
        type=str,
        default="per_sample",
        choices=["crop", "per_sample"],
        help="批组装模式：`per_sample` 逐样本 decode，最稳；`crop` 把样本裁到公共最短长度后再拼批，吞吐更高。",
    )

    ap.add_argument("--lr", type=float, default=5e-5, help="Adapter 优化器学习率。")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减系数。")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子；不同 DDP rank 会在此基础上再偏移。")
    ap.add_argument("--save_every", type=int, default=4, help="每隔多少个 epoch 保存一次 checkpoint。")
    ap.add_argument("--log_every", type=int, default=20, help="每隔多少个 step 打印一次训练日志。")
    ap.add_argument("--tqdm", action="store_true", default=False, help="是否启用 tqdm 进度条；通常只在 rank0 有意义。")
    ap.add_argument("--log_file", type=str, default="train.log", help="rank0 文件日志名称。")
    ap.add_argument("--log_dir", type=str, default="", help="rank0 文件日志目录；留空时默认写入 `out_dir`。")
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 AMP 混合精度；通常建议开启，若需排查数值问题可用 `--no-amp` 关闭。",
    )
    ap.add_argument("--grad_clip", type=float, default=2.0, help="梯度裁剪最大范数；设为 0 表示关闭。")

    # 旧版 launcher 默认：cosine + MSE schedule（mean/std 默认关闭）。
    ap.add_argument("--loss_cosine_w", type=float, default=1.0, help="cosine 蒸馏项的固定权重，或分段 schedule 的默认起点。")
    ap.add_argument("--loss_mean_w", type=float, default=0.0, help="token 均值对齐项权重，用来约束分布中心。")
    ap.add_argument("--loss_std_w", type=float, default=0.0, help="token 标准差对齐项权重，用来约束分布离散程度。")
    ap.add_argument("--loss_mse_w", type=float, default=0.1, help="逐元素 MSE 蒸馏项的固定权重，或分段 schedule 的默认起点。")
    ap.add_argument(
        "--loss_schedule",
        type=str,
        default="piecewise_linear",
        choices=["none", "piecewise_linear"],
        help="loss 权重的 epoch 级调度方式；`piecewise_linear` 表示按 1 起始的 epoch 边界做分段线性切换。",
    )
    ap.add_argument("--loss_hold_epochs", type=int, default=40, help="前 N 个 epoch 保持起始 loss 权重（1 起始计数）。")
    ap.add_argument("--loss_ramp_start_epoch", type=int, default=45, help="loss 权重开始线性切换的 epoch（含，1 起始计数）。")
    ap.add_argument("--loss_ramp_end_epoch", type=int, default=55, help="loss 权重结束线性切换的 epoch（含，1 起始计数）。")
    ap.add_argument("--loss_cosine_w_start", type=float, default=1.0, help="piecewise schedule 中 cosine 项的起始权重。")
    ap.add_argument("--loss_cosine_w_end", type=float, default=0.1, help="piecewise schedule 中 cosine 项的结束权重。")
    ap.add_argument("--loss_mse_w_start", type=float, default=0.1, help="piecewise schedule 中 MSE 项的起始权重。")
    ap.add_argument("--loss_mse_w_end", type=float, default=1.0, help="piecewise schedule 中 MSE 项的结束权重。")
    ap.add_argument("--adapter_frames_chunk", type=int, default=2, help="Adapter 前向时按时间分块的帧数；越小越省显存，但速度更慢。")
    ap.add_argument(
        "--adapter_use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对 Adapter 前向启用 `torch.utils.checkpoint`；能省显存但会增加额外重算，可用 `--no-adapter_use_checkpoint` 关闭。",
    )

    ap.add_argument("--latent_chunk_len", type=int, default=3, help="窗口模式下每次从 latent 里取多少个时间步进行 decode/蒸馏。")
    # 兼容别名：有些 launch wrapper 会注入这个 flag。
    ap.add_argument("--min_latent_t", type=int, default=None, help="兼容旧启动脚本的别名；若设置，会覆盖 `latent_chunk_len`。")
    # 启用后，在不使用完整 latent 时为每个样本随机选择窗口起点。
    ap.add_argument("--latent_chunk_random", action=argparse.BooleanOptionalAction, default=False, help="窗口模式下是否为每个样本随机选择 latent 起点，增加时间位置扰动。")
    # 启用后，使用完整 latent 序列（不切窗口），确保所有时间片都参与蒸馏。
    ap.add_argument(
        "--latent_use_full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否直接使用完整 latent 序列而不切窗口；这样最直观，但显存占用更高，可用 `--no-latent_use_full` 关闭。",
    )
    # 启用后，枚举每条 latent 的所有窗口，不一次性 decode 全序列也能覆盖全部时间片。
    ap.add_argument("--latent_cover_all", action=argparse.BooleanOptionalAction, default=False, help="不全量 decode 时，是否枚举所有滑窗，保证每个时间片最终都被蒸馏到。")
    ap.add_argument("--latent_stride", type=int, default=1, help="latent 滑窗步长。")
    ap.add_argument("--latent_max_windows", type=int, default=0, help="每条样本最多使用多少个窗口；0 表示使用全部窗口。")

    ap.add_argument("--vae_disable_slicing", action="store_true", default=False, help="禁用 VAE slicing；更吃显存，但 hook 到的 batch 维行为更直观。")
    ap.add_argument("--vae_disable_tiling", action="store_true", default=False, help="禁用 VAE tiling；通常只在排查边界伪影或兼容性问题时使用。")
    ap.add_argument("--vae_num_sample_frames_batch_size", type=int, default=0, help="覆盖 VAE 内部一次 decode 的帧批大小；0 表示使用模型默认值。")

    ap.add_argument("--resume", type=str, default="", help="从已有的 Stage A Adapter checkpoint 恢复训练，例如 `stage1_adapter_last.pt`。")

    ap.add_argument("--export_combined", action="store_true", default=True, help="训练结束后是否额外导出组合 checkpoint，便于 Stage B 直接复用。")
    ap.add_argument("--export_name", type=str, default="infinitystar_up3_plus_adapter_latent2tokens.pt", help="导出的组合 checkpoint 文件名。")
    args = ap.parse_args()

    if args.min_latent_t is not None:
        args.latent_chunk_len = int(args.min_latent_t)

    if not str(args.out_dir).strip():
        raise ValueError("--out_dir 不能为空")
    os.makedirs(str(args.out_dir), exist_ok=True)

    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(int(args.seed) + int(get_rank()))

    # rank0 文件日志。
    global _RANK0_LOG_FH
    if get_rank() == 0:
        log_dir = str(args.log_dir).strip() or str(args.out_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(str(log_dir), str(args.log_file))
        global _RANK0_LOG_PATH, _RANK0_LOG_BROKEN
        _RANK0_LOG_PATH = log_path
        _RANK0_LOG_BROKEN = False
        _RANK0_LOG_FH = open(log_path, "ab", buffering=0)
        rank0_print(f"[日志] rank0 日志写入 {log_path}")
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
            raise ValueError("按 global_batch_size/world_size 算出的每张 GPU batch_size 小于 1")
        rank0_print(f"[batch] global_batch_size={args.global_batch_size} world_size={ws} per_gpu_batch={args.batch_size}")

    # 教师分支（冻结）：
    # 代码/形状说明：RGB frames -> TimesFormer patch_embed -> teacher tokens。
    # Stage A 只借用 TimesFormer 的视觉 token 空间，不更新 teacher 参数。
    tsformer = build_tsformer().to(device)
    load_tsformer(tsformer, str(args.tsformer_ckpt))
    tsformer.eval()
    for p in tsformer.parameters():
        p.requires_grad_(False)

    # 学生分支（可训练）：
    # 代码/形状说明：z_ext -> InfinityStar VAE decoder up_block_3 feature -> Adapter -> student tokens。
    # 只有 Adapter 是 Stage A 的主要训练目标。
    adapter = Vae96ToTSformerEmbedAdapter().to(device)
    adapter.train()
    if is_dist():
        adapter = DDP(adapter, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    # InfinityStar VAE（冻结）：
    # 用 decode hook 暴露中间特征，不把最终 RGB 作为训练目标。
    # 这对应“latent 不是拿来欣赏的图像，而是为动作服务的世界变化表征”。
    inf_root = str(args.infinitystar_root).strip() or None
    vae_model = load_infinitystar_vae(
        vae_path=str(args.infinitystar_vae_path),
        vae_type=int(args.infinitystar_vae_type),
        device=device,
        infinitystar_root=inf_root,
        proj_root=_ARCH_ROOT,
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

    class _VaeDecodeOnly(nn.Module):
        """只暴露 VAE decode 的薄封装，便于 DDP 包装和 hook 注册。"""

        def __init__(self, vae):
            """保存底层 InfinityStar VAE。"""
            super().__init__()
            self.vae = vae

        def forward(self, z_ext: torch.Tensor) -> torch.Tensor:
            """触发 VAE decode；训练真正使用的是 hook 捕获的中间 feature。"""
            return self.vae.decode(z_ext, return_dict=False)[0]

    vae_decode = _VaeDecodeOnly(vae_model)

    optimizer = torch.optim.AdamW(
        (adapter.module if isinstance(adapter, DDP) else adapter).parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=bool(args.amp) and torch.cuda.is_available())

    start_epoch = 0
    _resume_global_step = 0
    resume_path = str(args.resume).strip()
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"找不到 --resume 指定的 checkpoint：{resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(resume_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError(f"续训 checkpoint 必须是 dict，收到的是 {type(ckpt)}")
        ad_sd = ckpt.get("adapter_state_dict") or ckpt.get("adapter") or ckpt.get("state_dict")
        if ad_sd is None:
            raise ValueError("续训 checkpoint 缺少 adapter_state_dict")
        m_adapter = adapter.module if isinstance(adapter, DDP) else adapter
        missing, unexpected = m_adapter.load_state_dict(ad_sd, strict=False)
        if missing or unexpected:
            rank0_print(f"[续训] adapter 以 strict=False 加载：missing={missing[:10]} unexpected={unexpected[:10]}")
        opt_sd = ckpt.get("optimizer_state_dict") or ckpt.get("optimizer")
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
                for state in optimizer.state.values():
                    if isinstance(state, dict):
                        for k, v in list(state.items()):
                            if torch.is_tensor(v):
                                state[k] = v.to(device, non_blocking=True)
            except Exception as e:
                rank0_print(f"[续训] optimizer 状态加载失败，将从新的 optimizer 状态继续：{e}")
        # 用当前 --lr 覆盖学习率；load_state_dict 会恢复旧 LR。
        new_lr = float(args.lr)
        for pg in optimizer.param_groups:
            old_lr = pg.get("lr")
            pg["lr"] = new_lr
            if old_lr != new_lr:
                rank0_print(f"[续训] 使用当前 --lr 覆盖 optimizer LR：{old_lr} -> {new_lr}")
        scaler_sd = ckpt.get("scaler_state_dict")
        if scaler_sd is not None and scaler.is_enabled():
            try:
                scaler.load_state_dict(scaler_sd)
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0))
        _resume_global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
        rank0_print(f"[续训] 已从 {resume_path} 恢复：epoch={start_epoch} global_step={_resume_global_step}")

    # 数据集说明：
    # Stage A 同时需要 `z_ext` 和真实 RGB frames。
    # `z_ext` 产生 student tokens；`frames_rgb` 产生 teacher tokens。
    require_T = None if int(args.require_T) == 0 else int(args.require_T)
    png_tf = _build_png_transform()
    ws_root = str(args.workspace_root).strip() or _OPEN_ROOT
    ds_kwargs = dict(
        manifest_json=str(args.manifest_json),
        items_key=str(args.items_key),
        workspace_root=ws_root,
        transform=png_tf,
        load_frames=True,
        max_items=(int(args.max_items) if int(args.max_items) > 0 else None),
        require_T=require_T,
    )
    # PNG-teacher 需要读取 traj 来知道 T，并从 images_dir 加载 frames。
    # 有些环境里的 LatentTrajManifestDataset 可能较旧、没有这些 kwargs；这里优雅降级。
    try:
        ds = LatentTrajManifestDataset(
            **ds_kwargs,
            load_traj=True,
            io_timeout_s=60.0,
            on_error="empty",
        )
    except TypeError:
        rank0_print("[警告] 当前 LatentTrajManifestDataset 不支持 load_traj/io_timeout_s/on_error，已回退到旧版参数签名。")
        ds = LatentTrajManifestDataset(**ds_kwargs)
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if is_dist() else None
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.persistent_workers) if int(args.num_workers) > 0 else False,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        drop_last=True,
        collate_fn=partial(collate_fn, mode=str(args.collate_mode), latent_chunk_len=int(args.latent_chunk_len)),
    )

    expected_T = None if bool(args.latent_use_full) else _expected_frames_from_latent_chunk(int(args.latent_chunk_len))
    rank0_print(
        f"[stage1] items={len(ds)} batch_size={int(args.batch_size)} global_batch_size={int(args.batch_size) * get_world_size()}"
        f" latent_chunk_len={int(args.latent_chunk_len)}"
        f" latent_use_full={bool(args.latent_use_full)} latent_cover_all={bool(args.latent_cover_all)}"
        f" expected_T={expected_T}"
    )

    global_step = _resume_global_step
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_num = int(epoch) + 1
        w_cos_ep, w_mse_ep = _loss_weights_for_epoch(epoch_num, args)
        if get_rank() == 0:
            rank0_log(
                f"[loss_w] epoch={epoch_num}/{int(args.epochs)} schedule={str(args.loss_schedule)} "
                f"w_cos={w_cos_ep:.6f} w_mse={w_mse_ep:.6f} w_mean={float(args.loss_mean_w):.6f} w_std={float(args.loss_std_w):.6f}"
            )

        if sampler is not None:
            sampler.set_epoch(epoch)
        (adapter.module if isinstance(adapter, DDP) else adapter).train()

        running = torch.zeros((), device=device)
        running_parts: dict = {}
        nb = 0
        t_after_step = time.time()
        use_tqdm = bool(args.tqdm) and (get_rank() == 0) and (tqdm is not None)
        pbar = tqdm(total=len(dl), desc=f"stage1 第 {epoch+1}/{int(args.epochs)} 个 epoch", dynamic_ncols=True, leave=True) if use_tqdm else None

        for batch in dl:
            nb += 1
            global_step += 1
            data_s = time.time() - t_after_step
            t_step0 = time.time()

            optimizer.zero_grad(set_to_none=True)
            if global_step <= 3:
                rank0_log(f"[数据] 已取到 batch：step={global_step} data_s={data_s:.3f}")

            # 支持 per_sample collate，避免变长 T_lat 带来的 shape 问题。
            if isinstance(batch, dict) and "samples" in batch:
                samples = batch["samples"]
                if get_rank() == 0 and int(batch.get("skipped_no_frames", 0)) > 0 and (global_step % int(args.log_every) == 0):
                    rank0_log(
                        f"[dl_skip] step={global_step} skipped_no_frames={int(batch.get('skipped_no_frames',0))} "
                        f"skipped_no_latent={int(batch.get('skipped_no_latent',0))} total_in={int(batch.get('total_in',0))} "
                        f"total_out={int(batch.get('total_out',0))}"
                    )
            else:
                z_ext = batch["z_ext"].to(device, non_blocking=True)
                fr = batch.get("frames_rgb", None)
                if fr is None:
                    raise ValueError("batch 缺少 frames_rgb；请确认 dataset 设置了 load_frames=True")
                fr = fr.to(device, non_blocking=True)
                samples = [{"z_ext": z_ext[b : b + 1], "frames_rgb": fr[b : b + 1]} for b in range(int(z_ext.shape[0]))]

            loss_sum = torch.zeros((), device=device)
            step_parts: dict = {}
            valid = 0
            skip_short = 0
            skip_decode = 0
            skip_no_frames = 0
            did_backward = False
            backward_count = 0

            for i, s in enumerate(samples):
                z = s["z_ext"].to(device, non_blocking=True)
                fr_full = s.get("frames_rgb", None)
                if fr_full is None:
                    # 底层 dataset 为了容忍 I/O 失败（on_error="empty"）时，可能返回没有 frames 的样本。
                    # 这里跳过这些样本。
                    skip_no_frames += 1
                    continue
                fr_full = fr_full.to(device, non_blocking=True)
                # 统一单样本 frame tensor shape 为 (1,3,T,H,W)。
                if fr_full.ndim == 4:
                    fr_full = fr_full.unsqueeze(0)
                if fr_full.ndim != 5 or int(fr_full.shape[1]) != 3:
                    raise ValueError(f"frames_rgb 必须是 (3,T,H,W) 或 (1,3,T,H,W)，收到的是 {tuple(fr_full.shape)}")
                T_lat = int(z.shape[2])
                L = int(args.latent_chunk_len)
                # 保留 (latent_start, z_sub) 对，便于一致地裁剪 teacher 帧。
                z_sub_list: List[Tuple[int, torch.Tensor]] = []
                if bool(args.latent_use_full):
                    z_sub_list = [(0, z.contiguous())]
                elif bool(args.latent_cover_all):
                    if T_lat < L:
                        skip_short += 1
                        continue
                    stride = max(1, int(args.latent_stride))
                    max_s = T_lat - L
                    starts = list(range(0, max_s + 1, stride))
                    mw = int(args.latent_max_windows)
                    if mw > 0 and len(starts) > mw:
                        starts = starts[:mw]
                    z_sub_list = [(int(st), z[:, :, st : st + L].contiguous()) for st in starts]
                else:
                    if T_lat < L:
                        skip_short += 1
                        continue
                    max_s = T_lat - L
                    if bool(args.latent_chunk_random):
                        st = random.randint(0, max_s)
                    else:
                        st = 0
                    z_sub_list = [(int(st), z[:, :, st : st + L].contiguous())]

                for j, (st_lat, z_sub) in enumerate(z_sub_list):
                    if global_step <= 3 and i == 0 and j == 0:
                        rank0_log(f"[step] 解码前：step={global_step} z_sub={tuple(z_sub.shape)}")

                    # 为 latent window 裁剪 teacher 帧。
                    # Latent->frame 对齐（类 Wan 规则）：T_frames = 4*(T_lat-1)+1。
                    # 当窗口从 latent index st_lat>0 开始时，近似对应的帧起点为 4*st_lat-3。
                    # 这个值匹配前缀累计帧数。
                    if bool(args.latent_use_full):
                        fr_sub = fr_full
                    else:
                        if st_lat <= 0:
                            f0 = 0
                        else:
                            f0 = 4 * int(st_lat) - 3
                        # expected_T 是长度为 L 的 chunk 对应的帧长度（L=3 时默认 9）。
                        t_need = int(expected_T) if expected_T is not None else int(fr_full.shape[2])
                        f1 = min(int(fr_full.shape[2]), f0 + t_need)
                        fr_sub = fr_full[:, :, f0:f1].contiguous()

                    tok_s, tok_t, T = _decode_teacher_student_tokens(
                        vae_decode_module=vae_decode,
                        adapter=adapter,
                        tsformer=tsformer,
                        z_sub=z_sub,
                        frames_teacher=fr_sub,
                        amp_enabled=bool(args.amp),
                        adapter_frames_chunk=int(args.adapter_frames_chunk),
                        adapter_use_checkpoint=bool(args.adapter_use_checkpoint),
                        expected_T=expected_T,
                    )
                    if global_step <= 3 and i == 0 and j == 0:
                        rank0_log(f"[step] 解码后：step={global_step} decoded_T={int(T)}")
                    if int(T) <= 0:
                        skip_decode += 1
                        continue
                    loss_i, parts_i = compute_distill_loss(
                        tok_s, tok_t,
                        w_cos=float(w_cos_ep),
                        w_mean=float(args.loss_mean_w),
                        w_std=float(args.loss_std_w),
                        w_mse=float(w_mse_ep),
                    )
                    loss_sum = loss_sum + loss_i.detach()
                    for k, v in parts_i.items():
                        step_parts[k] = step_parts.get(k, 0.0) + v
                    valid += 1
                    # 立刻 backward，避免为所有样本/窗口长期保留计算图（显著节省显存）。
                    scaler.scale(loss_i).backward()
                    did_backward = True
                    backward_count += 1

            m_adapter = adapter.module if isinstance(adapter, DDP) else adapter
            if not did_backward:
                # 即使没有有效样本，也运行一次 backward（DDP-safe），并初始化 optimizer states（优化器状态）。
                # 这样会产生零梯度，但能让所有 rank 的 step 结构保持一致。
                s0 = None
                for p in m_adapter.parameters():
                    if p.requires_grad:
                        s0 = p.float().sum() if s0 is None else (s0 + p.float().sum())
                loss_batch = (s0 * 0.0) if s0 is not None else torch.zeros((), device=device)
                scaler.scale(loss_batch).backward()

            # 同一个 step 内多次 backward 后对梯度求平均，
            # 避免某个样本枚举多个窗口时放大更新幅度。
            if backward_count > 1:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                for p in m_adapter.parameters():
                    if p.grad is not None:
                        p.grad.div_(float(backward_count))

            if float(args.grad_clip) > 0:
                if scaler.is_enabled() and backward_count <= 1:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(m_adapter.parameters(), max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            loss_mean = reduce_mean(loss_sum / max(1, valid))
            running += loss_mean.detach()
            if valid > 0:
                for k in step_parts:
                    step_parts[k] /= valid
            for k, v in step_parts.items():
                running_parts[k] = running_parts.get(k, 0.0) + v

            step_s = time.time() - t_step0
            parts_str = " ".join(f"{k}={v:.5f}" for k, v in sorted(step_parts.items()) if k != "total")
            if pbar is not None:
                pbar.update(1)
                try:
                    pbar.set_postfix({
                        "loss": float(loss_mean.item()),
                        "cos": step_parts.get("cos", 0.0),
                        "avg": float((running / nb).item()),
                        "data_s": data_s,
                        "step_s": step_s,
                    })
                except Exception:
                    pass
            else:
                if (get_rank() == 0) and (int(args.log_every) > 0) and (global_step % int(args.log_every) == 0):
                    rank0_print(
                        f"stage1 e{epoch+1}/{int(args.epochs)} step={global_step} "
                        f"loss={float(loss_mean.item()):.6f} {parts_str} "
                        f"avg={float((running/nb).item()):.6f} data_s={data_s:.2f} step_s={step_s:.2f}"
                    )

            rank0_log(
                f"[step] epoch={epoch+1}/{int(args.epochs)} step={global_step} "
                f"loss={float(loss_mean.item()):.6f} {parts_str} "
                f"avg={float((running/nb).item()):.6f} "
                f"valid={valid} bw={backward_count} skip_short={skip_short} skip_decode={skip_decode} skip_no_frames={skip_no_frames} "
                f"data_s={data_s:.3f} step_s={step_s:.3f}"
            )

            t_after_step = time.time()

        if pbar is not None:
            pbar.close()

        # 保存 checkpoint。
        if (get_rank() == 0) and ((epoch + 1) % int(args.save_every) == 0):
            m_adapter = adapter.module if isinstance(adapter, DDP) else adapter
            opt_sd = optimizer.state_dict()
            state = {
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "adapter_state_dict": m_adapter.state_dict(),
                "optimizer_state_dict": opt_sd,
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "args": vars(args),
                # 向后兼容别名：旧 checkpoint 使用过这些 key。
                "adapter": m_adapter.state_dict(),
                "optimizer": opt_sd,
                "step": int(global_step),
            }
            path_e = os.path.join(str(args.out_dir), f"stage1_adapter_e{epoch+1}.pt")
            _atomic_torch_save(state, path_e)
            _atomic_torch_save(state, os.path.join(str(args.out_dir), "stage1_adapter_last.pt"))
            rank0_print(f"[保存] {path_e}")

            if bool(args.export_combined):
                export_path = os.path.join(str(args.out_dir), str(args.export_name))
                combined = {
                    "vae_state_dict": vae_model.state_dict(),
                    "adapter_state_dict": m_adapter.state_dict(),
                    "exported_at": datetime.now().isoformat(timespec="seconds"),
                    "args": vars(args),
                }
                _atomic_torch_save(combined, export_path)
                rank0_print(f"[export] {export_path}")

    if _RANK0_LOG_FH is not None:
        try:
            _RANK0_LOG_FH.close()
        except Exception:
            pass

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
