# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import json
import os
import random
import sys
import time
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from tap import Tap

import infinity.utils.dist as dist
from infinity.utils.sequence_parallel import SequenceParallelManager as sp_manager


class Args(Tap):
    # ==================================================================================================================
    # 中文说明：============================================= 路径与目录 ============================================
    # ==================================================================================================================
    """
    Infinity 训练 / 推理统一参数对象。

    小白阅读建议：
    先看这个类的字段分组，再去看具体脚本如何读取这些字段。很多脚本表面上入口不同，
    但最终都会落到这里的同一套参数命名上。
    """
    local_out_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # 中文说明：本地输出目录，主要用于保存 checkpoint、日志和中间产物
    data_path: str = ''  # 中文说明：图像数据集根路径
    video_data_path: str = ''  # 中文说明：视频数据集根路径
    bed: str = ''  # 中文说明：除 local_out_path 外额外复制 checkpoint 的目录
    vae_path: str = ''  # 中文说明：VAE checkpoint 路径
    log_txt_path: str = ''  # 中文说明：文本日志文件路径
    t5_path: str = ''  # 中文说明：T5 文本编码器路径；若不显式指定，程序会尽量自动寻找
    token_cache_dir: str = ''  # 中文说明：token cache 目录，常用于复用已提取的文本/视觉 token

    # ==================================================================================================================
    # 中文说明：=============================================== 通用训练参数 =================================================
    # ==================================================================================================================
    exp_name: str = ''  # 中文说明：实验名；通常也会体现在日志目录和 wandb run 名称中
    project_name: str = 'infinitystar'  # 中文说明：wandb 项目名
    tf32: bool = True  # 中文说明：是否启用 TensorFloat32；通常能在 A100 等设备上提升吞吐
    auto_resume: bool = True  # 中文说明：是否自动从最后一个 checkpoint 继续训练
    rush_resume: str = ''  # 中文说明：快速续训的 Infinity 预训练 checkpoint 路径
    rush_omnistore_resume: str = ''  # 中文说明：从 omnistore 版本 checkpoint 快速续训的路径
    torchshard_resume: str = ''      # 中文说明：torch shard checkpoint 的恢复路径
    log_every_iter: bool = False  # 中文说明：是否每个 iteration 都打印日志
    checkpoint_type: str = 'torch'  # 中文说明：checkpoint 格式类型；常见有 `torch`、`torch_shard`、`omnistore`
    device: str = 'cpu'  # 中文说明：训练使用的设备，例如 `cpu` 或 `cuda`
    is_master_node: bool = None  # 中文说明：当前节点是否为 master 节点
    epoch: int = 300  # 中文说明：总训练 epoch 数
    cur_epoch: int = 0  # 中文说明：运行时记录的当前 epoch 下标；用于数据打乱和日志，不表示总训练轮数
    log_freq: int = 1  # 中文说明：stdout 打印频率
    save_model_iters_freq: int = 1000  # 中文说明：按 iteration 保存模型的频率
    short_cap_prob: float = 0.2  # 中文说明：使用短 caption 训练的概率
    label_smooth: float = 0.0  # 中文说明：label smoothing 系数
    cfg: float = 0.1  # 中文说明：训练时 classifier-free guidance 的条件丢弃概率
    rand_uncond: bool = False  # 中文说明：是否使用随机、不可学习的 unconditional embedding
    twoclip_alternatingtraining: int = 0  # 中文说明：是否启用 two-clip 交替训练
    wp_it: int = 100  # 中文说明：warm-up iteration 数

    # ==================================================================================================================
    # 中文说明：===================================================== 模型结构 ======================================================
    # ==================================================================================================================
    model: str = ''  # 中文说明：模型类型；某些脚本会用 `'b'` 代表 VAE 训练，其它值通常走 GPT/Infinity 训练
    sdpa_mem: bool = True  # 中文说明：是否启用节省显存的 SDPA
    rms_norm: bool = False  # 中文说明：是否使用 RMSNorm
    tau: float = 1  # 中文说明：GPT 自注意力中的温度系数 tau
    tini: float = -1  # 中文说明：初始化相关超参数
    topp: float = 0.0                     # 中文说明：top-p 采样阈值
    topk: float = 0.0                     # 中文说明：top-k 采样阈值
    fused_norm: bool = False  # 中文说明：是否使用 fused normalization
    flash: bool = False  # 中文说明：是否启用自定义 flash-attention kernel
    use_flex_attn: bool = False  # 中文说明：是否使用 flex attention 加速训练
    norm_eps: float = 1e-6  # 中文说明：归一化层 epsilon
    Ct5: int = 2048  # 中文说明：文本编码器输出特征维度
    simple_text_proj: int = 1  # 中文说明：是否使用简单版文本投影层
    mask_type: str = 'infinity_elegant_clip20frames_v2'  # 中文说明：自注意力 mask 类型，例如 `var`、`video_tower` 或 Infinity 各种 schedule 专用变体
    mask_video_first_frame: int = 0  # 中文说明：计算 loss 时是否遮掉视频首帧

    use_fsdp_model_ema: int = 0  # 中文说明：是否在 FSDP 训练时启用模型 EMA
    model_ema_decay: float = 0.9999  # 中文说明：模型 EMA 衰减率；越接近 1，EMA 参数变化越平滑

    rope_type: str = '4d'  # 中文说明：RoPE 类型，可选 `2d`/`3d`/`4d`；决定位置编码覆盖的空间/时间维度
    rope2d_each_sa_layer: int = 1  # 中文说明：是否在每一层 self-attention 都应用 2D RoPE
    rope2d_normalized_by_hw: int = 2  # 中文说明：是否按目标高宽模板归一化 2D RoPE 坐标
    add_lvl_embeding_on_first_block: int = 0  # 中文说明：是否只在第一个 Transformer block 注入 level/scale embedding

    # ==================================================================================================================
    # 中文说明：================================================== 多尺度/分辨率调度 =============================================
    # ==================================================================================================================
    semantic_scales: int = 8  # 中文说明：语义尺度的层数/个数；越大表示 coarse-to-fine 里更强调语义层级
    semantic_scale_dim: int = 16  # 中文说明：语义尺度 embedding 的通道维度
    detail_scale_dim: int = 64  # 中文说明：细节尺度 embedding 的通道维度
    use_learnable_dim_proj: int = 0  # 中文说明：是否使用可学习的维度投影层
    detail_scale_min_tokens: int = 80  # 中文说明：细节尺度最少保留多少 token；避免最细层过早退化得太小
    pn: str = ''  # 中文说明：像素预算预设名，例如 `0.06M`、`0.25M`、`1M`；它是推导 scale schedule 的入口
    scale_schedule: tuple = None  # 代码/形状说明：运行时根据 `pn` 自动推导出的尺度计划，通常形如多级 `(t, h, w)` 网格
    patch_size: int = None  # 代码/形状说明：根据 `scale_schedule` 自动推导的 patch 大小
    dynamic_scale_schedule: str = ''  # 中文说明：视频动态尺度计划名称；决定每一级尺度怎样沿时间/空间展开
    min_scale_ind: int = 3  # 中文说明：Infinity frame pack 中允许参与训练/推理的最小尺度下标
    max_reweight_value: int = 40  # 中文说明：按尺度重加权 loss 时的截断上限，避免个别尺度权重过大
    image_scale_repetition: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'  # 中文说明：图像路径中各尺度重复采样/重复训练的次数配置
    video_scale_repetition: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'  # 中文说明：视频路径中各尺度重复采样/重复训练的次数配置
    inner_scale_boost: int = 0  # 中文说明：是否额外放大中间尺度的重要性；常用于让过渡尺度学得更稳
    drop_720p_last_scale: int = 1  # 中文说明：720p 路径下是否丢掉最后一个最细尺度，以控制 token 数和显存开销
    reweight_loss_by_scale: int = 0  # 中文说明：是否按尺度对 loss 重新加权，而不是所有尺度一视同仁

    # ==================================================================================================================
    # 中文说明：================================================== 优化器与训练稳定性 ==================================================
    # ==================================================================================================================
    tlr: float = 2e-5  # 中文说明：基础学习率
    grad_clip: float = 5  # 中文说明：梯度裁剪阈值；超过该范数时会被截断
    cdec: bool = False  # 中文说明：是否随训练过程逐步衰减梯度裁剪阈值
    opt: str = 'adamw'  # 中文说明：优化器类型，可选 `adamw` 或 `lion`
    ada: str = '0.9_0.97'  # 中文说明：Adam/Lion 的 beta 参数字符串，例如 `0.9_0.999`
    adam_eps: float = 0.0  # 中文说明：Adam 的 epsilon
    fused_adam: bool = True  # 中文说明：是否使用 fused Adam 优化器
    disable_weight_decay: int = 1  # 中文说明：是否对稀疏参数禁用 weight decay，避免这类参数被过度正则
    fp16: int = 2  # 中文说明：浮点精度模式：1 表示 fp16，2 表示 bf16

    # ==================================================================================================================
    # 中文说明：====================================================== 数据读取 ======================================================
    # ==================================================================================================================
    video_fps: int = 16  # 中文说明：视频帧率
    video_frames: int = 81  # 中文说明：每段视频采样的目标帧数
    video_sample_mode: str = 'duration'  # 中文说明：视频采样模式。`duration` 是当前默认行为；`fixed_full` 表示在整段视频上均匀采样固定帧数；`segment_full` 表示尽量保留完整片段并满足 4n+1 规则；`segment_full_floor_sec` 表示按整秒长度向下取整后再做 1+fps*n 对齐
    video_batch_size: int = 1  # 中文说明：视频数据 batch size
    workers: int = 16  # 中文说明：dataloader worker 数
    image_batch_size: int = 0  # 中文说明：每张 GPU 的图像 batch size；运行时会自动设置
    ac: int = 1  # 中文说明：梯度累积步数
    r_accu: float = 1.0  # 中文说明：梯度累积步数的倒数；运行时自动设置
    tlen: int = 512  # 中文说明：文本特征最长截断到多少 token
    num_of_label_value: int = 2  # 中文说明：每个 label 的取值数；2 常表示 bitwise 二值标签，0 常表示 index-wise 路径
    dynamic_resolution_across_gpus: int = 1  # 中文说明：是否允许不同 GPU 处理不同动态分辨率
    enable_dynamic_length_prompt: int = 0  # 中文说明：训练时是否启用动态长度 prompt
    use_streaming_dataset: int = 0  # 中文说明：是否使用 streaming dataset
    iterable_data_buffersize: int = 90000  # 中文说明：streaming dataset 的缓冲区大小
    image_batches_multiply: float = 1.0  # 中文说明：每个 epoch 中图像 batch 数的倍率
    down_size_limit: int = 10000  # 中文说明：视频下载大小上限，单位 MB
    addition_pn_list: str = '[]'  # 中文说明：额外启用的像素预算列表
    video_caption_type: str = 'tarsier2_caption'  # 中文说明：视频 caption 字段类型
    only_images4extract_feats: int = 0  # 中文说明：是否只为图像提取特征
    train_max_token_len: int = -1  # 中文说明：训练时允许的最大 token 长度
    train_with_var_seq_len: int = 0  # 中文说明：是否使用变长序列训练
    video_var_len_prob: str = '[30, 30, 30, 5, 3, 2]'  # 中文说明：变长视频采样的概率分布
    duration_resolution: int = 1  # 中文说明：duration 分辨率
    seq_pack_bucket: int = 1000  # 中文说明：sequence packing 的 bucket 大小
    drop_long_video: int = 0  # 中文说明：是否丢弃过长视频
    min_video_frames: int = -1  # 中文说明：最少视频帧数
    restrict_data_size: int = -1  # 中文说明：限制数据集规模
    allow_less_one_elem_in_seq: int = 0  # 中文说明：是否允许少于一个有效元素的序列
    train_192pshort: int = 0  # 中文说明：是否训练 192p short video 路径
    steps_per_frame: int = 3  # 中文说明：video tower 每帧对应的步数
    add_motion_score2caption: int = 0  # 中文说明：是否把 motion score 前置到 caption
    context_frames: int = 10000  # 中文说明：video tower 的上下文帧数
    cached_video_frames: int = 81  # 中文说明：缓存视频帧数
    frames_inner_clip: int = 20  # 中文说明：Infinity frame pack 中每个 clip 的帧数
    context_interval: int = 2  # 中文说明：上下文采样间隔
    context_from_largest_no: int = 1  # 中文说明：是否优先从最大编号上下文采样
    append_duration2caption: int = 0  # 中文说明：是否把时长标签追加到 caption
    cache_check_mode: int = 0  # 中文说明：cache 检查模式
    online_t5: bool = True  # 中文说明：是否在线运行 T5，而不是加载本地缓存特征

    # ==================================================================================================================
    # 中文说明：============================================= 分布式训练 ===============================================
    # ==================================================================================================================
    enable_hybrid_shard: bool = False  # 中文说明：是否启用 hybrid shard/FSDP
    inner_shard_degree: int = 8  # 中文说明：FSDP 内部 shard 分组大小（inner shard degree）
    zero: int = 0  # 中文说明：DeepSpeed ZeRO stage
    buck: str = 'chunk'  # 中文说明：FSDP 的 module bucketing 方式
    fsdp_orig: bool = True  # 中文说明：是否使用原版 FSDP 实现
    enable_checkpointing: str = None  # 中文说明：激活重计算（activation checkpointing）策略，例如 `full-block` 或 `self-attn`
    pad_to_multiplier: int = 128  # 中文说明：把序列长度 pad 到该倍数，便于并行和 kernel 对齐
    sp_size: int = 0  # 中文说明：sequence parallel 的并行度
    fsdp_save_flatten_model: int = 1  # 中文说明：FSDP 下是否保存 flatten 后的模型
    inject_sync: int = 0  # 中文说明：是否注入额外同步逻辑
    model_init_device: str = 'cuda'  # 中文说明：模型初始化设备
    fsdp_init_device: str = 'cuda'  # 中文说明：FSDP 初始化设备

    # ==================================================================================================================
    # ======================================================= VAE ======================================================
    # ==================================================================================================================
    vae_type: int = 64  # 中文说明：VAE 类型；例如 16/32/64 常对应不同的 BSQ VAE quant bits
    fake_vae_input: bool = False  # 中文说明：调试时是否使用伪造的 VAE 输入
    use_slice: int = 1  # 中文说明：VAE 编码时是否启用 slicing，以节省显存
    use_vae_token_cache: int = 1  # 中文说明：是否复用 VAE token cache
    save_vae_token_cache: int = 0  # 中文说明：是否把 VAE token cache 持久化到磁盘
    allow_online_vae_feature_extraction: int = 1  # 中文说明：是否允许在线提取 VAE 特征
    use_text_token_cache: int = 0  # 中文说明：是否使用文本 token cache
    videovae: int = 10  # 中文说明：是否启用视频 VAE 路径
    use_feat_proj: int = 2  # 中文说明：是否启用特征投影层
    use_two_stage_lfq: int = 0  # 中文说明：是否启用两阶段 LFQ
    casual_multi_scale: int = 0  # 中文说明：是否启用 causal multi-scale
    temporal_compress_rate: int = 4  # 中文说明：时间压缩率；例如 4 表示 4 帧压成 1 个 latent 时间步
    apply_spatial_patchify: int = 0  # 中文说明：是否启用空间 patchify，把 2x2 空间块折到通道维


    # ==================================================================================================================
    # 中文说明：============================================ Bitwise 自校正 =============================================
    # ==================================================================================================================
    noise_apply_layers: int = 1000  # 中文说明：把噪声注入到哪些层；常用于 bitwise self-correction 调试或鲁棒性实验
    noise_apply_strength: str = '-1'  # 中文说明：噪声强度配置；通常按字符串解析成分层/分尺度参数
    noise_apply_requant: int = 1  # 中文说明：注入噪声后是否重新量化到离散 token/bit 空间
    noise_apply_random_one: int = 0  # 中文说明：是否只随机挑一个尺度执行“加噪 + requant”
    debug_bsc: int = 0  # 中文说明：是否保存调试图并设置断点，用于分析 BSC 过程
    noise_input: int = 0  # 中文说明：是否直接对输入 token/latent 加噪，而不是只扰动中间层
    reduce_accumulate_error_method: str = 'bsc'  # 中文说明：累计误差抑制方法，默认是 `bsc`



    ############################ 中文说明：注意：下面这些参数和配置会在运行时自动设置，第一遍阅读可以先跳过 ###############################
    ############################ 中文说明：注意：下面这些参数和配置会在运行时自动设置，第一遍阅读可以先跳过 ###############################
    ############################ 中文说明：注意：下面这些参数和配置会在运行时自动设置，第一遍阅读可以先跳过 ###############################


    # 会在运行时自动设置
    branch: str = '' # 中文说明：当前 git 分支名；运行时自动设置，不需要手动传入
    commit_id: str = '' # 中文说明：当前 git commit id；运行时自动设置
    commit_msg: str = ''# 中文说明：当前 git commit message 最后一行；运行时自动设置
    cmd: str = ' '.join(a.replace('--exp_name=', '').replace('--exp_name ', '') for a in sys.argv[7:])  # 中文说明：本次运行的命令行摘要；自动设置
    tag: str = 'UK'                     # 中文说明：运行标签；自动设置
    cur_it: str = ''                    # 中文说明：当前 iteration 记录字段；自动设置
    MFU: float = None                   # 中文说明：Model FLOPs Utilization；自动统计
    HFU: float = None                   # 中文说明：Hardware FLOPs Utilization；自动统计
    # ==================================================================================================================
    # 中文说明：======================== 下面这些参数主要服务调试，第一遍阅读可以先跳过 ==============================
    # ==================================================================================================================

    dbg: bool = 'KEVIN_LOCAL' in os.environ       # 中文说明：仅在排查 DDP unused parameter 等本地调试问题时使用
    prof: int = 0           # 中文说明：是否开启 profile
    prof_freq: int = 50     # 中文说明：profile 采样频率
    profall: int = 0
    # ==================================================================================================================
    # 中文说明：======================== 上面这些参数主要服务调试，第一遍阅读可以先跳过 ==============================
    # ==================================================================================================================

    @property
    def gpt_training(self):
        """是否处于 GPT/Infinity 训练路径；通常 `model` 非空时为真。"""
        return len(self.model) > 0

    def set_initial_seed(self, benchmark: bool):
        """
        设置随机种子和 cudnn 相关选项，尽量保证数据顺序、初始化和采样过程可复现。

        注意：
        这里做的是“尽量可复现”，不是数学上绝对一致；分布式、混精和不同 kernel 仍可能带来细微差异。
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        assert self.seed
        seed = self.seed
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def dump_log(self):
        """把本次运行的关键参数写入 `log_txt_path`，方便复现实验。

        新手提示：这里只写轻量的 JSON 摘要，不保存模型权重；checkpoint 保存由训练器负责。
        """
        if not dist.is_local_master():
            return
        nd = {'is_master': dist.is_visualizer()}
        for k, v in {
            'name': self.exp_name,
            'tag': self.tag,
            'cmd': self.cmd,
            'commit': self.commit_id,
            'branch': self.branch,
            'cur_it': self.cur_it,
            'last_upd': time.strftime("%Y-%m-%d %H:%M", time.localtime()),
            'opt': self.opt,
            'is_master_node': self.is_master_node,
        }.items():
            if hasattr(v, 'item'):v = v.item()
            if v is None or (isinstance(v, str) and len(v) == 0): continue
            nd[k] = v

        with open(self.log_txt_path, 'w') as fp:
            json.dump(nd, fp, indent=2)

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        """中文说明：`state_dict` 导出当前参数对象的可序列化字段，供日志、checkpoint 或复现实验时保存配置快照。

        新手提示：这里不会保存整个训练状态，只会把 `Args` 里可序列化的字段整理成 dict。
        阅读重点：看哪些字段被排除（如 `device`），以及调用方把这份配置快照写到哪里。
        """
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # 中文说明：这些字段不可 JSON 序列化，不能直接写入配置快照
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        """中文说明：`load_state_dict` 把保存下来的参数快照重新灌回 `Args` 对象。

        新手提示：输入既可以是 dict，也兼容旧版本遗留的字符串表示。
        阅读重点：它只恢复字段值，不负责重新初始化进程组、模型或 dataloader。
        """
        if isinstance(d, str):  # 中文说明：兼容旧版本把参数快照保存成字符串的格式
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            if k in {'is_large_model', 'gpt_training'}:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e

    @staticmethod
    def set_tf32(tf32: bool):
        """中文说明：`set_tf32` 统一设置 cuDNN / matmul 的 TF32 开关。

        新手提示：TF32 只影响支持该特性的 CUDA 设备，CPU 路径不会生效。
        阅读重点：它会同时改 `cudnn.allow_tf32`、`matmul.allow_tf32`，并打印当前状态。
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')

    def __str__(self):
        """中文说明：`__str__` 把当前参数对象格式化成人类可读的多行字符串。

        新手提示：这主要用于日志打印，方便在启动时快速核对最终生效的参数。
        阅读重点：它和 `state_dict()` 一样会跳过不可序列化字段。
        """
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # 中文说明：这些字段不可 JSON 序列化，日志里跳过
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    """中文说明：`init_dist_and_get_args` 负责解析命令行参数、初始化分布式环境，并返回整理好的 `Args`。

    新手提示：这是训练入口最常见的第一站，后面的 dataset/model/trainer 构建都会依赖这里产出的 `args`。
    阅读重点：它除了 parse args，还会处理 `local_rank`、extra args、随机种子和分布式初始化副作用。
    """
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)

    if len(args.extra_args) > 0 and args.is_master_node == 0:
        print(f'======================================================================================')
        print(f'=========================== 警告：发现未识别的额外参数 ===========================\n{args.extra_args}')
        print(f'=========================== 警告：发现未识别的额外参数 ===========================')
        print(f'======================================================================================\n\n')

    args.set_tf32(args.tf32)

    try: os.makedirs(args.bed, exist_ok=True)
    except: pass
    try: os.makedirs(args.local_out_path, exist_ok=True)
    except: pass

    dist.init_distributed_mode(local_out_path=args.local_out_path, fork=False, timeout_minutes=30)
    args.device = dist.get_device()

    # 同步随机种子
    args.seed = int(time.time())
    seed = torch.tensor([args.seed], device=args.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(seed, op=torch.distributed.ReduceOp.MIN)
    args.seed = seed.item()

    if args.sp_size > 1:
        print(f"信息：sp_size={args.sp_size}")
        sp_manager.init_sp(args.sp_size)


    args.r_accu = 1 / args.ac   # 中文说明：梯度累积系数的倒数，后续用于把 loss 按累积步数缩放
    args.ada = args.ada or ('0.9_0.96' if args.gpt_training else '0.5_0.9')
    args.opt = args.opt.lower().strip()

    # 模型参数：GPT
    if args.gpt_training:
        assert args.vae_path, '训练 GPT/Infinity 时必须指定 VAE checkpoint 路径'
        from infinity.models import alias_dict
        if args.model in alias_dict:
            args.model = alias_dict[args.model]

    args.log_txt_path = os.path.join(args.local_out_path, 'log.txt')

    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, "0"] else args.enable_checkpointing
    args.enable_checkpointing = "full-block" if args.enable_checkpointing in [True, 1, "1"] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, "full-block", "full-attn", "self-attn"], \
        f"只支持不开启 checkpointing，或使用 full-block/full-attn/self-attn；当前收到 {args.enable_checkpointing}。"

    if len(args.exp_name) == 0:
        args.exp_name = os.path.basename(args.bed) or 'test_exp'

    if '-' in args.exp_name:
        args.tag, args.exp_name = args.exp_name.split('-', maxsplit=1)
    else:
        args.tag = 'UK'

    if dist.is_master():
        os.system(f'rm -rf {os.path.join(args.bed, "ready-node*")} {os.path.join(args.local_out_path, "ready-node*")}')

    if args.sdpa_mem:
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)
    print(args)
    if isinstance(args.noise_apply_strength, str):
        args.noise_apply_strength = list(map(float, args.noise_apply_strength.split(',')))
    elif isinstance(args.noise_apply_strength, float):
        args.noise_apply_strength = [args.noise_apply_strength]
    return args
