# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
"""
WorldVLN 世界模型的 Stage-1 监督微调入口。

中文导读：
这个文件负责把视频生成骨干迁移到导航数据上。训练脚本会把语言指令编码为文本条件，
把视频帧经 InfinityStar video VAE 编成 latent token，然后训练 causal/VAR transformer
在真实历史 latent 上预测下一段 latent 世界状态变化。

读代码时建议先看 `build_everything_from_args()`：它串起 T5、VAE、GPT/Infinity transformer、
optimizer 和 trainer；真正的训练循环在 `infinity.trainer.sft_trainer` 中。
"""
import gc
import json
import math
import os
import os.path as osp
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
# 代码/形状说明：os.environ["TORCH_LOGS"] = "+dynamo"
# 代码/形状说明：os.environ["TORCHDYNAMO_VERBOSE"] = '1'

import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 64
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.dataset.build import build_joint_dataset
from infinity.utils.save_and_load import CKPTSaver, omnistoreCheckpoint, auto_resume, omnistore_auto_resume
from infinity.models.ema import get_ema_model
from infinity.utils import arg_util, misc, wandb_utils
from infinity.trainer import get_trainer
# 代码/形状说明：from infinity.utils.mfu.mfu import mfutool

def build_everything_from_args(args: arg_util.Args, saver):
    """
    根据解析后的 args 构建 Stage-1 监督训练所需的全部组件。

    中文导读：
    这里是 Stage 1 的组装点：文本编码器负责语言条件，VAE 负责把导航视频压成 latent，
    GPT/Infinity transformer 负责自回归预测 latent，Trainer 负责把这些部件接到监督训练循环。
    """
    # 设置随机种子，保证分布式训练各 rank 的初始化和数据采样可复现。
    args.set_initial_seed(benchmark=True)
    # 构建文本 tokenizer / encoder：导航指令会被编码成世界模型的文本条件。
    print(f'正在从 {args.t5_path} 加载 T5...')
    if 'flan-t5' in args.t5_path:
        from transformers import T5EncoderModel, T5TokenizerFast
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True) # text_tokenizer.model_max_length 默认为 512
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        args.text_tokenizer_type = 'flan_t5'
        args.text_tokenizer = text_tokenizer
    else: # umt5 分支目前未启用。
        raise ValueError("当前 Stage-1 训练入口仅支持 flan-t5 文本编码器。")

    # 构建模型。这里的 gpt 不是纯文本 GPT，而是带文本条件的 causal/VAR 世界模型，
    # 目标是按多尺度 schedule 预测下一段视频 latent。
    vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args)

    # 注意：InfinityTrainer 依赖较重；这里延迟导入，避免训练启动阶段不必要的显存/内存峰值。
    InfinityTrainer = get_trainer(args)
    # 构建 trainer：它把 VAE、世界模型、optimizer 和损失计算封装到 train_step()。
    trainer = InfinityTrainer(
        device=args.device,
        raw_scale_schedule=args.scale_schedule,
        vae_local=vae_local,
        gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp,
        gpt_opt=gpt_optim,
        label_smooth=args.label_smooth,
        zero=args.zero,
        vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale,
        gpt_wo_ddp_ema=gpt_wo_ddp_ema,
        gpt_ema=gpt_ddp_ema,
        use_fsdp_model_ema=args.use_fsdp_model_ema,
        other_args=args,
    )

    # 从中断的实验自动 resume。
    global_it = 0
    if args.checkpoint_type == 'torch':
        auto_resume_info, start_ep, global_it, acc_str, _, trainer_state, _ = auto_resume(args, 'global_step_*')
        if trainer_state is not None and len(trainer_state):
            trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)
    elif args.checkpoint_type == 'omnistore':
        resume_path, info = omnistore_auto_resume(args, 'global_step_*')
        if not resume_path and args.rush_omnistore_resume:
            resume_path = args.rush_omnistore_resume
        if resume_path:
            print(f"omnistore 从断点恢复：{resume_path}", flush=True)
            args_state, start_ep, start_it, global_it, acc_str, eval_milestone = saver.load(resume_path, fsdp_object=trainer.gpt, optimizer_object=trainer.gpt_opt.optimizer)
            dist.barrier()
        if args.rush_omnistore_resume == resume_path:
            global_it = 0
        auto_resume_info, acc_str, eval_milestone, trainer_state, args_state =  info, '[no acc str]', [], {}, {}

    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return text_tokenizer, text_encoder, trainer, global_it


def build_model_optimizer(args):
    """
    构建 VAE、world-model transformer 和分布式 optimizer。

    中文导读：
    `gpt_wo_ddp` 是这里的世界模型主体，不是传统文本 GPT；它消费文本条件和视频 latent
    token，学习按 schedule 预测后续 latent scale。FSDP/ZeRO 相关代码只负责大模型分布式训练。
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.models.infinity import Infinity, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt

    # 关闭 PyTorch Linear/LayerNorm 默认初始化，加快大模型构建；后面会调用自定义 init_weights。
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    count_p = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    num_para = count_p(gpt_wo_ddp)
    if num_para/1000 < 20: # 小于 20B 参数时直接放到 cuda。
        gpt_wo_ddp = gpt_wo_ddp.to('cuda')

    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init()
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None

    if args.rush_resume:
        print(f"{args.rush_resume=}")
        cpu_d = torch.load(args.rush_resume, 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            """rush_resume 时丢弃 shape 不匹配的旧权重，允许从相近结构继续微调。"""
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                print(f'[rush_resume] 丢弃 word_embed.weight')
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                print(f'[rush_resume] 丢弃 head.weight')
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                print(f'[rush_resume] 丢弃 head.bias')
                del state_dict['head.bias']
            if 'text_proj_for_sos.ca.mat_kv.weight' in state_dict and \
                (state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape):
                print(f'[rush_resume] 丢弃 cfg_uncond')
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            if 'semantic_head.weight' in state_dict:
                print(f'[rush_resume] 用 semantic_head2 替换 semantic_head')
                state_dict['semantic_head2.weight'] = state_dict['semantic_head.weight']
                state_dict['semantic_head2.bias'] = state_dict['semantic_head.bias']
                del state_dict['semantic_head.weight']
                del state_dict['semantic_head.bias']
            if 'semantic_head2.weight' in state_dict and (state_dict['semantic_head2.weight'].shape[0] != gpt_wo_ddp.semantic_head2.out_features):
                print(f'[rush_resume] 丢弃 semantic_head2.weight 和 semantic_head2.bias')
                del state_dict['semantic_head2.weight']
                del state_dict['semantic_head2.bias']
            return state_dict
        print(gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False))
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)
    elif args.torchshard_resume:
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(gpt_wo_ddp, args.torchshard_resume, strict=False)

    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}

    print(f'[PT] GPT 模型 = {gpt_wo_ddp}\n\n')
    print(f'[PT][#para], GPT={num_para:.2f}\n\n')

    gpt_uncompiled = gpt_wo_ddp

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # 使用混合精度；背景见 https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # 没有 block chunks。
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])

        if args.enable_hybrid_shard:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP 初始化：{args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)

        if args.fsdp_init_device == 'cpu':
            gpt_wo_ddp = gpt_wo_ddp.cpu()

        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp,
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy,
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)

        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema,
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy,
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy,
                use_orig_params=args.fsdp_orig,
                sync_module_states=True,
                limit_all_gathers=True,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== 构建 optimizer ===============
    nowd_keys = set()
    if args.disable_weight_decay:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1

    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.fused_adam),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.fused_adam),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.adam_eps: opt_kw['eps'] = args.adam_eps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.grad_clip, args.zero)
    del names, paras, para_groups
    return vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim


def build_dataset(args):
    """
    构建 Stage-1 监督训练数据集。

    返回的 dataset 不是普通 image-label 样本，而是导航视频训练样本：
    `images` 是可选 RGB 帧，`raw_features_bcthw` 是可选预缓存 VAE latent，
    `captions` 是语言指令，`media`/`feature_cache_files4images` 用于 trainer 判断样本来源和缓存。
    这里保持 `load_vae_instead_of_image=False`，表示是否使用预缓存 latent 由 dataset 内部按样本决定。
    """
    train_dataset = build_joint_dataset(
        args,
        args.data_path,
        args.video_data_path,
        max_caption_len=args.tlen,
        short_prob=args.short_cap_prob,
        load_vae_instead_of_image=False
    )
    return train_dataset

def main_train(args: arg_util.Args):
    """
    Stage-1 主训练循环。

    这个函数只负责编排：checkpoint saver、模型组件、dataset/dataloader、epoch loop。
    反向传播和损失细节在 `trainer.train_step()` 内部；读训练主线时可以先把这里看成
    “每个 iteration 准备语言条件 + 视频 latent，再交给 trainer”。
    """
    # checkpoint_type 决定保存/恢复格式：
    # - torch: 普通 .pth 风格，便于单机或简单调试；
    # - omnistore: 大规模分布式训练环境的对象存储 checkpoint。
    if args.checkpoint_type == 'torch':
        saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    elif args.checkpoint_type == 'omnistore':
        saver = omnistoreCheckpoint(eval_milestone=None)
    else:
        raise ValueError(f"不支持的 checkpoint_type：{args.checkpoint_type}")
    ret = build_everything_from_args(args, saver)

    if ret is None:
        return

    text_tokenizer, text_encoder, trainer, start_global_it = ret
    gc.collect(), torch.cuda.empty_cache()
    seg5 = np.linspace(1, args.epoch, 5+1, dtype=int).tolist()

    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.epoch // 10) if args.epoch <= 100 else max(1, args.epoch // 20)

    # ============================================= epoch 循环开始 =============================================
    # 主进程初始化 wandb 日志。
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})
    total_epochs = int(args.epoch)
    for ep in range(total_epochs):
        # 每个 epoch 重新构建 dataset，是为了让 worker 重新读取 meta/采样配置。
        # 注意：args.epoch 始终表示总 epoch 数；当前 epoch 写到 args.cur_epoch，供 dataset 做采样策略。
        args.cur_epoch = ep

        if ep == 0:
            train_dataset = build_dataset(args)
            iters_train = len(train_dataset)
            # 未显式指定保存间隔时，默认一轮保存一次；这样 resume 时 next_ep/next_it 语义最直观。
            if int(getattr(args, "save_model_iters_freq", 0)) <= 0:
                args.save_model_iters_freq = int(iters_train)
                print(f'[PT 信息] 自动设置 save_model_iters_freq={args.save_model_iters_freq}（每个 epoch 保存一次）')
            start_ep = start_global_it // iters_train
            start_it = start_global_it % iters_train
            print(f'[PT 信息] 从 ep{start_ep} it{start_it} 继续，{iters_train=} =======> bed: {args.bed} <=======\n')

        if ep < start_ep:
            continue
        if ep > start_ep:
            train_dataset = build_dataset(args)
            iters_train = len(train_dataset)
            if int(getattr(args, "save_model_iters_freq", 0)) <= 0:
                args.save_model_iters_freq = int(iters_train)
                print(f'[PT 信息] 自动设置 save_model_iters_freq={args.save_model_iters_freq}（每个 epoch 保存一次）')

        # [训练一个 epoch]
        # 到这里为止，准备逻辑都结束了：
        # - build_everything_from_args() 负责“模型/optimizer/trainer”；
        # - build_dataset() 负责“样本从哪里来、每个样本带哪些字段”；
        # - train_one_epoch() 负责“把字段送进 trainer.train_step()”。
        # dataset 已经负责 batching/拼样本，所以这里 batch_size=None。
        # DataLoader 只承担 worker 并行、pin memory 和迭代器封装。
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.workers, pin_memory=True, batch_size=None)
        stats = train_one_epoch(
            epoch=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            start_global_it=start_global_it,
            me=None,
            saver=saver,
            args=args,
            dataloader_iter=iter(train_dataloader),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
        )

        del stats, train_dataset, train_dataloader
    return


g_speed_ls = deque(maxlen=128)
def train_one_epoch(
    epoch: int, is_first_ep: bool, start_it: int, start_global_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, dataloader_iter, iters_train: int,
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer,
):
    """
    执行一个 epoch 的 Stage-1 SFT。

    输入 batch 的关键字段：
    - `captions`: 语言导航指令，会被 T5 编成 compact KV 条件；
    - `images`: 原始 RGB 帧列表，供 trainer 内部 VAE 编码；
    - `raw_features_bcthw`: 已预缓存的视频 latent，能绕过重复 VAE 编码；
    - `feature_cache_files4images`: RGB 对应的 cache 路径，trainer 可按需写回/读取；
    - `media`: 样本媒体类型或来源标记，用于多数据集混训统计。

    `stepping` 由梯度累积间隔 `args.ac` 决定：不是每个 micro-batch 都会 optimizer.step()。
    """
    # 注意：重依赖尽量放在 DataLoader 之后，避免 worker/主进程初始化时出现额外 OOM。
    step_cnt = 0
    header = f'[Ep]: [{epoch:4d}/{args.epoch}]'

    last_touch = time.time()
    g_it, max_it = epoch * iters_train, args.epoch * iters_train

    doing_profiling = args.prof and epoch == 0 and (args.profall or dist.is_master())
    maybe_record_function = record_function if doing_profiling else nullcontext
    trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function

    last_t_perf = time.time()
    speed_ls: deque = g_speed_ls
    FREQ = min(args.prof_freq, iters_train//2-1)
    NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
    ranges = set([2 ** i for i in range(20)])
    if epoch <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
    PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

    me = misc.MetricLogger()
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['L', 'L_i', 'L_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Acc', 'Acc_i', 'Acc_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['seq_usage']]
    # ============================================= iteration 循环开始 =============================================
    for it, data in me.log_every(start_it, iters_train, dataloader_iter, args.log_freq, args.log_every_iter, header, args):
        g_it = epoch * iters_train + it
        # 中文说明：mfutool.step()
        # mfu_val = mfutool.get_mfu() * 100 # 转成百分比。
        # 代码/形状说明：print(f"[MFU] step={g_it}, mfu={mfu_val:.2f} %, mfu.iter_time = {mfutool.iter_time():.4f} s")


        if (it+1) % FREQ == 0:
            speed_ls.append((time.time() - last_t_perf) / FREQ)
            last_t_perf = time.time()

        if (g_it+1) % args.save_model_iters_freq == 0:
            if args.checkpoint_type == 'torch':
                saver.sav(args=args, g_it=(g_it+1), next_ep=epoch, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
            elif args.checkpoint_type == 'omnistore':
                saver.sav(args=args, global_it=(g_it+1), next_ep=epoch, next_it=it+1, fsdp_object=trainer.gpt, optimizer_object=trainer.gpt_opt.optimizer, acc_str=None, eval_milestone=None)

        with maybe_record_function('before_train'):
            # [取数据]
            # 这里的 data 已经是 dataset 组好的一个训练 item/batch。
            # 同一批里可能既有 RGB，也有预缓存 latent；trainer.train_step() 会按非空字段选择路径。
            images, captions, raw_features_bcthw, feature_cache_files4images, media = data['images'], data['captions'], data['raw_features_bcthw'], data['feature_cache_files4images'], data['media']

            # [准备文本特征]
            # T5 输出是 padded batch `(B,L,C)`；Infinity attention 更偏好 compact 形式：
            # - kv_compact: 把每条 caption 的有效 token 串接成 `(sum_lens,C)`；
            # - lens/cu_seqlens_k: 记录每条样本的边界，供 flash-attn/varlen attention 使用；
            # - Ltext: 当前 batch 最大文本长度，用于 shape/位置编码相关逻辑。
            if args.text_tokenizer_type == 'flan_t5':
                tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # 待办：后续可下沉到 dataset。
                input_ids = tokens.input_ids.cuda(non_blocking=True)
                mask = tokens.attention_mask.cuda(non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
                kv_compact = []
                for text_ind, (len_i, feat_i) in enumerate(zip(lens, text_features.unbind(0))):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)
            else:
                text_features = text_encoder(captions, args.device)
                lens = [len(item) for item in text_features]
                cu_seqlens_k = [0]
                for len_i in lens:
                    cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
                cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
                Ltext = max(lens)
                kv_compact = torch.cat(text_features, dim=0).float()
                text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)

            if len(images):
                images = [item.to(args.device, non_blocking=True) for item in images]
            if len(raw_features_bcthw):
                raw_features_bcthw = [item.to(args.device, non_blocking=True) for item in raw_features_bcthw]

            # [日志]
            # 长时间训练时周期性把 args/log 刷到磁盘，避免中断后只剩 stdout。
            if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                args.dump_log()
                last_touch = time.time()

            # [获取调度后的超参数]
            # clip_decay_ratio 是训练后期更严格/更温和裁剪的调度项；具体使用在 trainer 内部。
            progress = g_it / (max_it - 1)
            clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1

            # 梯度累积：只有到达 args.ac 的边界才做 optimizer step，其余 iteration 只累积梯度。
            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)

        with maybe_record_function('in_training'):
            # Stage-1 的核心训练调用：
            # 语言条件 text_cond_tuple + 视频观测 latent/RGB 被送入世界模型，
            # 监督目标是下一段真实 latent token，而不是直接预测动作。
            grad_norm_t, scale_log2_t = trainer.train_step(
                epoch=epoch,
                it=it,
                g_it=g_it,
                stepping=stepping,
                clip_decay_ratio=clip_decay_ratio,
                metric_lg=me,
                inp_B3HW=images,
                raw_features_bcthw=raw_features_bcthw,
                feature_cache_files4images=feature_cache_files4images,
                text_cond_tuple=text_cond_tuple,
                media=media,
                args=args,
            )

        with maybe_record_function('after_train'):
            me.update(tlr=args.tlr)
    # ============================================= iteration 循环结束 =============================================

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


def main():
    """CLI 入口：初始化分布式参数、启动 Stage-1 训练并在退出前落盘最终 args。"""
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main_train(args)
    print(f'最终 args:\n\n{str(args)}')
    args.dump_log()
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    dist.barrier()

if __name__ == '__main__':
    main()
