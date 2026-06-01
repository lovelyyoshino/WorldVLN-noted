# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
"""
`Worldmodel/runtime` 下保留的旧版/备用 InfinityStar 训练入口（runtime 镜像）。

中文导读：
这个文件和顶层 `train/train.py` 承担相同的核心职责：组装 T5、VideoVAE、Infinity
世界模型、optimizer 和 trainer，然后跑分布式训练循环。当前版本额外保留了 GRPO
相关 batch 字段和 hybrid stepping 逻辑，主要用于 StageB/rollout 数据混训实验。

它和顶层 `train/train.py` 的关系：
- 顶层 `train/train.py` 是“开源 backbone 训练”的主入口，更新更频繁、依赖更精简；
- 本文件位于 `Worldmodel/runtime/`，作为 runtime 自带的镜像副本而存在；
- 服务端（`infer/server.py`、`action_aware_grpo/grpo_server.py`）以及本目录下的
  其他工具（`tools/run_infinity.py`、`tools/infinity_streaming_session.py` 等）
  在共享部分代码路径，因此 runtime 自带一份训练入口便于在“服务环境”里直接复跑训练
  而不必依赖外部 train 目录；
- 如果需要修改 SFT/GRPO 训练协议，应该先在顶层 `train/train.py` 落地，再同步到这里，
  避免两份入口长期分叉。

阅读建议：
1. 先看 `build_everything_from_args()`：tokenizer、model、optimizer、trainer 怎么组装；
2. 再看 `main_train()`：epoch / iters_train 怎么决定，auto_resume 如何处理；
3. 最后看 `train_one_epoch()`：每个 batch 包含哪些 GRPO 字段，hybrid_step_on_role 如何影响 step。
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
    根据解析后的 runtime 参数构建 tokenizer、模型、optimizer 和 trainer。

    中文导读：
    这是训练入口的组装点。它固定使用 flan-T5 作为语言条件编码器，调用
    `build_model_optimizer()` 构建 VideoVAE + Infinity transformer，并根据 checkpoint
    格式恢复 trainer 状态。返回的 trainer 后续会在 `train_one_epoch()` 中消费视频/GRPO batch。
    """
    # 设置随机种子。
    args.set_initial_seed(benchmark=True)
    # 构建 tokenizer。
    print(f'Loading T5 from {args.t5_path}...')
    if 'flan-t5' in args.t5_path:
        from transformers import T5EncoderModel, T5TokenizerFast
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True) # text_tokenizer.model_max_length 默认为 512。
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        args.text_tokenizer_type = 'flan_t5'
        args.text_tokenizer = text_tokenizer
    else: # umt5 分支当前不支持。
        raise ValueError("Only flan-t5 is supported now.")

    # 构建模型。这里的 gpt 是 causal VAR transformer，用文本条件做下一 scale 预测。
    vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args)

    # 注意：在 Dataloader 对象创建/迭代之后再导入较重的 `InfinityTrainer`，避免 OOM。
    InfinityTrainer = get_trainer(args)
    # 构建 trainer。
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

    # 从中断的实验自动恢复。
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
            print(f"omnistore resume from {resume_path}", flush=True)
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
    构建 VAE、Infinity 世界模型、分布式包装和 optimizer。

    中文导读：
    `gpt_wo_ddp` 是这里的世界模型主体，FSDP/DDP 只是在不同训练规模下包住它。`rush_resume`
    路径允许从结构相近但不完全一致的 checkpoint 启动，因此内部会丢弃 shape 不匹配的权重。
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.models.infinity import Infinity, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.freeze_utils import apply_stageb_partial_freeze
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt

    # 关闭内置初始化以加快启动。
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    count_p = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    num_para = count_p(gpt_wo_ddp)
    if num_para/1000 < 20: # 小于 20B 参数。
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
            """删除或迁移旧 checkpoint 中与当前模型结构不兼容的权重。"""
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                print(f'[rush_resume] drop word_embed.weight')
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                print(f'[rush_resume] drop head.weight')
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                print(f'[rush_resume] drop head.bias')
                del state_dict['head.bias']
            if 'text_proj_for_sos.ca.mat_kv.weight' in state_dict and \
                (state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape):
                print(f'[rush_resume] drop cfg_uncond')
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            if 'semantic_head.weight' in state_dict:
                if hasattr(gpt_wo_ddp, 'semantic_head2'):
                    print(f'[rush_resume] replace semantic_head with semantic_head2')
                    state_dict['semantic_head2.weight'] = state_dict['semantic_head.weight']
                    state_dict['semantic_head2.bias'] = state_dict['semantic_head.bias']
                else:
                    print(f'[rush_resume] current model has no semantic_head2; keep/drop by shape checks on semantic_head')
                del state_dict['semantic_head.weight']
                del state_dict['semantic_head.bias']
            if 'semantic_head2.weight' in state_dict:
                if (not hasattr(gpt_wo_ddp, 'semantic_head2')) or (
                    state_dict['semantic_head2.weight'].shape[0] != gpt_wo_ddp.semantic_head2.out_features
                ):
                    print(f'[rush_resume] drop semantic_head2.weight, semantic_head2.bias')
                    del state_dict['semantic_head2.weight']
                    if 'semantic_head2.bias' in state_dict:
                        del state_dict['semantic_head2.bias']
            return state_dict
        print(gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False))
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)
    elif args.torchshard_resume:
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(gpt_wo_ddp, args.torchshard_resume, strict=False)

    apply_stageb_partial_freeze(
        gpt_wo_ddp,
        args.freeze_chunk_prefix,
        bool(args.partial_freeze_print_summary),
    )

    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}

    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    print(f'[PT][#para], GPT={num_para:.2f}\n\n')

    gpt_uncompiled = gpt_wo_ddp

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # 使用 mixed precision；背景见：https://github.com/pytorch/pytorch/issues/76607
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
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)

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
    构建 runtime 训练数据集。

    数据集可能返回 SFT 视频样本，也可能返回 GRPO rollout/replay 样本；后者会携带 rewards、
    old_logprobs、clip ids、trace files 等字段，并在 trainer.train_step() 内部被相应损失使用。
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
    训练主循环：恢复 checkpoint、按 epoch 构建 dataset/dataloader，并调用单 epoch 训练。

    和顶层 Stage-1 入口相比，这里额外支持 `max_train_iters` 提前停止，以及
    `hybrid_step_on_role` 控制混合 batch 中哪些角色真正触发 optimizer step。
    """
    if args.checkpoint_type == 'torch':
        saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    elif args.checkpoint_type == 'omnistore':
        saver = omnistoreCheckpoint(eval_milestone=None)
    else:
        raise ValueError(f'{args.checkpoint_type=}')
    ret = build_everything_from_args(args, saver)

    if ret is None:
        return

    text_tokenizer, text_encoder, trainer, start_global_it = ret
    gc.collect(), torch.cuda.empty_cache()
    # 预先构建 epoch-0 dataset，拿到本次运行真实的 iters_train。
    # 这样可以避免 auto_resume 使用旧 g_it，而 iters_train 变化时直接空循环退出
    # （例如 GPU 数、dataloader worker 取整或 token_len 变化）。
    train_dataset = build_dataset(args)
    iters_train = len(train_dataset)
    if int(getattr(args, "save_model_iters_freq", 0)) <= 0:
        args.save_model_iters_freq = int(iters_train)
        print(f'[PT info] auto save_model_iters_freq={args.save_model_iters_freq} (once per epoch)')
    start_ep = start_global_it // iters_train
    start_it = start_global_it % iters_train
    if bool(getattr(args, "auto_resume", True)) and int(args.epoch) <= int(start_ep):
        extra = int(getattr(args, "extra_epochs_after_resume", 1) or 1)
        extra = max(1, extra)
        old_epoch = int(args.epoch)
        args.epoch = int(start_ep + extra)
        print(
            f'[PT info] Adjusted args.epoch from {old_epoch} to {args.epoch} '
            f'(start_ep={start_ep}, extra_epochs_after_resume={extra})'
        )

    seg5 = np.linspace(1, args.epoch, 5+1, dtype=int).tolist()
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.epoch // 10) if args.epoch <= 100 else max(1, args.epoch // 20)

    # ============================================= epoch 循环开始 =============================================
    # 构建 wandb logger。
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})
    total_epochs = int(args.epoch)
    for ep in range(total_epochs):
        # 每个 epoch 重新构建数据，确保每个 dataloader worker 都能读到最新 meta。
        # 注意：args.epoch 仍表示总 epoch 数；当前 epoch 存到 args.cur_epoch，供 dataset 使用。
        args.cur_epoch = ep

        if ep == 0:
            print(f'[PT info]  from ep{start_ep} it{start_it} {iters_train=}=======>  bed: {args.bed}  <=======\n')

        if ep < start_ep:
            continue
        if ep > start_ep:
            train_dataset = build_dataset(args)
            iters_train = len(train_dataset)
            if int(getattr(args, "save_model_iters_freq", 0)) <= 0:
                args.save_model_iters_freq = int(iters_train)
                print(f'[PT info] auto save_model_iters_freq={args.save_model_iters_freq} (once per epoch)')

        # [训练一个 epoch]
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.workers, pin_memory=True, batch_size=None)
        stats, stop_after_max_iters = train_one_epoch(
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
        if stop_after_max_iters:
            print(f'[PT info] stop after max_train_iters={args.max_train_iters}', flush=True)
            break
    return


g_speed_ls = deque(maxlen=128)
def train_one_epoch(
    epoch: int, is_first_ep: bool, start_it: int, start_global_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, dataloader_iter, iters_train: int,
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer,
):
    """
    执行一个 epoch 的 runtime 训练。

    每个 batch 会先准备 compact T5 文本条件，再把 RGB/latent 视频字段和可选 GRPO 字段
    传给 `trainer.train_step()`。GRPO 字段包括奖励、old/ref logprob、优势、轨迹 id 和 trace
    文件路径，用于 StageB 策略优化或 SFT+GRPO 混合训练。
    """
    # 注意：在 Dataloader 对象创建/迭代之后再导入较重的包，避免 OOM。
    step_cnt = 0
    header = f'[Ep]: [{epoch:4d}/{args.epoch}]'

    last_touch = time.time()
    g_it, max_it = epoch * iters_train, args.epoch * iters_train

    doing_profiling = args.prof and epoch == 0 and (args.profall or dist.is_master())
    maybe_record_function = record_function if doing_profiling else nullcontext
    trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function

    last_t_perf = time.time()
    speed_ls: deque = g_speed_ls
    freq_upper = max(1, iters_train//2 - 1)
    FREQ = max(1, min(int(args.prof_freq), freq_upper))
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
    stop_after_max_iters = False
    max_train_iters = int(getattr(args, 'max_train_iters', 0) or 0)
    stop_global_it_exclusive = (int(start_global_it) + max_train_iters) if max_train_iters > 0 else None
    # ============================================= iteration 循环开始 =============================================
    for it, data in me.log_every(start_it, iters_train, dataloader_iter, args.log_freq, args.log_every_iter, header, args):
        g_it = epoch * iters_train + it
        if stop_global_it_exclusive is not None and g_it >= stop_global_it_exclusive:
            stop_after_max_iters = True
            break
        # 中文说明：mfutool.step()
        # mfu_val = mfutool.get_mfu() * 100  # 转成百分比。
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
            # 视频 SFT 字段（必有）：
            # - images / raw_features_bcthw：RGB 帧或预先编码的 latent；
            # - captions：每条样本对应的文本提示；
            # - feature_cache_files4images / media：缓存路径与媒体类型元数据。
            images, captions, raw_features_bcthw, feature_cache_files4images, media = data['images'], data['captions'], data['raw_features_bcthw'], data['feature_cache_files4images'], data['media']
            # GRPO/replay 字段（可选）：
            # 这些字段只有在使用 GRPO 或 hybrid SFT+GRPO 数据集时才存在，否则为 None。
            # 它们最终会被一起传给 trainer.train_step()，由训练器内部决定是否启用对应的损失项。
            grpo_rewards = data.get('grpo_rewards', None)
            grpo_old_logprobs = data.get('grpo_old_logprobs', None)
            grpo_adv_finals = data.get('grpo_adv_finals', None)
            grpo_reward_acts = data.get('grpo_reward_acts', None)
            grpo_reward_tasks = data.get('grpo_reward_tasks', None)
            grpo_reward_task_raws = data.get('grpo_reward_task_raws', None)
            grpo_reward_task_dense_raws = data.get('grpo_reward_task_dense_raws', None)
            grpo_reward_task_success_raws = data.get('grpo_reward_task_success_raws', None)
            grpo_succs = data.get('grpo_succs', None)
            grpo_succ_trajs = data.get('grpo_succ_trajs', None)
            grpo_task_final_costs = data.get('grpo_task_final_costs', None)
            grpo_task_final_pos_errs = data.get('grpo_task_final_pos_errs', None)
            grpo_task_final_yaw_errs = data.get('grpo_task_final_yaw_errs', None)
            grpo_reward_ces = data.get('grpo_reward_ces', None)
            grpo_ref_logprobs = data.get('grpo_ref_logprobs', None)
            grpo_group_ids = data.get('grpo_group_ids', None)
            grpo_clip_ids = data.get('grpo_clip_ids', None)
            grpo_trace_files = data.get('grpo_trace_files', None)
            traj_ids = data.get('traj_ids', None)
            # hybrid_roles：每条样本是 "sft" anchor 还是 "grpo" 候选；
            # 后面 `hybrid_step_on_role` 会用它决定 batch 是否真正触发 optimizer step。
            hybrid_roles = data.get('hybrid_roles', None)

            # # [准备文本特征]
            if args.text_tokenizer_type == 'flan_t5':
                tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # TODO：后续放进 dataset。
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

            # [记录日志]
            if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                args.dump_log()
                last_touch = time.time()

            # [获取按进度调度的超参数]
            progress = g_it / (max_it - 1)
            clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1

            stepping = (g_it + 1) % args.ac == 0
            # 可选 hybrid stepping：只在指定 batch role 上 step，例如 "sft" anchor。
            step_on_role = str(getattr(args, "hybrid_step_on_role", "") or "").strip().lower()
            if step_on_role:
                role0 = ""
                if isinstance(hybrid_roles, list) and len(hybrid_roles) > 0:
                    role0 = str(hybrid_roles[0] or "").strip().lower()
                stepping = role0 == step_on_role
            step_cnt += int(stepping)

        with maybe_record_function('in_training'):
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
                grpo_rewards=grpo_rewards,
                grpo_old_logprobs=grpo_old_logprobs,
                grpo_adv_finals=grpo_adv_finals,
                grpo_reward_acts=grpo_reward_acts,
                grpo_reward_tasks=grpo_reward_tasks,
                grpo_reward_task_raws=grpo_reward_task_raws,
                grpo_reward_task_dense_raws=grpo_reward_task_dense_raws,
                grpo_reward_task_success_raws=grpo_reward_task_success_raws,
                grpo_succs=grpo_succs,
                grpo_succ_trajs=grpo_succ_trajs,
                grpo_task_final_costs=grpo_task_final_costs,
                grpo_task_final_pos_errs=grpo_task_final_pos_errs,
                grpo_task_final_yaw_errs=grpo_task_final_yaw_errs,
                grpo_reward_ces=grpo_reward_ces,
                grpo_ref_logprobs=grpo_ref_logprobs,
                grpo_group_ids=grpo_group_ids,
                grpo_clip_ids=grpo_clip_ids,
                grpo_trace_files=grpo_trace_files,
                traj_ids=traj_ids,
                hybrid_roles=hybrid_roles,
            )

        with maybe_record_function('after_train'):
            me.update(tlr=args.tlr)
    # ============================================= iteration 循环结束 =============================================

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, stop_after_max_iters


def main():
    """CLI 入口：初始化分布式参数，启动 runtime 训练，并在退出前写出最终参数日志。"""
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main_train(args)
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    dist.barrier()

if __name__ == '__main__':
    main()
