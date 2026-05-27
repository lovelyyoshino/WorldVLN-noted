# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import time
import hashlib
import shutil
import re
import json
from typing import Dict

import cv2
import numpy as np
import torch
# 允许通过环境变量覆盖 dynamo cache 上限。
try:
    # 默认值和 train.py 保持一致；需要时可用环境变量覆盖。
    torch._dynamo.config.cache_size_limit = int(os.getenv("TORCHDYNAMO_CACHE_SIZE_LIMIT", "4096"))
except Exception:
    pass
from transformers import AutoTokenizer
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast
from timm.models import create_model
import imageio

from infinity.models.infinity import Infinity
from infinity.utils.load import load_visual_tokenizer
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file as safe_save_file


def split_state_dict(state_dict: Dict[str, torch.Tensor], save_directory: str, max_shard_size='8GB'):
    """把大模型 state_dict 切成 safetensors 分片，并写出 HF 风格 index。"""
    state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size)
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        safe_save_file(
            shard,
            os.path.join(save_directory, filename),
            metadata={"format": "pt"},
        )
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))

def extract_key_val(text):
    """解析形如 `<key:value>` 的 prompt 附加字段，返回键值字典。"""
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def encode_prompt(t5_path, text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False):
    """
    把文本 prompt 编成 Infinity transformer 使用的 compact text condition。

    返回 `(kv_compact, lens, cu_seqlens_k, Ltext)`：这是后续 varlen/flex attention 需要的
    紧凑文本表示，而不是简单 padded hidden states。

    代码/形状说明：
    - `kv_compact`: `(sum_i L_i, C_text)`，把 batch 内每条文本的有效 token 首尾相接；
    - `lens`: Python `List[int]`，记录每条文本各自的有效长度 `L_i`；
    - `cu_seqlens_k`: `(B+1,)` 的前缀和，供 varlen/flex attention 快速知道每条文本
      在 `kv_compact` 里的起止偏移；
    - `Ltext`: 这一批文本里的最大长度，通常用来构造 attention mask 上界。

    注意：`enable_positive_prompt` 在这个 runtime 里仍是预留开关，当前主路径还没有
    独立的“positive prompt 分支”逻辑；打开它不会改变编码结果。

    为什么后续不直接传 padded hidden states：
    - varlen/flex attention 只关心每条文本的有效 token；
    - `kv_compact + cu_seqlens_k` 能避免把 padding token 也送进注意力计算，节省算力与显存。
    """
    if enable_positive_prompt:
        pass
    print(f'文本提示词: {prompt}')
    captions = [prompt]
    if 'flan-t5' in t5_path:
        tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.cuda(non_blocking=True)
        mask = tokens.attention_mask.cuda(non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    else:
        text_features = text_encoder(captions, 'cuda')
        lens = [len(item) for item in text_features]
        cu_seqlens_k = [0]
        for len_i in lens:
            cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
        Ltext = max(lens)
        kv_compact = torch.cat(text_features, dim=0).float()
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def gen_one_example(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    input_use_interplote_up=False,
    low_vram_mode=False,
    args=None,
    get_visual_rope_embeds=None,
    context_info=None,
    noise_list=None,
    return_summed_code_only=False,
    mode='',
    former_clip_features=None,
    first_frame_features=None,
):
    """
    单条 prompt 的 InfinityStar 推理封装。

    中文导读：
    该函数负责把 prompt/negative prompt 编码成文本条件，规范化 cfg/tau 列表，然后调用
    `autoregressive_infer()`。服务端路径会复用它的 `return_summed_code_only=True` 模式，
    直接拿 latent/summed_codes 给动作头，而不是一定解码成视频。
    """
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, prompt, enable_positive_prompt, low_vram_mode=low_vram_mode)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, negative_prompt, low_vram_mode=low_vram_mode)
    else:
        negative_label_B_or_BLT = None
    print(f'采样参数 cfg={cfg_list}, tau={tau_list}')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        out = infinity_test.autoregressive_infer(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            input_use_interplote_up=input_use_interplote_up,
            low_vram_mode=low_vram_mode,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=noise_list,
            return_summed_code_only=return_summed_code_only,
            mode=mode,
            former_clip_features=former_clip_features,
            first_frame_features=first_frame_features,
        )
        if return_summed_code_only:
            return out
        else:
            pred_multi_scale_bit_labels, img_list = out

    print(f"推理总耗时={time.time() - sstt:.3f}s，Infinity 自回归耗时={time.time() - stt:.3f}s")
    img = img_list[0]
    return img, pred_multi_scale_bit_labels

def get_prompt_id(prompt):
    """用 MD5 为 prompt 生成稳定 id，方便缓存/输出目录命名。"""
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    """从训练 checkpoint 中抽取 GPT 权重，保存成更小的推理用 slim checkpoint。"""
    print('[保存 slim 推理模型]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # 代码/形状说明：ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'输出路径：{save_file}')
    torch.save(infinity_slim, save_file)
    print('[保存 slim 推理模型] 完成')
    return save_file

def load_tokenizer(t5_path =''):
    """加载 flan-T5 tokenizer 和 text encoder，并冻结到 eval 模式。"""
    print('[加载 tokenizer 与文本编码器]')
    if 'flan-t5' in t5_path:
        from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
        # 代码/形状说明：这里实际以 float16 加载 T5 encoder，与当前 runtime 路径保持一致。
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
        text_encoder.to('cuda')
        text_encoder.eval()
        text_encoder.requires_grad_(False)
    else:
        raise ValueError(f'当前仅支持 flan-t5 系列文本编码器，收到的 t5_path={t5_path}')
    return text_tokenizer, text_encoder

def transform(pil_img, tgt_h, tgt_w):
    """按保持比例 resize + center crop，把 PIL RGB 图变成 [-1,1] 的 CHW tensor。"""
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # 从中心裁剪出目标尺寸。
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def load_transformer(vae, args):
    """
    构建 Infinity transformer 并加载 torch/safetensors/omnistore checkpoint。

    中文导读：
    这里是推理 runtime 的模型入口。普通训练 checkpoint 会先在 CPU 上读取，再只把
    `trainer.gpt_fsdp` 权重加载进 CUDA 模型，避免把 optimizer/trainer 状态一起搬到显存。

    可以把 `checkpoint_type` 理解成三条加载分支：
    - `torch`：读取单个 `.pth`，如果里面带 `trainer` 字段，就只抽取 `trainer.gpt_fsdp`；
    - `torch_shard`：读取 Hugging Face 风格的分片 safetensors/pytorch 分片目录；
    - `omnistore`：先把远端/缓存目录中的分片合并成完整 state_dict，再加载到推理模型。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path

    print('[加载 Infinity 主模型]')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = create_model(
            args.model_type,
            vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
            raw_scale_schedule=None,
            checkpointing='full-block',
            pad_to_multiplier=128,
            use_flex_attn=args.use_flex_attn,
            add_lvl_embeding_on_first_block=0,
            num_of_label_value=args.num_of_label_value,
            rope2d_each_sa_layer=args.rope2d_each_sa_layer,
            rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
            pn=args.pn,
            apply_spatial_patchify=args.apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[0.571, 1.0],
            video_frames=args.video_frames,
            other_args=args,
        ).to(device=device)
        print(
            f'[Infinity 模型] 结构={args.model_type}，参数规模='
            f'{sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B，bf16={args.bf16}'
        )
        if args.bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()
        infinity_test.eval()
        infinity_test.requires_grad_(False)
        infinity_test.cuda()
        torch.cuda.empty_cache()

        if not model_path:
            return infinity_test

        print('============== [加载 Infinity 权重] ==============')
        if args.checkpoint_type == 'torch':
            # 注意：
            # 训练 checkpoint（global_step_*.pth）包含 optimizer/trainer 状态。
            # 不要把整个 checkpoint 直接加载到 CUDA，否则 80G GPU 也可能 OOM。
            # 这里先加载到 CPU，再只把模型权重拷到 CUDA 模型。
            state_dict = torch.load(model_path, map_location='cpu')
            if 'trainer' in state_dict:
                print(infinity_test.load_state_dict(state_dict['trainer']['gpt_fsdp']))
            else:
                print(infinity_test.load_state_dict(state_dict))
        elif args.checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            print(load_sharded_checkpoint(infinity_test, model_path, strict=False))
        elif args.checkpoint_type == 'omnistore':
            from infinity.utils.save_and_load import merge_ckpt
            if args.enable_model_cache and osp.exists(args.cache_dir):
                local_model_dir = osp.abspath(osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_')))
            else:
                local_model_dir = osp.abspath(model_path)
            print(f'从 {local_model_dir} 加载并合并 omnistore checkpoint')
            state_dict = merge_ckpt(local_model_dir, osp.join(local_model_dir, 'ouput'), save=False, fsdp_save_flatten_model=args.fsdp_save_flatten_model)
            print(infinity_test.load_state_dict(state_dict))
        infinity_test.rng = torch.Generator(device=device)
    return infinity_test

def save_video(ndarray_image_list, fps=24, save_filepath='tmp.mp4', force_all_keyframes: bool = False):
    """保存 BGR 帧序列为 mp4；单帧时保存 jpg，并可强制全关键帧方便短片回放。"""
    # 也接受 torch tensor，这在演示里很常见。
    if isinstance(ndarray_image_list, torch.Tensor):
        ndarray_image_list = ndarray_image_list.detach().cpu().numpy()
    if len(ndarray_image_list) == 1:
        save_filepath = save_filepath.replace('.mp4', '.jpg')
        cv2.imwrite(save_filepath, ndarray_image_list[0])
        print(f"单帧图像已保存到 {osp.abspath(save_filepath)}")
    else:
        # 负步长通道翻转需要 numpy array；Torch slicing 不支持 step=-1。
        if not isinstance(ndarray_image_list, np.ndarray):
            ndarray_image_list = np.asarray(ndarray_image_list)
        # 确保内存连续；部分视频写入器处理非连续视图时会出错。
        ndarray_image_list = np.ascontiguousarray(ndarray_image_list)
        h, w = ndarray_image_list[0].shape[:2]
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        rgb = np.ascontiguousarray(ndarray_image_list[:, :, :, ::-1])
        # 短视频如果以非 I 帧或重排序帧开头，某些播放器可能显示损坏的首帧。
        # 需要时强制所有帧都是 keyframe。
        if force_all_keyframes and save_filepath.endswith(".mp4"):
            try:
                writer = imageio.get_writer(
                    save_filepath,
                    fps=fps,
                    codec="libx264",
                    ffmpeg_params=[
                        "-pix_fmt",
                        "yuv420p",
                        "-g",
                        "1",
                        "-keyint_min",
                        "1",
                        "-sc_threshold",
                        "0",
                    ],
                )
                for fr in rgb:
                    writer.append_data(fr)
                writer.close()
            except Exception:
                # 回退到默认 mimsave 写法。
                imageio.mimsave(save_filepath, rgb, fps=fps)
        else:
            imageio.mimsave(save_filepath, rgb, fps=fps)
        print(f"视频已保存到 {osp.abspath(save_filepath)}")

def read_video_as_frames(video_path):
    """用 OpenCV 把视频或 jpg 读取成 BGR numpy frame 序列。"""
    if video_path.endswith('.jpg'):
        return cv2.imread(video_path)[None, ...]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return None
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()
    frames = np.stack(frames)
    return frames

def add_common_arguments(parser):
    """
    为 Infinity 推理/导出脚本注册通用模型、采样和分辨率参数。

    初学者可以把这组参数分成四类来记：
    1. 模型结构是否对齐：`model_type`、`vae_type`、`videovae`、`text_channels`；
    2. 采样行为如何变化：`cfg`、`tau`、`sampling_per_bits`；
    3. 分辨率/时间长度如何决定：`pn`、`dynamic_scale_schedule`、`video_frames`、`h_div_w_template`；
    4. 加载与缓存如何工作：`model_path`、`checkpoint_type`、`cache_dir`。
    """
    parser.add_argument('--cfg', type=str, default='3', help='Classifier-Free Guidance 强度；可写单值或与多尺度采样对应的列表。值越大越强调条件，但也更容易变僵。')
    parser.add_argument('--tau', type=float, default=1, help='采样温度；越大随机性越强，越小越保守。')
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '0.40M', '0.90M'], help='动态分辨率预设名，决定 token 预算/patch 数规模，是推导 scale schedule 的核心入口。')
    parser.add_argument('--model_path', type=str, default='', help='Infinity 主模型 checkpoint 路径；留空时只构建模型结构，不加载权重。')
    parser.add_argument('--cfg_insertion_layer', type=int, default=0, help='CFG 注入层位置；不同实验分支会在不同层插入条件引导，通常保持训练时默认值。')
    parser.add_argument('--vae_type', type=int, default=64, help='Video VAE 结构编号，必须与 `vae_path` 中的 checkpoint 对齐。')
    parser.add_argument('--vae_path', type=str, default='', help='Video VAE checkpoint 路径。')
    parser.add_argument('--add_lvl_embeding_on_first_block', type=int, default=0, choices=[0,1], help='是否在第一个 Transformer block 额外加入 level embedding；需与训练配置一致。')
    parser.add_argument('--num_of_label_value', type=int, default=2, help='离散 label/value 的取值个数，通常和量化/bit 表示方案绑定。')
    parser.add_argument('--model_type', type=str, default='infinity_2b', help='Infinity Transformer 架构名，例如 `infinity_2b`。')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1], help='是否在每层 self-attention 上应用 2D RoPE。')
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2], help='2D RoPE 坐标归一化策略；不同取值对应是否按目标高宽模板归一。')
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1], help='是否把 scale schedule 本身编码成额外 embedding 注入模型。')
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16], help='每个 bit/token 采样几次；值越大通常越慢，但有时能带来更稳的采样。')
    parser.add_argument('--text_encoder_ckpt', type=str, default='', help='文本编码器 checkpoint，当前运行时主要支持 flan-T5。')
    parser.add_argument('--text_channels', type=int, default=2048, help='文本条件通道数，必须与主模型训练时看到的文本特征维度一致。')
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1], help='是否对视觉 latent/token 启用空间 patchify 表示。')
    parser.add_argument('--h_div_w_template', type=float, default=1.000, help='目标高宽比模板 H/W；会影响动态分辨率调度与部分 RoPE/网格推导。')
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1], help='是否启用 flex/varlen attention 实现。')
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1], help='预留的 positive prompt 开关；当前 runtime 主路径仍是占位实现，开启后通常不会改变结果。')
    parser.add_argument('--cache_dir', type=str, default='/dev/shm', help='模型缓存目录；启用模型缓存或 omnistore 合并时会把分片落到这里。')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1], help='是否启用本地模型缓存，减少重复从远端或共享盘搬运 checkpoint。')
    parser.add_argument('--checkpoint_type', type=str, default='torch', help='checkpoint 格式：`torch`、`torch_shard` 或 `omnistore`。')
    parser.add_argument('--seed', type=int, default=0, help='随机种子。')
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1], help='是否以 bf16 运行主模型 block；通常显存更省、速度更好。')
    parser.add_argument('--dynamic_scale_schedule', type=str, default='13_hand_craft', help='动态尺度计划名称，决定各尺度 token 网格和采样顺序。')
    parser.add_argument('--video_frames', type=int, default=81, help='目标视频帧数；常见默认是 81，对应 world model 里常用的 4n+1 时间规则。')
    parser.add_argument('--videovae', type=int, default=10, help='Video VAE 运行模式/注册编号，需与当前 runtime 的 VAE 实现匹配。')
    parser.add_argument('--fake_vae_input', type=int, default=0, choices=[0,1], help='是否使用假的 VAE 输入做调试/占位，不走真实视觉编码链路。')
    parser.add_argument('--casual_multi_scale', type=int, default=0, choices=[0,1], help='是否启用当前实现中的 `casual_multi_scale` 开关；仅在对应训练配置下使用。')
    parser.add_argument('--scale_embeds_num', type=int, default=128, help='scale embedding 表的容量/数量上限。')
    parser.add_argument('--train_h_div_w_list', type=float, default=None, nargs='+', help='训练阶段见过的高宽比列表；某些旧 checkpoint 需要它来对齐动态分辨率行为。')
    parser.add_argument('--mask_type', type=str, default='infinity_elegant_clip20frames_v2', help='注意力 mask 方案名称，决定时间/尺度间哪些 token 可以互相看见。')
    parser.add_argument('--context_frames', type=int, default=1000, help='可回看的上下文帧上限；长视频或流式推理时，它决定模型最多保留多少历史。')
