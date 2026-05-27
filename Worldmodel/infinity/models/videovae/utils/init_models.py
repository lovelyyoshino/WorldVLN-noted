# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from infinity.models.videovae.utils.misc import is_torch_optim_sch


def inflate_gen(state_dict, temporal_patch_size, spatial_patch_size, strategy="average", inflation_pe=False):
    """中文说明：`inflate_gen` 实现VideoVAE 权重加载和维度膨胀工具中的 `inflate_gen` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    new_state_dict = state_dict.copy()

    pe_image0_w = state_dict["encoder.to_patch_emb_first_frame.1.weight"] # 代码/形状说明：image_channel * patch_width * patch_height
    pe_image0_b = state_dict["encoder.to_patch_emb_first_frame.1.bias"] # 代码/形状说明：image_channel * patch_width * patch_height
    pe_image1_w = state_dict["encoder.to_patch_emb_first_frame.2.weight"] # 代码/形状说明：image_channel * patch_width * patch_height, dim
    pe_image1_b = state_dict["encoder.to_patch_emb_first_frame.2.bias"] # 代码/形状说明：image_channel * patch_width * patch_height
    pe_image2_w = state_dict["encoder.to_patch_emb_first_frame.3.weight"] # 代码/形状说明：image_channel * patch_width * patch_height
    pe_image2_b = state_dict["encoder.to_patch_emb_first_frame.3.bias"] # 代码/形状说明：image_channel * patch_width * patch_height

    pd_image0_w = state_dict["decoder.to_pixels_first_frame.0.weight"] # 代码/形状说明：dim, image_channel * patch_width * patch_height
    pd_image0_b = state_dict["decoder.to_pixels_first_frame.0.bias"] # 代码/形状说明：image_channel * patch_width * patch_height

    pe_video0_w = state_dict["encoder.to_patch_emb.1.weight"]

    old_patch_size = int(math.sqrt(pe_image0_w.shape[0] // 3))
    old_patch_size_temporal = pe_video0_w.shape[0] // (3 * old_patch_size * old_patch_size)

    if old_patch_size != spatial_patch_size or old_patch_size_temporal != temporal_patch_size:
        if not inflation_pe:
            del new_state_dict["encoder.to_patch_emb_first_frame.1.weight"]
            del new_state_dict["encoder.to_patch_emb_first_frame.1.bias"]
            del new_state_dict["encoder.to_patch_emb_first_frame.2.weight"]

            del new_state_dict["decoder.to_pixels_first_frame.0.weight"]
            del new_state_dict["decoder.to_pixels_first_frame.0.bias"]

            del new_state_dict["encoder.to_patch_emb.1.weight"]
            del new_state_dict["encoder.to_patch_emb.1.bias"]
            del new_state_dict["encoder.to_patch_emb.2.weight"]

            del new_state_dict["decoder.to_pixels.0.weight"]
            del new_state_dict["decoder.to_pixels.0.bias"]

            return new_state_dict


        print(f"将 patch embedding 尺寸从 {old_patch_size_temporal}x{old_patch_size}x{old_patch_size} 膨胀到 {temporal_patch_size}x{spatial_patch_size}x{spatial_patch_size}。")
        pe_image0_w = F.interpolate(pe_image0_w.unsqueeze(0).unsqueeze(0), size=(3 * spatial_patch_size * spatial_patch_size)).squeeze(0).squeeze(0)
        pe_image0_b = F.interpolate(pe_image0_b.unsqueeze(0).unsqueeze(0), size=(3 * spatial_patch_size * spatial_patch_size)).squeeze(0).squeeze(0)
        pe_image1_w = F.interpolate(pe_image1_w.unsqueeze(0), size=(3 * spatial_patch_size * spatial_patch_size)).squeeze(0)

        new_state_dict["encoder.to_patch_emb_first_frame.1.weight"] = pe_image0_w
        new_state_dict["encoder.to_patch_emb_first_frame.1.bias"] = pe_image0_b
        new_state_dict["encoder.to_patch_emb_first_frame.2.weight"] = pe_image1_w

        pd_image0_w = F.interpolate(pd_image0_w.permute(1, 0).unsqueeze(0), size=(3 * spatial_patch_size * spatial_patch_size)).squeeze(0).permute(1, 0)
        pd_image0_b = F.interpolate(pd_image0_b.unsqueeze(0).unsqueeze(0), size=(3 * spatial_patch_size * spatial_patch_size)).squeeze(0).squeeze(0)

        new_state_dict["decoder.to_pixels_first_frame.0.weight"] = pd_image0_w
        new_state_dict["decoder.to_pixels_first_frame.0.bias"] = pd_image0_b

        pe_video0_w = state_dict["encoder.to_patch_emb.1.weight"]
        pe_video0_b = state_dict["encoder.to_patch_emb.1.bias"]
        pe_video1_w = state_dict["encoder.to_patch_emb.2.weight"]

        pe_video0_w = F.interpolate(pe_video0_w.unsqueeze(0).unsqueeze(0), size=(3 * temporal_patch_size * spatial_patch_size * spatial_patch_size)).squeeze(0).squeeze(0)
        pe_video0_b = F.interpolate(pe_video0_b.unsqueeze(0).unsqueeze(0), size=(3 * temporal_patch_size* spatial_patch_size * spatial_patch_size)).squeeze(0).squeeze(0)
        pe_video1_w = F.interpolate(pe_video1_w.unsqueeze(0), size=(3 * temporal_patch_size * spatial_patch_size * spatial_patch_size)).squeeze(0)

        pd_video0_w = state_dict["decoder.to_pixels.0.weight"]
        pd_video0_b = state_dict["decoder.to_pixels.0.bias"]

        pd_video0_w = F.interpolate(pd_image0_w.permute(1, 0).unsqueeze(0), size=(3 * temporal_patch_size * spatial_patch_size * spatial_patch_size)).squeeze(0).permute(1, 0)
        pd_video0_b = F.interpolate(pd_image0_b.unsqueeze(0).unsqueeze(0), size=(3 * temporal_patch_size * spatial_patch_size * spatial_patch_size)).squeeze(0).squeeze(0)

        new_state_dict["encoder.to_patch_emb.1.weight"] = pe_video0_w
        new_state_dict["encoder.to_patch_emb.1.bias"] = pe_video0_b
        new_state_dict["encoder.to_patch_emb.2.weight"] = pe_video1_w

        new_state_dict["decoder.to_pixels.0.weight"] = pd_video0_w
        new_state_dict["decoder.to_pixels.0.bias"] = pd_video0_b

        return new_state_dict


    if strategy == "average":
        pe_video0_w = torch.cat([pe_image0_w/temporal_patch_size] * temporal_patch_size)
        pe_video0_b = torch.cat([pe_image0_b/temporal_patch_size] * temporal_patch_size)

        pe_video1_w = torch.cat([pe_image1_w/temporal_patch_size] * temporal_patch_size, dim=-1)
        pe_video1_b = pe_image1_b # 代码/形状说明：torch.cat([pe_image1_b/temporal_patch_size] * temporal_patch_size)

        pe_video2_w = pe_image2_w # 代码/形状说明：torch.cat([pe_image2_w/temporal_patch_size] * temporal_patch_size)
        pe_video2_b = pe_image2_b # 代码/形状说明：torch.cat([pe_image2_b/temporal_patch_size] * temporal_patch_size)

    elif strategy == "first":
        pe_video0_w = torch.cat([pe_image0_w] + [torch.zeros_like(pe_image0_w, dtype=pe_image0_w.dtype)] * (temporal_patch_size - 1))
        pe_video0_b = torch.cat([pe_image0_b] + [torch.zeros_like(pe_image0_b, dtype=pe_image0_b.dtype)] * (temporal_patch_size - 1))

        pe_video1_w = torch.cat([pe_image1_w] + [torch.zeros_like(pe_image1_w, dtype=pe_image1_w.dtype)] * (temporal_patch_size - 1), dim=-1)
        pe_video1_b = pe_image1_b # 代码/形状说明：torch.cat([pe_image1_b] + [torch.zeros_like(pe_image1_b, dtype=pe_image1_b.dtype)] * (temporal_patch_size - 1))

        pe_video2_w = pe_image2_w # 代码/形状说明：torch.cat([pe_image2_w] + [torch.zeros_like(pe_image2_w, dtype=pe_image2_w.dtype)] * (temporal_patch_size - 1))
        pe_video2_b = pe_image2_b # 代码/形状说明：torch.cat([pe_image2_b] + [torch.zeros_like(pe_image2_b, dtype=pe_image2_b.dtype)] * (temporal_patch_size - 1))


    else:
        raise NotImplementedError


    new_state_dict["encoder.to_patch_emb.1.weight"] = pe_video0_w
    new_state_dict["encoder.to_patch_emb.1.bias"] = pe_video0_b

    new_state_dict["encoder.to_patch_emb.2.weight"] = pe_video1_w
    new_state_dict["encoder.to_patch_emb.2.bias"] = pe_video1_b

    new_state_dict["encoder.to_patch_emb.3.weight"] = pe_video2_w
    new_state_dict["encoder.to_patch_emb.3.bias"] = pe_video2_b


    if strategy == "average":
        pd_video0_w = torch.cat([pd_image0_w/temporal_patch_size] * temporal_patch_size)
        pd_video0_b = torch.cat([pd_image0_b/temporal_patch_size] * temporal_patch_size)

    elif strategy == "first":
        pd_video0_w = torch.cat([pd_image0_w] + [torch.zeros_like(pd_image0_w, dtype=pd_image0_w.dtype)] * (temporal_patch_size - 1))
        pd_video0_b = torch.cat([pd_image0_b] + [torch.zeros_like(pd_image0_b, dtype=pd_image0_b.dtype)] * (temporal_patch_size - 1))

    else:
        raise NotImplementedError


    new_state_dict["decoder.to_pixels.0.weight"] = pd_video0_w
    new_state_dict["decoder.to_pixels.0.bias"] = pd_video0_b

    return new_state_dict


def inflate_dis(state_dict, strategy="center"):
    """中文说明：`inflate_dis` 实现VideoVAE 权重加载和维度膨胀工具中的 `inflate_dis` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    print("#" * 50)
    print(f"使用 {strategy} 策略初始化视频判别器。")
    print("#" * 50)
    idis_weights = {k: v for k, v in state_dict.items() if "image_discriminator" in k}
    vids_weights = {k: v for k, v in state_dict.items() if "video_discriminator" in k}

    new_state_dict = state_dict.copy()
    for k in vids_weights.keys():
        del new_state_dict[k]


    for k in idis_weights.keys():
        new_k = "video_discriminator" + k[len("image_discriminator"):]
        if "weight" in k and new_state_dict[k].ndim == 4:
            old_weight = state_dict[k]
            if strategy == "average":
                new_weight = old_weight.unsqueeze(2).repeat(1, 1, 4, 1, 1) / 4
            elif strategy == "center":
                new_weight_ = old_weight# 中文说明：.unsqueeze(2) # O I 1 K K
                new_weight = torch.zeros((new_weight_.size(0), new_weight_.size(1), 4, new_weight_.size(2), new_weight_.size(3)), dtype=new_weight_.dtype)
                new_weight[:, :, 1] = new_weight_

            elif strategy == "first":
                new_weight_ = old_weight# 中文说明：.unsqueeze(2)
                new_weight = torch.zeros((new_weight_.size(0), new_weight_.size(1), 4, new_weight_.size(2), new_weight_.size(3)), dtype=new_weight_.dtype)
                new_weight[:, :, 0] = new_weight_

            elif strategy == "last":
                new_weight_ = old_weight# 中文说明：.unsqueeze(2)
                new_weight = torch.zeros((new_weight_.size(0), new_weight_.size(1), 4, new_weight_.size(2), new_weight_.size(3)), dtype=new_weight_.dtype)
                new_weight[:, :, -1] = new_weight_
            else:
                raise NotImplementedError

            new_state_dict[new_k] = new_weight

        elif "bias" in k:
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[new_k] = state_dict[k]


    return new_state_dict

def load_unstrictly(state_dict, model, loaded_keys=[]):
    """中文说明：`load_unstrictly` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    missing_keys = []
    for name, param in model.named_parameters():
        if name in state_dict:
            try:
                param.data.copy_(state_dict[name])
            except:
                # 调试输出：print(f"{name} 形状不匹配：参数 {name} 的 shape 是 {param.data.shape}，state_dict 中是 {state_dict[name].shape}")
                missing_keys.append(name)
        elif name not in loaded_keys:
            missing_keys.append(name)
    return model, missing_keys

def init_vae_only(state_dict, vae):
    """中文说明：`init_vae_only` 实现VideoVAE 权重加载和维度膨胀工具中的 `init_vae_only` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    vae, missing_keys = load_unstrictly(state_dict, vae)
    print(f"加载 vae 时缺失的权重键: {[key for key in missing_keys if not key.startswith('flux')]}")
    return vae

def init_image_disc(state_dict, image_disc, args):
    """中文说明：`init_image_disc` 实现VideoVAE 权重加载和维度膨胀工具中的 `init_image_disc` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if args.no_init_idis or args.init_idis == "no":
        state_dict = {}
    else:
        state_dict = state_dict["image_disc"]
    # 将 nn.GroupNorm 权重加载到 Normalize 类
    delete_keys = []
    loaded_keys = []
    model = image_disc
    for key in state_dict:
        if key.endswith(".weight"):
            norm_key = key.replace(".weight", ".norm.weight")
            if norm_key and norm_key in model.state_dict():
                model.state_dict()[norm_key].copy_(state_dict[key])
                delete_keys.append(key)
                loaded_keys.append(norm_key)
        if key.endswith(".bias"):
            norm_key = key.replace(".bias", ".norm.bias")
            if norm_key and norm_key in model.state_dict():
                model.state_dict()[norm_key].copy_(state_dict[key])
                delete_keys.append(key)
                loaded_keys.append(norm_key)
    for key in delete_keys:
        del state_dict[key]
    msg = image_disc.load_state_dict(state_dict, strict=False)
    print(f"加载图像判别器时缺失的权重键: {[key for key in msg.missing_keys if key not in loaded_keys]}")
    print(f"加载图像判别器时多余的权重键: {msg.unexpected_keys}")
    return image_disc

def init_video_disc(state_dict, video_disc, args):
    # 初始化视频判别器
    """中文说明：`init_video_disc` 读取、采样或保存视频帧序列；重点看 fps、帧号范围和输出维度。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if args.init_vdis == "no":
        video_disc_state_dict = {}
    elif args.init_vdis == "keep":
        video_disc_state_dict = state_dict["video_disc"]
    else:
        video_disc_state_dict = inflate_dis(state_dict["video_disc"], strategy=args.init_vdis)
    msg = video_disc.load_state_dict(video_disc_state_dict, strict=False)
    print(f"加载视频判别器时缺失的权重键: {msg.missing_keys}")
    print(f"加载视频判别器时多余的权重键: {msg.unexpected_keys}")
    return video_disc

def init_vit_from_image(state_dict, vae, image_disc, video_disc, args):
    """中文说明：`init_vit_from_image` 实现VideoVAE 权重加载和维度膨胀工具中的 `init_vit_from_image` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    if args.init_vgen == "no":
        vae_state_dict = state_dict["vae"]
        del vae_state_dict["encoder.to_patch_emb.1.weight"]
        del vae_state_dict["encoder.to_patch_emb.1.bias"]
        del vae_state_dict["encoder.to_patch_emb.2.weight"]
        del vae_state_dict["encoder.to_patch_emb.2.bias"]
        del vae_state_dict["encoder.to_patch_emb.3.weight"]
        del vae_state_dict["encoder.to_patch_emb.3.bias"]

        del vae_state_dict["decoder.to_pixels.0.weight"]
        del vae_state_dict["decoder.to_pixels.0.bias"]
        vae_state_dict = state_dict["vae"]

    elif args.init_vgen == "keep":
        vae_state_dict = state_dict["vae"]
    else:
        vae_state_dict = inflate_gen(state_dict["vae"], temporal_patch_size=args.temporal_patch_size, spatial_patch_size=args.patch_size, strategy=args.init_vgen, inflation_pe=args.inflation_pe)

    if args.vq_to_vae:
        del vae_state_dict["pre_vq_conv.1.weight"]
        del vae_state_dict["pre_vq_conv.1.bias"]

    msg = vae.load_state_dict(vae_state_dict, strict=False)
    print(f"加载 vae 时缺失的权重键: {msg.missing_keys}")
    print(f"加载 vae 时多余的权重键: {msg.unexpected_keys}")

    image_disc = init_image_disc(state_dict, image_disc, args)
    # 可选代码：video_disc = init_video_disc(state_dict, image_disc, args)  # 随机初始化视频判别器

    return vae, image_disc, video_disc

def load_cnn(model, state_dict, prefix, expand=False, use_linear=False):
    """中文说明：`load_cnn` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    delete_keys = []
    loaded_keys = []
    for key in state_dict:
        if key.startswith(prefix):
            _key = key[len(prefix):]
            if _key in model.state_dict():
                # 将 nn.Conv2d 或 nn.Linear 权重加载到 nn.Linear
                if use_linear and (".q.weight" in key or ".k.weight" in key or ".v.weight" in key or ".proj_out.weight" in key):
                    load_weights = state_dict[key].squeeze()
                elif _key.endswith(".conv.weight") and expand:
                    if model.state_dict()[_key].shape == state_dict[key].shape:
                        # 2D cnn 加载到 2D cnn
                        load_weights = state_dict[key]
                    else:
                        # 2D cnn 加载到 3D cnn
                        _expand_dim = model.state_dict()[_key].shape[2]
                        load_weights = state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                        load_weights = load_weights / _expand_dim # 中文标题：沿 expand 维度归一化
                else:
                    load_weights = state_dict[key]
                model.state_dict()[_key].copy_(load_weights)
                delete_keys.append(key)
                loaded_keys.append(prefix+_key)
            # 将 nn.Conv2d 权重加载到 Conv 类
            conv_list = ["conv"] if use_linear else ["conv", ".q.", ".k.", ".v.", ".proj_out.", ".nin_shortcut."]
            if any(k in _key for k in conv_list):
                if _key.endswith(".weight"):
                    conv_key = _key.replace(".weight", ".conv.weight")
                    if conv_key and conv_key in model.state_dict():
                        if model.state_dict()[conv_key].shape == state_dict[key].shape:
                            # 2D cnn 加载到 2D cnn
                            load_weights = state_dict[key]
                        else:
                            # 2D cnn 加载到 3D cnn
                            _expand_dim = model.state_dict()[conv_key].shape[2]
                            load_weights = state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                            load_weights = load_weights / _expand_dim # 中文标题：沿 expand 维度归一化
                        model.state_dict()[conv_key].copy_(load_weights)
                        delete_keys.append(key)
                        loaded_keys.append(prefix+conv_key)
                if _key.endswith(".bias"):
                    conv_key = _key.replace(".bias", ".conv.bias")
                    if conv_key and conv_key in model.state_dict():
                        model.state_dict()[conv_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+conv_key)
            # 将 nn.GroupNorm 权重加载到 Normalize 类
            if "norm" in _key:
                if _key.endswith(".weight"):
                    norm_key = _key.replace(".weight", ".norm.weight")
                    if norm_key and norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+norm_key)
                if _key.endswith(".bias"):
                    norm_key = _key.replace(".bias", ".norm.bias")
                    if norm_key and norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+norm_key)

    for key in delete_keys:
        del state_dict[key]

    return model, state_dict, loaded_keys

def init_cnn_from_image(state_dict, vae, image_disc, video_disc, args, expand=False):
    """中文说明：`init_cnn_from_image` 实现VideoVAE 权重加载和维度膨胀工具中的 `init_cnn_from_image` 步骤，供训练、推理或调试流程复用。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    vae.encoder, state_dict["vae"], loaded_keys1 = load_cnn(vae.encoder, state_dict["vae"], prefix="encoder.", expand=expand)
    vae.decoder, state_dict["vae"], loaded_keys2 = load_cnn(vae.decoder, state_dict["vae"], prefix="decoder.", expand=expand)
    loaded_keys = loaded_keys1 + loaded_keys2
    # 可选代码：msg = vae.load_state_dict(state_dict["vae"], strict=False)
    # 调试输出：print(f"加载 vae 时缺失的权重键: {[key for key in msg.missing_keys if key not in loaded_keys]}")
    # 调试输出：print(f"加载 vae 时多余的权重键: {msg.unexpected_keys}")
    vae, missing_keys = load_unstrictly(state_dict["vae"], vae, loaded_keys)

    if image_disc:
        image_disc = init_image_disc(state_dict, image_disc, args)
    ### 中文标题：随机初始化视频判别器
    # 可选条件：if video_disc:
    # 代码/形状说明：video_disc = init_video_disc(state_dict, image_disc, args)
    return vae, image_disc, video_disc

def resume_from_ckpt(state_dict, model_optims, load_optims=True):
    """中文说明：`resume_from_ckpt` 处理 checkpoint 保存/加载/恢复；重点看路径选择、key 匹配和缺失权重处理。

    新手提示：重点看 2D 权重如何 inflate 到 3D 卷积/注意力，以及 strict=False 时哪些键会被跳过。
    阅读重点：确认输入、输出和副作用，再回到调用方看它在整条链路中的位置。
    """
    all_missing_keys = []
    # 先加载权重
    for k in model_optims:
        if model_optims[k] and state_dict[k] and (not is_torch_optim_sch(model_optims[k])) and k in state_dict:
            model_optims[k], missing_keys = load_unstrictly(state_dict[k], model_optims[k])
            all_missing_keys += missing_keys

    if len(all_missing_keys) == 0 and load_optims:
        print("正在加载优化器状态")
        for k in model_optims:
            if model_optims[k] and state_dict[k] and is_torch_optim_sch(model_optims[k]) and k in state_dict:
                model_optims[k].load_state_dict(state_dict[k])
    else:
        print(f"存在缺失权重: {all_missing_keys}, load_optims={load_optims}，因此不加载优化器状态")
    return model_optims, state_dict["step"]

### 中文标题：旧版本
# 旧版函数草稿：def get_last_ckpt(root_dir):
# 代码/形状说明：if not os.path.exists(root_dir): return None, None
# 代码/形状说明：ckpt_files = {}
# 代码/形状说明：for dirpath, dirnames, filenames in os.walk(root_dir):
# 代码/形状说明：for filename in filenames:
# 代码/形状说明：if filename.endswith('.ckpt'):
# 代码/形状说明：num_iter = int(filename.split('-')[1].split('=')[1])
# 中文说明：ckpt_files[num_iter]=os.path.join(dirpath, filename)
# 代码/形状说明：iter_list = list(ckpt_files.keys())
# 代码/形状说明：if len(iter_list) == 0: return None, None
# 代码/形状说明：max_iter = max(iter_list)
# 代码/形状说明：return ckpt_files[max_iter], max_iter
