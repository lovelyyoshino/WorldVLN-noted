# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from infinity.schedules.dynamic_resolution import get_first_full_spatial_size_scale_index


def labels2image(all_indices, label_type='int_label', scale_schedule=None):
    """把离散标签索引解码回可视化图像，方便观察量化误差。"""
    summed_codes, recons_imgs = self.vae.decode_from_indices(all_indices, scale_schedule, label_type)
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

def features2image(raw_features):
    """把连续 VAE latent 特征直接解码成图像。"""
    recons_imgs = self.vae.decode(raw_features.squeeze(-3))
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

class SelfCorrection(object):
    """训练时对量化标签注入噪声，用于模拟自回归误差与纠错过程。"""
    def __init__(self, vae, args):
        """读取噪声注入配置，并保存 VAE 句柄供重建使用。"""
        self.noise_apply_layers = args.noise_apply_layers
        self.noise_apply_requant = args.noise_apply_requant
        self.noise_apply_strength = args.noise_apply_strength
        if not isinstance(self.noise_apply_strength, list):
            self.noise_apply_strength = str(self.noise_apply_strength)
            self.noise_apply_strength = list(map(float, self.noise_apply_strength.split(',')))
        if len(self.noise_apply_strength) == 1:
            self.noise_apply_strength = self.noise_apply_strength[0]
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.vae = vae
        print(f'self.noise_apply_strength: {self.noise_apply_strength}')

    def apply_noise_requant(self, bit_indices, quantized, args, device, si, lfq=None, noise_apply_strength=None):
        """随机扰动离散标签，并在需要时重新量化得到新的连续特征。"""
        if lfq is None:
            lfq = self.vae.quantizer.lfq
        if noise_apply_strength is None:
            noise_apply_strength = self.noise_apply_strength
        if isinstance(noise_apply_strength, list):
            noise_apply_strength = np.random.randint(0, max(1, 100 * noise_apply_strength[si]+1)) * 0.01
        else:
            noise_apply_strength = np.random.randint(0, max(1, 100 * noise_apply_strength+1)) * 0.01
        # `noise_apply_strength` 表示每个 token 被替换的概率。
        mask = torch.rand(*bit_indices.shape, device=device) < noise_apply_strength
        pred_bit_indices = bit_indices.clone()
        if args.num_of_label_value == 2:
            pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
        else:
            noise = torch.randint(0, args.num_of_label_value, bit_indices.shape, dtype=bit_indices.dtype, device=device)
            pred_bit_indices[mask] = noise[mask]
        if self.noise_apply_requant:
            quantized = lfq.indices_to_codes(pred_bit_indices, label_type = 'bit_label')
        return pred_bit_indices, quantized

    def visualize(self, vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices):
        """把原图、真值重建、预测重建并排保存，便于人工检查。"""
        gt_img = (inp_B3HW.squeeze(-3) + 1) / 2 * 255
        gt_img = gt_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)[:,:,::-1]
        recons_img_2 = labels2image(gt_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        recons_img_3 = labels2image(pred_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        cat_image = np.concatenate([gt_img, recons_img_2, recons_img_3], axis=1)
        save_path = osp.abspath('non_teacher_force.jpg')
        cv2.imwrite(save_path, cat_image)
        print(f'自校正可视化图已保存到 {save_path}')
        import pdb; pdb.set_trace()
        print(cat_image.shape)
