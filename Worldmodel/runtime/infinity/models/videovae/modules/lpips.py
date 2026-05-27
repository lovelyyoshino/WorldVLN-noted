# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""LPIPS 感知损失实现。

该文件是 `PerceptualSimilarity` 的轻量改写版，用预训练 VGG 特征衡量两张图像在
“感知空间”中的距离，而不仅仅是像素级差异。
"""

import os, hashlib
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from infinity.models.videovae.utils.misc import bytenas_manager, set_tf32_flags

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def download(url, local_path, chunk_size=1024):
    """把远程权重流式下载到本地缓存目录。"""
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    """计算文件的 MD5 校验值。"""
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    """返回 LPIPS 权重路径；若本地不存在则自动下载。"""
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("正在下载 {} 模型：从 {} 到 {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

class ResNet50LPIPS(nn.Module):
    """基于 ResNet50 特征的简化感知距离。"""
    def __init__(self):
        """用 ResNet50 特征做一个简单的感知差异度量。"""
        super().__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.lpips_net = nn.Sequential(*(list(resnet50.children())[:-2]))
        self.lpips_loss = nn.MSELoss()

    def forward(self, input, target):
        """比较两张图像在 ResNet50 特征空间中的 MSE 差异。"""
        return self.lpips_loss(self.lpips_net(input), self.lpips_net(target),)

class LPIPS(nn.Module):
    """标准 LPIPS 感知损失。

    它会提取多层 VGG 特征，先做通道归一化，再经 1x1 线性层聚合，最后把各层结果求和。
    相比 L1/L2，LPIPS 更接近人眼对纹理与结构差异的感知。
    """
    def __init__(self, use_dropout=True, upcast_tf32=False):
        """构造 LPIPS 网络并加载预训练权重。"""
        super().__init__()
        self.upcast_tf32 = upcast_tf32
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 各层输出通道数。
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        """从缓存目录读取预训练 LPIPS 权重。"""
        ckpt = get_ckpt_path(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True), strict=False)
        print("已从 {} 加载预训练 LPIPS loss".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        """构造并返回一个已加载预训练权重的 LPIPS 实例。"""
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True), strict=False)
        return model

    def forward(self, input, target):
        """计算两张图像的 LPIPS 感知距离。"""
        with set_tf32_flags(not self.upcast_tf32):
            in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
            outs0, outs1 = self.net(in0_input), self.net(in1_input)
            feats0, feats1, diffs = {}, {}, {}
            lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            for kk in range(len(self.chns)):
                feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

            res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
            val = res[0]
            for l in range(1, len(self.chns)):
                # 代码/形状说明：print(res[l].shape)
                val += res[l]

            return val


class ScalingLayer(nn.Module):
    """LPIPS 输入归一化层。"""
    def __init__(self):
        """把 RGB 输入缩放到 LPIPS 训练时使用的归一化分布。"""
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        """执行逐通道平移与缩放。"""
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """单层 1x1 卷积头，用于把某层特征差异映射成标量权重。"""
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        """根据需要构造 dropout + 1x1 conv 的线性头。"""
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    """按 LPIPS 需求切片的 VGG16 特征提取器。"""
    def __init__(self, requires_grad=False, pretrained=True):
        """构造按层切片的 VGG16 特征提取器。"""
        super(vgg16, self).__init__()
        # 权重从本地缓存读取，避免重复联网下载。
        assert pretrained == True
        vgg_model = models.vgg16()
        vgg_model.load_state_dict(torch.load(bytenas_manager("checkpoints/vgg16-397923af.pth"), weights_only=True))
        vgg_pretrained_features = vgg_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """返回 LPIPS 所需的五组中间特征。"""
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    """按通道做 L2 归一化，使特征比较更接近余弦相似度。"""
    norm_factor = x.norm(p=2, dim=1, keepdim=True)
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    """对空间维求平均，把特征图压成标量响应。"""
    return x.mean([2,3],keepdim=keepdim)
