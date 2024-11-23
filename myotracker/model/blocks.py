# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat

from myotracker.model.model_utils import bilinear_sampler


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="instance", stride=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        if norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=64, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = 16

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim)

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        
        self.layer1 = ConvBlock(self.in_planes, 16, norm_fn="instance", stride=1)
        self.layer2 = ConvBlock(16, 16, norm_fn="instance", stride=2)
        self.layer3 = ConvBlock(16, 32, norm_fn="instance", stride=2)
        self.layer4 = ConvBlock(32, 32, norm_fn="instance", stride=2)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        a = F.interpolate(a, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)
        b = F.interpolate(b, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)
        c = F.interpolate(c, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)
        d = F.interpolate(d, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class CorrBlock:
    def __init__(
        self,
        fmaps: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
    ):
        B, S, C, H, W = fmaps.shape
        
        self.S, self.C, self.H, self.W = S, C, H, W
        self.num_levels = num_levels
        self.radius = radius

        # placeholders
        self.fmaps_pyramid = [torch.zeros(1, dtype=fmaps.dtype, device=fmaps.device)]*self.num_levels
        self.corrs_pyramid = [torch.zeros(1, dtype=fmaps.dtype, device=fmaps.device)]*self.num_levels

        self.fmaps_pyramid[0] = fmaps
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            H, W = fmaps_.shape[-2:]
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid[i+1] = fmaps

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            H, W = corrs.shape[-2:]

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)#.to(coords.device)

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
            )
            corrs = corrs.reshape(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).reshape(B * N, S, -1)
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets
        
        for i, fmaps in enumerate(self.fmaps_pyramid):
            H, W = fmaps.shape[-2:]
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.tensor(C)
            self.corrs_pyramid[i] = corrs


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, query_dim, num_heads=4, dim_head=16):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.scale = 1.0 / (dim_head*dim_head)
        self.num_heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(query_dim, inner_dim)
        self.to_v = nn.Linear(query_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x):
        B, N1, C = x.shape
        h = self.num_heads
        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        k, v = self.to_k(x), self.to_v(x)

        N2 = x.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)

class AttnBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, mlp_ratio=1.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(in_features=hidden_size, out_features=int(hidden_size*mlp_ratio))

    def forward(self, x):
        x = self.norm1(x)
        attention_output = self.attn(x)
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x