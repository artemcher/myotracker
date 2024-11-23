# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from myotracker.model.model_utils import sample_features4d
from myotracker.model.embeddings import get_2d_embedding
from myotracker.model.blocks import BasicEncoder, AttnBlock, CorrBlock


torch.manual_seed(0)

class MyoTracker(nn.Module):
    def __init__(self, stride=4):
        super(MyoTracker, self).__init__()
        self.stride = stride
        self.latent_dim = 64
        self.fnet = BasicEncoder(output_dim=self.latent_dim)
        self.model_res = (256,256)
        self.input_dim = 196 # latent (64) + LRR (100) + coord_embedding (32)
        self.updateformer = UpdateFormer(
            depth=4,
            input_dim=self.input_dim,
            hidden_size=64,
            num_heads=4,
            output_dim=2,
            mlp_ratio=1.0,
        )

    def get_track_features(self, fmaps, queried_coords):
        sample_track_features = sample_features4d(fmaps[:,0], queried_coords)
        return torch.unsqueeze(sample_track_features, 1)

    def forward(self, video: torch.Tensor, queries: torch.Tensor):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            is_train (bool, optional): enables training mode. Defaults to False.

        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
        """
        B, T, H, W = video.shape
        C = 1
        B, N, __ = queries.shape
        device = queries.device

        # B = batch size
        # T = number of frames in the window of the padded video
        # N = number of tracks
        # C = color channels (1)
        # E = positional embedding size
        # LRR = local receptive field radius: ((2*radius+1)**2) * levels
        # D = dimension of the transformer input tokens

        # video = [B,T,H,W], C is inserted when necessary
        # queries = [B,N,2]
        # coords_init = [B,T,N,2]

        queried_coords = queries / self.stride
        # compute convolutional features for the video
        fmaps = self.fnet(video.reshape(-1, C, H, W)).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        
        # compute track features
        track_features = self.get_track_features(fmaps, queried_coords).repeat(1, T, 1, 1)

        # get correlation maps, sample correlation features around each point
        corr_block = CorrBlock(fmaps, num_levels=4, radius=2)
        corr_block.corr(track_features)
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2)
        fcorrs = corr_block.sample(coords_init)  # [B*N,T,LRR], LRR = ((2*radius+1)**2)*levels

        # get the initial coordinates' embeddings
        coords_input = coords_init.permute(0, 2, 1, 3).reshape(B * N, T, 2)
        coords_input_emb = get_2d_embedding(coords_input, 16, cat_coords=False)  # [N,T,E]

        track_features_ = track_features.permute(0, 2, 1, 3).reshape(B * N, T, self.latent_dim)
        transformer_input = torch.cat([coords_input_emb, track_features_, fcorrs], dim=2)
        x = transformer_input.reshape(B, N, T, self.input_dim)  # [B*N,T, D] -> [B,N,T,D]
        
        delta_coords = self.updateformer(x)
        coords = coords_init + delta_coords
        coord_predicted = coords * self.stride
        return coord_predicted


class UpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(self, depth, input_dim, hidden_size, num_heads, output_dim, mlp_ratio):
        # depth: number of time and space layers (total = depth*2)
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        
        self.time_blocks = nn.ModuleList([
            AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.space_virtual_blocks = nn.ModuleList([
            AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, input_tensor):
        tokens = self.input_transform(input_tensor)
        B, N, T, C = tokens.shape
        
        for _, (time_block, space_block) in enumerate(zip(self.time_blocks, self.space_virtual_blocks)):
            time_tokens = tokens.reshape(B*N, T, C)  # [B,N,T,C] -> [B*N,T,C]
            time_tokens = time_block(time_tokens)
            time_tokens = time_tokens.reshape(B, N, T, C)  # [B*N,T,C] -> [B,N,T,C]

            space_tokens = time_tokens.permute(0, 2, 1, 3).reshape(B*T, N, C)  # [B,N,T,C] -> [B*T,N,C]
            space_tokens = space_block(space_tokens)
            tokens_out = space_tokens.reshape(B, T, N, C) #  [B*T,N,C] -> [B,T,N,C]
            tokens = tokens_out.permute(0, 2, 1, 3) # prepare for next time block

        flow = self.flow_head(tokens_out)
        return flow