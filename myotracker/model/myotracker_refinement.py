# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from cotracker_small.models.core.model_utils import sample_features4d
from cotracker_small.models.core.embeddings import get_2d_embedding
from cotracker_small.models.core.cotracker.small_blocks import (
    Mlp,
    BasicEncoder,
    AttnBlock,
    CorrBlock,
    #Attention,
)

torch.manual_seed(0)


class CoTracker2(nn.Module):
    def __init__(
        self,
        stride=4,
        add_space_attn=True,
    ):
        super(CoTracker2, self).__init__()
        self.stride = stride
        self.latent_dim = 64
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(output_dim=self.latent_dim)
        self.model_res = (256,256)
        self.input_dim = 196 # latent + LRR + coord_embedding
        self.updateformer = EfficientUpdateFormer(
            depth=4,
            input_dim=self.input_dim,
            hidden_size=64,
            num_heads=4,
            output_dim=self.latent_dim+2,
            mlp_ratio=1.0,
        )
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.track_feat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        
    def forward_window(self, fmaps, coords, track_feat=None, iters=6):
        # B = batch size
        # S = number of frames in the window)
        # N = number of tracks
        # C = channels of a point feature vector
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # track_feat = B S N C

        B, S, N, _ = track_feat.shape
        
        corr_block = CorrBlock(fmaps, num_levels=4, radius=2)

        coord_predictions = []
        for __ in range(iters):
            coords = coords.detach()  # B S N 2
            corr_block.corr(track_feat)

            # Sample correlation features around each point
            fcorrs = corr_block.sample(coords)  # (B N) S LRR, LRR = (2*radius+1)*levels

            # Get the coords embeddings
            '''
            coords_input = coords.permute(0, 2, 1, 3).reshape(B * N, S, 2) # B S N 2
            coords_input_emb = get_2d_embedding(coords_input, 16, cat_coords=False)  # N S E
            '''
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            flow_emb = get_2d_embedding(flows, 16, cat_coords=False)  # N S E

            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim) # B S N 64
            transformer_input = torch.cat([flow_emb, track_feat_, fcorrs], dim=2)#.half()
            x = transformer_input.reshape(B, N, S, self.input_dim)  # (B N) S D -> B N S D
        
            delta = self.updateformer(x)
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            coords = coords + delta_coords
            coord_predictions.append(coords * self.stride)

            delta_feats_ = delta[..., 2:].reshape(B * N * S, self.latent_dim)
            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)
            track_feat_ = self.track_feat_updater(self.norm(delta_feats_)) + track_feat_
            track_feat = track_feat_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # (B N S) C -> B S N C
        return coord_predictions

    def get_track_feat(self, fmaps, queried_coords):
        sample_track_feats = sample_features4d(fmaps[:,0], queried_coords)
        return torch.unsqueeze(sample_track_feats, 1)

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
        # S = number of frames in the window of the padded video
        # N = number of tracks
        # C = color channels (1)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 2
        # coords_init = B S N 2

        queried_coords = queries / self.stride
        # Compute convolutional features for the video or for the current chunk in case of online mode
        fmaps = self.fnet(video.reshape(-1, C, H, W)).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        
        # We compute track features
        track_feat = self.get_track_feat(
            fmaps,
            queried_coords,
        ).repeat(1, T, 1, 1)

        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2)
        coords_predicted = self.forward_window(
            fmaps=fmaps,
            coords=coords_init,
            track_feat=track_feat,
        )
        return coords_predicted


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        depth=4, # number of time and space layers (total = depth*2)
        input_dim=196,
        hidden_size=64,
        num_heads=4,
        output_dim=2,
        mlp_ratio=1.0,
    ):
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
            time_tokens = tokens.reshape(B*N, T, C)  # B N T C -> (B N) T C
            time_tokens = time_block(time_tokens)
            time_tokens = time_tokens.reshape(B, N, T, C)  # (B N) T C -> B N T C

            space_tokens = time_tokens.permute(0, 2, 1, 3).reshape(B*T, N, C)  # B N T C -> (B T) N C
            space_tokens = space_block(space_tokens)
            tokens_out = space_tokens.reshape(B, T, N, C) #  (B T) N C -> B T N C
            tokens = tokens_out.permute(0, 2, 1, 3) # prepare for next time block

        flow = self.flow_head(tokens)
        return flow