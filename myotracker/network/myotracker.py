import torch
import torch.nn as nn
import torch.nn.functional as F

from myotracker.network.model_utils import sample_features4d
from myotracker.network.embeddings import get_2d_embedding
from myotracker.network.blocks import ConvEncoder, AttnBlock, CorrBlock

torch.manual_seed(0)

class MyoTracker(nn.Module):
    # Simplified point tracking architecture based on CoTracker/CoTracker2
    
    def __init__(self, input_shape=(64, 1, 256, 256)):
        super(MyoTracker, self).__init__()
        self.stride = 4
        self.latent_dim = 64

        self.encoder = ConvEncoder(output_dim=self.latent_dim)
        self.model_res = input_shape[-2:]
        
        self.input_dim = 196 # latent (64) + LRR (100) + coord_embedding (32)
        self.updateformer = TrackFormer(
            depth=4,
            input_dim=self.input_dim,
            hidden_size=64,
            num_heads=4,
            output_dim=2,
            mlp_ratio=1.0,
        )

    def get_track_features(self, feature_maps, queried_coords):
        sample_track_features = sample_features4d(feature_maps[:,0], queried_coords)
        return torch.unsqueeze(sample_track_features, 1)

    def forward(self, frames: torch.Tensor, queries: torch.Tensor):
        B, T, C, H, W = frames.shape
        B, N, __ = queries.shape # [B, N, 2], (y,x) - order
        device = queries.device

        queried_coords = queries / self.stride
        feature_maps = self.encoder(frames.reshape(-1, C, H, W)).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        
        # compute track features
        track_features = self.get_track_features(feature_maps, queried_coords).repeat(1, T, 1, 1)

        # get correlation maps, sample correlation features around each point
        corr_block = CorrBlock(feature_maps, num_levels=4, radius=2)
        corr_block.corr(track_features)
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2) # [B, T, N, 2]
        fcorrs = corr_block.sample(coords_init)  # [B*N,T,LRR], LRR = ((2*radius+1)**2)*levels

        # get the initial coordinates' embeddings
        coords_input = coords_init.permute(0, 2, 1, 3).reshape(B * N, T, 2)
        coords_input_emb = get_2d_embedding(coords_input, 16, cat_coords=False)  # [N,T,E]

        # assemble input to transformer
        track_features_ = track_features.permute(0, 2, 1, 3).reshape(B * N, T, self.latent_dim)
        transformer_input = torch.cat([coords_input_emb, track_features_, fcorrs], dim=2)
        x = transformer_input.reshape(B, N, T, self.input_dim)  # [B*N,T, D] -> [B,N,T,D]
        
        # predict track coordinates directly
        # -> see myotracker_iterative.py for version with refinement iterations
        delta_coords = self.updateformer(x)
        coords = coords_init + delta_coords
        coord_predicted = coords * self.stride
        return coord_predicted


class TrackFormer(nn.Module):
    """
    Simple transformer for estimating tracks from different features.
    """

    def __init__(self, depth, input_dim, hidden_size, num_heads, output_dim, mlp_ratio):
        # depth: number of time and space layers (total layers = depth*2)
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        
        self.time_blocks = nn.ModuleList([
            AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.space_blocks = nn.ModuleList([
            AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, input_tensor):
        tokens = self.input_transform(input_tensor)
        B, N, T, C = tokens.shape
        
        # alternate between time- and space-attention blocks
        for _, (time_block, space_block) in enumerate(zip(self.time_blocks, self.space_blocks)):
            time_tokens = tokens.reshape(B*N, T, C)  # [B,N,T,C] -> [B*N,T,C]
            time_tokens = time_block(time_tokens)
            time_tokens = time_tokens.reshape(B, N, T, C)  # [B*N,T,C] -> [B,N,T,C]

            space_tokens = time_tokens.permute(0, 2, 1, 3).reshape(B*T, N, C)  # [B,N,T,C] -> [B*T,N,C]
            space_tokens = space_block(space_tokens)
            tokens_out = space_tokens.reshape(B, T, N, C) #  [B*T,N,C] -> [B,T,N,C]
            tokens = tokens_out.permute(0, 2, 1, 3) # prepare for next time block

        flow = self.flow_head(tokens_out)
        return flow