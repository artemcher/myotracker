# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import torch
import torch.nn.functional as F
from myotracker.models.myotracker import MyoTracker
from myotracker.models.model_utils import smart_cat, get_points_on_a_grid

def build_myotracker(checkpoint=None):
    #if checkpoint is not None:
    if checkpoint.split('.')[-1] == "pth":
        myotracker = MyoTracker(stride=4)
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cuda:0")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        myotracker.load_state_dict(state_dict)

    else:
        myotracker = torch.load(checkpoint, map_location="cuda:0")
    return myotracker.cuda()

class MyoTrackerPredictor(torch.nn.Module):
    def __init__(self, checkpoint="../checkpoints/myotracker.pt"):
        super().__init__()
        self.support_grid_size = 0
        self.interp_shape = (256,256)
        self.model = build_myotracker(checkpoint)
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video,  # [B,T,H,W]
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None, # [B,N,2]
        grid_size: int = 0, # grid of N*N points from the first frame
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
    ):

        tracks = self._compute_sparse_tracks(video, queries)
        return tracks

    def _compute_sparse_tracks(self, video, queries):
        B, T, H, W = video.shape
        C = 1

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            queries = queries.clone()
            queries *= queries.new_tensor(
                [(self.interp_shape[1]-1) / (W-1), (self.interp_shape[0]-1) / (H-1)]
            )
        tracks = self.model.forward(video, queries=queries)

        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

        tracks *= tracks.new_tensor(
            [(W-1) / (self.interp_shape[1]-1), (H-1) / (self.interp_shape[0]-1)]
        )
        return tracks

if __name__=="__main__":
    predictor = MyoTrackerPredictor()
    video = torch.rand((1,64,256,256), dtype=torch.float32)
    queries = torch.rand((1,88,2), dtype=torch.float32)
    predictor.forward(video, queries)
