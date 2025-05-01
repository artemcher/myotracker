import torch
import torch.nn as nn

class TrackLoss(nn.Module):
    # Composite loss based on mean absolute error.
    # Minimizes the error between pred/true point coordinates,
    #  but also considers the difference between motion patterns within
    #  both true and pred tracks.
    # The second component may introduce (excessive?) regularization.

    def __init__(self):
        super(TrackLoss, self).__init__()
        self.factor = 0.25 # regularization factor

    def forward(self, tracks_pred, tracks_gt):
        '''
        with torch.no_grad():
            diff = tracks_pred - tracks_gt
            diff_hypot = torch.sqrt(diff[...,0]**2 + diff[...,1]**2)
            print("Dist:", torch.mean(diff_hypot))
        '''

        # error in true/pred points
        track_error = (tracks_pred - tracks_gt).abs().mean(dim=(1,2,3))

        # diff between motion patterns (might over-regularize?)
        flow_pred = tracks_pred[:,1:] - tracks_pred[:,:-1]
        flow_gt   = tracks_gt[:,1:] - tracks_gt[:,:-1]
        flow_error = (flow_pred - flow_gt).abs().mean(dim=(1,2,3))

        track_error = track_error * (1.0-self.factor)
        flow_error = flow_error * self.factor
        return track_error.mean() + flow_error.mean()


class RefinementLoss(nn.Module):
    ''' One of the original losses from CoTracker2 for iterative refinement. '''
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.

    def __init__(self):
        super(RefinementLoss, self).__init__()
        self.gamma = 0.8

    def forward(self, tracks_preds, tracks_gt):
        """Loss function defined over sequence of flow predictions"""
        B, S, N, D = tracks_gt.shape
        flow_loss = 0.0

        n_predictions = len(tracks_preds)
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            tracks_pred = tracks_preds[i]
            i_loss = torch.abs(tracks_pred - tracks_gt)  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            flow_loss += i_weight * i_loss
        flow_loss = flow_loss / n_predictions
        return flow_loss.mean()
