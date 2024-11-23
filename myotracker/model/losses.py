# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from cotracker_small.models.core.model_utils import reduce_masked_mean

EPS = 1e-6

def refinement_loss(flow_preds, flow_gt, gamma=0.8):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    flow_loss = 0.0

    n_predictions = len(flow_preds)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]
        i_loss = torch.abs(flow_pred - flow_gt)  # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=3)  # B, S, N
        flow_loss += i_weight * i_loss
    flow_loss = flow_loss / n_predictions
    print(flow_loss.mean())
    return flow_loss

def mean_absolute_error(coord_preds, coord_gt):
    with torch.no_grad():
        diff = coord_preds - coord_gt
        diff_hypot = torch.sqrt(diff[...,0]**2 + diff[...,1]**2)
        print("Dist:", torch.mean(diff_hypot))

    abs_coord_error = (coord_preds - coord_gt).abs()
    coord_err_samples = abs_coord_error.mean(dim=(1,2,3))

    flow_preds = coord_preds[:,1:] - coord_preds[:,:-1]
    flow_gt = coord_gt[:,1:] - coord_gt[:,:-1]
    abs_flow_error = (flow_preds - flow_gt).abs()
    flow_err_samples = abs_flow_error.mean(dim=(1,2,3))
    return coord_err_samples + flow_err_samples
