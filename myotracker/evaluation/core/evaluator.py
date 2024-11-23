# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
from typing import Optional
import torch
from tqdm import tqdm
import numpy as np

#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from cotracker_small.datasets.echo_utils import dataclass_to_cuda_
from cotracker_small.utils.visualizer import Visualizer
from cotracker_small.models.core.model_utils import reduce_masked_mean
from cotracker_small.evaluation.core.eval_utils import compute_tapvid_metrics

import logging


class Evaluator:
    """
    A class defining the CoTracker evaluator.
    """

    def __init__(self, exp_dir) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualisations")

    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
        
        if "echo" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone().cpu()
            pred_traj = pred_trajectory.cpu()
            
            diff = pred_traj - traj
            eucl_dist = torch.sqrt(diff[...,0]**2 + diff[...,1]**2).float()
            dist_per_sample = eucl_dist.mean(dim=(1,2))
            out_metrics = {"dist_px": dist_per_sample.numpy()}

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                    metrics["avg"][metric_name] = {}
                #print(metrics.items())
                metrics["avg"][metric_name] = float(
                    [np.mean(v[metric_name]) for k, v in metrics.items() if k != "avg"][0]
                )

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualize_every: int = 1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
    ):
        metrics = {}

        vis = Visualizer(
            save_dir=self.exp_dir,
            fps=10,
        )

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            queries = sample.trajectory[:,0].to(device)

            pred_tracks = model(sample.video, queries)
            if "strided" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            seq_name = str(ind)
            if ind % visualize_every == 0:
                vis.visualize(
                    sample.video*255,
                    pred_tracks[0] if isinstance(pred_tracks, tuple) else pred_tracks,
                    filename=dataset_name + "_" + seq_name,
                    writer=writer,
                    step=step,
                )

            self.compute_metrics(metrics, sample, pred_tracks, dataset_name)
        return metrics
