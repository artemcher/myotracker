# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datetime
import random
import torch
import signal
import socket
import sys
import json

import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter
from lightning import Fabric
from torchinfo import summary

from myotracker.model.evaluation_predictor import EvaluationPredictor
from myotracker.model.myotracker_model import MyoTracker
from myotracker.utils.visualizer import Visualizer

from myotracker.datasets.echo_dataset import EchoDataset
from myotracker.evaluation.core.evaluator import Evaluator
from myotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from myotracker.model.losses import mean_absolute_error, refinement_loss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99995
    )
    return optimizer, scheduler

def forward_batch(batch, model, args):
    video = batch.video
    trajs_gt = batch.trajectory
    #device = video.device

    coord_predictions = model(video=video, queries=trajs_gt[:,0])        
    output = {"flow": {"predictions": coord_predictions.detach()}}
    output["flow"]["loss"] = mean_absolute_error(coord_predictions, trajs_gt).mean()
    return output


def run_test_eval(evaluator, model, dataloaders, writer, step):
    model.eval()
    for ds_name, dataloader in dataloaders:
        if ds_name == "echo_val":
            visualize_every = 5
            grid_size = 0

        predictor = EvaluationPredictor(
            model,
            grid_size=grid_size,
            local_grid_size=0,
            single_point=False,
        )
        if torch.cuda.is_available():
            predictor.model = predictor.model.cuda()

        metrics = evaluator.evaluate_sequence(
            model=predictor,
            test_dataloader=dataloader,
            dataset_name=ds_name,
            train_mode=True,
            writer=writer,
            step=step,
            visualize_every=visualize_every,
        )

        metrics = {f"{ds_name}_avg_{k}": v for k, v in metrics["avg"].items()}
        print(metrics)
        with open(f"checkpoints/log.txt", 'a') as metric_log:
            metric_log.write(f"Step {step} - {metrics}\n")
        writer.add_scalars(f"Eval_{ds_name}", metrics, step)


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        self.total_steps += 1

        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0

            self.running_loss[task_key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


class Lite(Fabric):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        if self.global_rank == 0:
            eval_dataloaders = []

            if "echo_val" in args.eval_datasets:
                data_root = os.path.join(args.dataset_root, "val")
                eval_dataset = EchoDataset(data_root=data_root, use_augs=False)
                eval_dataloader_echo_val = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=collate_fn,
                )
                eval_dataloaders.append(("echo_val", eval_dataloader_echo_val))

            evaluator = Evaluator(args.ckpt_path)

            visualizer = Visualizer(
                save_dir=args.ckpt_path,
                pad_value=80,
                fps=10,
                show_first_frame=0,
                tracks_leave_trace=0,
            )

        if args.model_name == "myotracker":
            model = MyoTracker(
                stride=args.model_stride,
            )
            #summary(model, input_size=[(1,64,256,256), (1,64,2)])
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Parameters: {params}")
        else:
            raise ValueError(f"Model {args.model_name} doesn't exist")

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()
        train_dataset = EchoDataset(
            data_root=f"{args.dataset_root}/train",
            crop_size=args.crop_size,
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            #sample_vis_1st_frame=args.sample_vis_1st_frame,
            use_augs=not args.dont_use_augs,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            collate_fn=collate_fn_train,
            drop_last=True,
        )

        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))
        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        if self.global_rank == 0:
            logger = Logger(model, scheduler)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".pt")
            logging.info("Loading checkpoint...")

            strict = True
            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=strict)

            logging.info(f"Done loading checkpoint")
        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        # model.cuda()
        model.train()

        save_freq = args.save_freq
        scaler = GradScaler(enabled=args.mixed_precision)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1

        while should_keep_training:
            epoch += 1
            for i_batch, batch in enumerate(tqdm(train_loader)):
                batch, gotit = batch
                if not all(gotit):
                    print("batch is None")
                    continue
                dataclass_to_cuda_(batch)

                optimizer.zero_grad()

                assert model.training

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = forward_batch(batch, model, args)
                    loss = 0
                    for k, v in output.items():
                        if "loss" in v:
                            loss += v["loss"]

                if self.global_rank == 0:
                    for k, v in output.items():
                        if "loss" in v:
                            logger.writer.add_scalar(
                                f"live_{k}_loss", v["loss"].item(), total_steps
                            )
                        if "metrics" in v:
                            logger.push(v["metrics"], k)

                    if total_steps % save_freq == save_freq - 1:
                        visualizer.visualize(
                            video=batch.video.clone()*255.0,
                            tracks=batch.trajectory.clone(),
                            filename="train_gt_traj",
                            writer=logger.writer,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone()*255.0,
                            tracks=output["flow"]["predictions"],
                            filename="train_pred_traj",
                            writer=logger.writer,
                            step=total_steps,
                        )
                    
                    if len(output) > 1:
                        logger.writer.add_scalar(f"live_total_loss", loss.item(), total_steps)
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1

                self.barrier()

                self.backward(scaler.scale(loss))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1
                if self.global_rank == 0:
                    if (i_batch >= len(train_loader) - 1) or (
                        total_steps == 1 and args.validate_at_start
                    ):
                        if (epoch + 1) % args.save_every_n_epoch == 0:
                            ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                            save_path = Path(
                                f"{args.ckpt_path}/myotracker_{ckpt_iter}.pth"
                            )

                            save_dict = {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "total_steps": total_steps,
                            }
                            logging.info(f"Saving file {save_path}")
                            self.save(save_path, save_dict)
                            torch.save( # save just the model too
                                model, f"{args.ckpt_path}/myotracker_{ckpt_iter}.pt"
                            )
                            #torch.jit.trace

                        if (epoch + 1) % args.evaluate_every_n_epoch == 0 or (
                            args.validate_at_start and epoch == 0
                        ):
                            run_test_eval(evaluator, model, eval_dataloaders, logger.writer, total_steps)
                            model.train()
                            torch.cuda.empty_cache()

                self.barrier()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
        if self.global_rank == 0:
            print("FINISHED TRAINING")

            PATH = f"{args.ckpt_path}/{args.model_name}_final.pth"
            torch.save(model.state_dict(), PATH)
            run_test_eval(evaluator, model, eval_dataloaders, logger.writer, total_steps)
            logger.close()


if __name__ == "__main__":
    #signal.signal(signal.SIGUSR1, sig_handler)
    #signal.signal(signal.SIGTERM, term_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="myotracker", help="model name")
    parser.add_argument("--restore_ckpt", help="path to restore a checkpoint")
    parser.add_argument("--ckpt_path", default=f"{os.getcwd()}/checkpoints/", help="path to save checkpoints")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size used during training."
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10, help="number of dataloader workers")

    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--lr", type=float, default=1e-3, help="max learning rate.")
    parser.add_argument("--wdecay", type=float, default=1e-5, help="Weight decay in optimizer.")
    parser.add_argument(
        "--num_steps", type=int, default=100000, help="length of training schedule."
    )
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=5,
        help="evaluate during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=5,
        help="save checkpoints during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=64,
        help="the number of trajectories to sample for training",
    )
    parser.add_argument( # bring your own data
        "--dataset_root",
        type=str,
        default="tracking_data/",
        help="path to all the datasets (train and eval)"
    )

    parser.add_argument("--sequence_len", type=int, default=64, help="train sequence length")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["echo_val"],
        help="what datasets to use for evaluation",
    )

    parser.add_argument(
        "--dont_use_augs",
        action="store_true",
        help="don't apply augmentations during training",
    )
    
    parser.add_argument(
        "--model_stride",
        type=int,
        default=4,
        help="stride of convolutional feature extractor",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[256, 256],
        help="crop videos to this resolution during training",
    )
    parser.add_argument(
        "--eval_max_seq_len",
        type=int,
        default=256,
        help="maximum length of evaluation videos",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)

    Lite(
        strategy="auto",
        devices=[0],
        accelerator="gpu",
        precision=32,
        num_nodes=args.num_nodes,
    ).run(args)
