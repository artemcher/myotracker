import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch import optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from myotracker.data_augments import *
from myotracker.datasets import HUNT4RVDataset
from myotracker.network.losses import TrackLoss, RefinementLoss
from myotracker.network.myotracker import MyoTracker
from myotracker.network.myotracker_iterative import MyoTrackerIter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# lightning module
class MyoTrackerModule(pl.LightningModule):
    def __init__(self, model_class, train_config=None):
        super().__init__()
        self.criterion = TrackLoss() # for normal MyoTracker
        if model_class is MyoTrackerIter:
            self.criterion = RefinementLoss()

        self.train_config = train_config
        if self.train_config is None:
            self.train_config = { # defaults
                'batch_size': 8,
                'frames_shape': (64, 1, 256, 256), # [T, C, H, W]
                'tracks_shape': (64, 64, 2), # [T, N, 2]
                'start_learning_rate': 0.001,
                'lr_decay_step': 0.999975,
                'epochs': 100,
                'early_stop_patience': 20
            }
        self.model = model_class(input_shape=self.train_config["frames_shape"])

    def get_train_config(self):
        return self.train_config

    def forward(self, frames, queries):
        return self.model(frames, queries)

    def training_step(self, batch, batch_idx):
        frames, queries, tracks = batch
        tracks_pred = self(frames, queries)
        loss = self.criterion(tracks_pred, tracks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, queries, tracks = batch
        tracks_pred = self(frames, queries)
        loss = self.criterion(tracks_pred, tracks)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.train_config["start_learning_rate"]
        lr_decay_step = self.train_config["lr_decay_step"]
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6, eps=1e-8)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_step)
        return {"optimizer": optimizer, "scheduler": scheduler}
# ---------------------------------------------------------------------------------------------- #

def train_myotracker(data_root, dataset_class, model_class, train_config=None, transform=None, log_dir="logs/test_run"):
    # trainer function

    tracker_module = MyoTrackerModule(model_class, train_config)
    config_dict = tracker_module.get_train_config()
    
    train_path, val_path = f"{data_root}/train", f"{data_root}/val"
    T, C, H, W = config_dict["frames_shape"]
    _, N, _ = config_dict["tracks_shape"]
    B = config_dict["batch_size"]

    # dataset and loaders
    dataset_train = dataset_class(
        train_path, split='train', 
        num_frames=T, num_points=N, frames_size=(H,W), 
        transform=transform, cache_dataset=True
    )
    dataset_val = dataset_class(
        val_path, split='val', 
        num_frames=T, num_points=N, frames_size=(H,W), 
        transform=None, cache_dataset=True
    )

    train_loader = DataLoader(dataset_train, batch_size=B, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=B, shuffle=False, num_workers=8)

    # logging/checkpoints
    config_dict = dataset_train.get_config(config_dict) # append train subjects
    config_dict = dataset_val.get_config(config_dict) # append val subjects

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/checkpoints',
        filename='{epoch:03d}_{val_loss:.4f}', # not an f-string, on purpose
        monitor='val_loss', mode='min', save_top_k=1, verbose=True
    )
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=config_dict["early_stop_patience"], mode='min', verbose=True)

    # train model
    trainer = pl.Trainer(
        max_epochs=config_dict["epochs"], callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=[0] 
    )
    trainer.fit(tracker_module, train_loader, val_loader)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=4) # write config for posterity

    # load best model & save
    best_model_path = checkpoint_callback.best_model_path
    tracker_module.load_state_dict(torch.load(best_model_path)['state_dict'])
    return tracker_module

if __name__ == "__main__":
    # -------------------------------------- Options ------------------------------------- #
    hunt4rv_path = "../../rv-tracking/data/prepared_data_small"
    
    # available (make your own dataset)
    data_paths      = [hunt4rv_path]
    dataset_classes = [HUNT4RVDataset]
    model_classes   = [MyoTracker, MyoTrackerIter]
    
    # selected
    data_path     = data_paths[0]
    dataset_class = dataset_classes[0]
    model_class   = model_classes[0]
    # ------------------------------------------------------------------------------------ #

    train_config = { # can set here or use defaults at the top
        'batch_size': 8,
        'frames_shape': (64, 1, 256, 256), # [T, C, H, W]
        'tracks_shape': (64, 64, 2), # [T, N, 2]
        'start_learning_rate': 0.001,
        'lr_decay_step': 0.99992, # RE-CALCULATE FOR YOUR DATA
        'epochs': 5,#00,
        'early_stop_patience': 100
    }

    transform = v2.Compose([
        v2.RandomApply([ApplyToKey(BlockErase(max_blocks=10), key="frames")], p=0.5),
        v2.RandomApply([ApplyToKey(BlockSwap(max_blocks=10), key="frames")], p=0.5),
        v2.RandomApply([ApplyToKey(RandomBlackout(max_ratio=0.25), key="frames")], p=0.5),
        
        v2.RandomApply([ApplyToKey(TemporalReverse(), key="all")], p=0.5),
        v2.RandomApply([ApplyToKey(Rotation(max_angle=30), key="all")], p=0.5),
        v2.RandomApply([ApplyToKey(Zoom(height=(0.4, 1.33), width=(0.4, 1.33)), key="all")], p=0.5),
        v2.RandomApply([ApplyToKey(Translation(height=0.25, width=0.25), key="all")], p=0.5),

        v2.RandomApply([ApplyToKey(v2.JPEG(quality=(40,75)), key="frames")], p=0.5),
        v2.RandomApply([ApplyToKey(v2.ColorJitter(0.3,0.3,0.3), key="frames")], p=0.5),
        v2.RandomApply([ApplyToKey(v2.GaussianBlur(3, sigma=(0.1,2.0)), key="frames")], p=0.5),

        ApplyToKey(v2.ToDtype(torch.float32, scale=True), key="frames"),
        v2.RandomApply([ApplyToKey(v2.GaussianNoise(0.0, sigma=0.1, clip=True), key="frames")], p=0.5),
        ApplyToKey(SaveExample(num_samples=10), key="all"),
    ])

    dataset_name = dataset_class.__name__.replace('Dataset', '').lower()
    log_dir = f'logs/run_{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    tracker_module = train_myotracker(data_path, dataset_class, model_class, train_config, transform, log_dir)
    torch_model = tracker_module.model # extract torch model itself from module 
    torch_model.eval()
    
    # save torch model (just in case) and script it (might fail)
    # scripted model is self-contained & can be used independently from the code
    torch.save(torch_model, os.path.join(log_dir, "myotracker.pt"))
    scripted_model = torch.jit.script(torch_model)
    scripted_model.save(os.path.join(log_dir, "myotracker_scripted.pt"))