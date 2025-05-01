import os
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize


class EchoDataset(Dataset):
    # Generic template for actual datasets to inherit from

    def __init__(self, 
        data_root, frame_size=(256, 256), num_frames=64, num_points=64,
        transform=True, cache_dataset=False, split="train", *args, **kwargs
    ):
        self.data_root = data_root
        self.split = split
        self.dataset_files = self._build_file_list(data_root)

        self.frame_size = frame_size
        self.num_frames = num_frames
        self.num_points = num_points
        self.transform = transform
        self.cache_dataset = cache_dataset

        self.cached = False
        if cache_dataset: # cache in a simple dictionary
            self.data_cache = {"frames": {}, "tracks": {}}
            dataset_name = self.__class__.__name__.replace('Dataset', '')
            for i in range(len(self.dataset_files)):
                frames, tracks = self.__getitem__(i)
                self.data_cache["frames"][self.dataset_files[i]] = frames # videos: [B, T, C=1, H, W]
                self.data_cache["tracks"][self.dataset_files[i]] = tracks # [B, T, N, 2]
                print(f"Caching the {dataset_name} {split} dataset: {i+1}/{len(self.dataset_files)}\r", end="")
            print()
            self.cached = True

    def _loader_function(self, file_path):
        # override for a given dataset, return: 
        #   frames  = [T, H, W] (dtype=np.uint8),
        #   tracks  = [T, N, 2] (dtype=np.float32)
        frames  = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1]), dtype=np.uint8)
        tracks  = np.zeros((self.num_frames, self.num_points, 2), dtype=np.float32)
        return frames, tracks

    def _build_file_list(self, data_root):
        # override later
        file_list = []
        return file_list

    def get_file_list(self):
        return self.dataset_files

    def get_config(self, config_dict=None):
        # override later
        if config_dict is None:
            config_dict = {}
        
        config_dict["data_path"] = self.data_root
        split_name = "validation" if "val" in self.split else "training"
        config_dict[split_name] = {"subjects": []}
        
        subjects_list = []
        for i in range(len(self.dataset_files)):
            subject = f"{i:04d}"
        config_dict[split_name]["subjects"] = subjects_list
        return config_dict

    def __len__(self):
        return len(self.dataset_files)

    def _getitem_from_cache(self, idx):
        frames  = self.data_cache["frames"][self.dataset_files[idx]]
        tracks  = self.data_cache["tracks"][self.dataset_files[idx]]

        frames, tracks = self._prepare_sample(frames, tracks)
        if self.transform is None: 
            frames = frames.float()/ 255.0
        else: # augment
            sample = (frames, tracks)
            sample = self.transform(sample)
            frames, tracks = sample
        return frames, tracks

    def _prepare_sample(self, frames, tracks):
        # prepare frames and tracks by:
        #   1. trimming frames down to *self.num_frames*
        #   (randomly skips frames and selects a slice)
        #   2. sampling tracks to conform to *self.num_points*

        T = frames.shape[0] # number of frames in loaded sample
        N = tracks.shape[1] # number of points/tracks in loaded sample

        if T > self.num_frames:
            if random.random() > 0.5:  # do skip
                num_skips = random.random() * (T - self.num_frames)
                num_keep = int(torch.round(torch.tensor(T - num_skips)).item())
                keep_idx = torch.sort(torch.randperm(T)[:num_keep])[0]
                frames, tracks = frames[keep_idx], tracks[keep_idx]

            # randomly select a slice of num_frames frames
            T = frames.shape[0]
            start_idx = torch.randint(0, T - self.num_frames + 1, (1,)).item()
            frames = frames[start_idx : start_idx + self.num_frames]
            tracks = tracks[start_idx : start_idx + self.num_frames]

        if N > self.num_points:
            # randomly select points
            keep_track_idx = torch.randperm(N)[:self.num_points]
            tracks = tracks[:, keep_track_idx, :]

        if N < self.num_points:
            # oversample (duplicate) some tracks
            num_duped = self.num_points - N
            dupe_track_idx = torch.randint(0, N, (num_duped,))
            duped_tracks = tracks[:, dupe_track_idx, :]
            duped_tracks = duped_tracks.reshape(self.num_frames, num_duped, 2) # just in case
            tracks = torch.cat([tracks, duped_tracks], dim=1)
        return frames, tracks

    def _resize_data(self, frames, tracks):
        new_dims = self.frame_size
        tracks_frac = tracks / np.array([frames.shape[-2], frames.shape[-1]])[None, None, :]
        frames = np.array([resize(frame, (new_dims), preserve_range=True) for frame in frames]).astype(np.uint8)
        tracks = tracks_frac * np.array([self.frame_size[0], self.frame_size[1]])[None, None, :]
        return frames, tracks

    def __getitem__(self, idx):
        # get one sample of frames and tracks
        # if caching enabled, cache first - and if already cached, then load from cache
        # returns frames, queries, tracks (where queries=tracks[0])

        if self.cached: # load from cache (un-augmented)
            frames, tracks = self._getitem_from_cache(idx)
            queries = tracks[0]
            return frames, queries, tracks
        
        # load normally, whether for use or for caching
        frames, tracks = self._loader_function(self.dataset_files[idx])
        frames, tracks = self._resize_data(frames, tracks)

        # if too short, keep padding with reverses
        frames_pad, tracks_pad = frames, tracks
        while len(frames) <= self.num_frames: 
            frames_pad = np.flip(frames_pad, axis=0)[1:]
            frames = np.concatenate([frames, frames_pad], axis=0)
            tracks_pad = np.flip(tracks_pad, axis=0)[1:]
            tracks = np.concatenate([tracks, tracks_pad], axis=0)

        T, H, W = frames.shape
        frames = torch.from_numpy(np.uint8(frames)).reshape(T, 1, H, W)
        tracks = torch.from_numpy(np.float32(tracks))
        
        if self.cache_dataset and not self.cached:
            return frames, tracks # will cache without transforms, just padded

        frames, tracks = self._prepare_sample(frames, tracks)
        if self.transform is None:
            frames = frames.float()/255.0
        else: # transforming but not caching
            sample = (frames, tracks)
            sample = self.transform(sample)
            frames, tracks = sample
        queries = tracks[0]
        return frames, queries, tracks


class HUNT4RVDataset(EchoDataset):
    # For using annotated and prepared RV data from HUNT4
    # (you don't have it)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_file_list(self, data_root):
        file_list = []
        for exam_name in os.listdir(self.data_root):
            exam_path = f"{self.data_root}/{exam_name}"
            file_path = f"{exam_path}/{os.listdir(exam_path)[0]}"
            with h5py.File(file_path, 'r') as h5_file:
                tracks = np.array(h5_file['meshes'])
                if tracks.shape[0] < 30:
                    continue
            file_list.append(file_path)
        return file_list

    def _loader_function(self, file_path):
        with h5py.File(file_path, "r") as h5_file:
            frames = np.array(h5_file["frames"]).astype(np.uint8) # [T,H,W]
            tracks = np.array(h5_file["meshes"]).astype(np.float32) # [T,N,2]
            tracks = tracks * np.array(frames.shape[-2:])[None, None, :] # in pixels
            #tracks = np.flip(tracks, axis=-1) # (x,y) -> (y,x)
        return frames, tracks

    def get_config(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        
        config_dict["data_path"] = self.data_root
        split_name = "validation" if "val" in self.split else "training"
        config_dict[split_name] = {"subjects": []}
        
        subjects_list = []
        for file_path in self.dataset_files:
            subject = file_path.split("/")[-1].replace(".h5", "")
            subjects_list.append(subject)
        config_dict[split_name]["subjects"] = subjects_list
        return config_dict