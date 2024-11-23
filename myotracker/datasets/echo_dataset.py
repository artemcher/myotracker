# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import cv2
import numpy as np
import h5py

from myotracker.datasets.utils import MyoTrackerData
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image
from skimage.transform import resize

from myotracker.datasets.augmentation import DataAugmentor


class MyoTrackerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(256, 256),
        seq_len=64,
        traj_per_sample=88,
        use_augs=True,
    ):
        super(MyoTrackerDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.use_augs = use_augs
        self.crop_size = crop_size

        # mlmia augmentation
        self.augmentor = self.make_mlmia_augmentor()

        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1 / 3.14)
        self.blur_aug = GaussianBlur(3, sigma=(0.5, 1.5))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [8, 32]
        self.eraser_max = 20

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [8, 32]
        self.replace_max = 20

    #def getitem_helper(self, index):
    #    return NotImplementedError

    def __getitem__(self, index):
        gotit = False

        sample, gotit = self.getitem_helper(index)
        
        if not gotit:
            print("warning: sampling failed")
            # fake sample, so we can still collate
            sample = MyoTrackerData(
                video=torch.zeros((self.seq_len, 3, self.crop_size[0], self.crop_size[1])),
                trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
            )

        return sample, gotit

    def add_photometric_augs(self, frames, trajs, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(frames)
        H, W = frames[0].shape[:2]
        assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            frames = [frame.astype(np.float32) for frame in frames]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(frames[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        frames[i][y0:y1, x0:x1, :] = mean_color

            frames = [frame.astype(np.uint8) for frame in frames]

        if replace:
            frames_alt = [
                np.array(self.photo_aug(Image.fromarray(frame)), dtype=np.uint8) for frame in frames
            ]
            frames_alt = [
                np.array(self.photo_aug(Image.fromarray(frame)), dtype=np.uint8) for frame in frames_alt
            ]

            ############ replace transform (per image after the first) ############
            frames = [frame.astype(np.float32) for frame in frames]
            frames_alt = [frame.astype(np.float32) for frame in frames_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = frames_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        frames[i][y0:y1, x0:x1, :] = rep

            frames = [frame.astype(np.uint8) for frame in frames]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            frames = [np.array(self.photo_aug(Image.fromarray(frame)), dtype=np.uint8) for frame in frames]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            frames = [np.array(self.blur_aug(Image.fromarray(frame)), dtype=np.uint8) for frame in frames]

        return np.array(frames, np.float32), np.array(trajs, np.float32)
        

    def crop(self, frames, trajs):
        T, N, _ = trajs.shape

        S = len(frames)
        H, W = frames[0].shape[1:]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else np.random.randint(0, H_new - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W_new else np.random.randint(0, W_new - self.crop_size[1])
        frames = [frame[0, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for frame in frames]
        frames = np.array(frames, np.float32)[:,None,...]

        trajs[:, :, 0] -= x0#/self.crop_size[0]
        trajs[:, :, 1] -= y0#/self.crop_size[1]

        return frames, trajs

    def make_mlmia_augmentor(self):
        data_augmentor = DataAugmentor()
        #data_augmentor.add_gaussian_shadow(apply_to=(0,), augmentation_chance=0.5)
        #data_augmentor.add_speckle_reduction(apply_to=(0,), augmentation_chance=0.5)
        data_augmentor.add_gamma_transformation(apply_to=(0,), low=0.75, high=1.25, augmentation_chance=0.5)
        data_augmentor.add_nonlinear_colormap(apply_to=(0,), augmentation_chance=0.5)

        data_augmentor.add_rotation(max_angle=30, graph_indices=(1,), augmentation_chance=0.5)
        data_augmentor.add_scale(min_scale_rate=0.75, max_scale_rate=1.25, graph_indices=(1,), augmentation_chance=0.5)
        data_augmentor.add_translation(max_height=0.2, max_width=0.2, graph_indices=(1,), augmentation_chance=0.5)

        #data_augmentor.add_jpeg_compression(apply_to=(0,), augmentation_chance=0.5)
        data_augmentor.add_random_noise(apply_to=(0,), augmentation_chance=0.5)
        #data_augmentor.add_blurring(apply_to=(0,), augmentation_chance=0.5)
        return data_augmentor

    def resize_data(self, frames, trajs):
        trajs_frac = trajs / np.array([frames.shape[2], frames.shape[1]])[None,None,:]
        frames_resized = resize(frames, (frames.shape[0],256,256,3), preserve_range=True)
        trajs_new = trajs_frac * np.array([self.crop_size[0], self.crop_size[1]])[None,None,:]
        return frames_resized, trajs_new

class EchoDataset(MyoTrackerDataset):
    def __init__(
        self,
        data_root,
        crop_size=(256, 256),
        seq_len=64,
        traj_per_sample=88, # max tracks (pad to this if not enough, cut off if more)
        use_augs=False,
    ):
        super(EchoDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            use_augs=use_augs,
        )

        self.traj_per_sample = traj_per_sample
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.seq_names = self.build_file_list()
        print("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    def build_file_list(self):
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


    def getitem_helper(self, index):
        gotit = True
        seq_name = self.seq_names[index]

        with h5py.File(seq_name, 'r') as h5_file:
            frames = np.array(h5_file['frames']).astype(np.float32) #/ 255.0 # (T,H,W)
            tracks = np.array(h5_file['meshes']) # (T,N,2)
        frames = np.array([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in frames]) # (T,H,W,3)
        
        while len(frames) <= self.seq_len: # if too short
            frames = np.concatenate([frames, frames[-2:0:-1]], axis=0)
            tracks = np.concatenate([tracks, tracks[-2:0:-1]], axis=0)
        
        if len(frames) > self.seq_len: # will trigger pretty much always
            # randomly skip frames, then take a part of remaining sequence
            num_frames_remove = (len(frames) - self.seq_len) // 2
            if num_frames_remove > 2: # remove random frames
                keep_indices = np.sort(np.random.choice(len(frames), len(frames)-num_frames_remove, replace=False))
                frames = frames[keep_indices]
                tracks = tracks[keep_indices]

            start_ind = np.random.choice(len(frames) - self.seq_len, 1)[0]
            frames = frames[start_ind : start_ind + self.seq_len]
            tracks = tracks[start_ind : start_ind + self.seq_len]
               

        if self.use_augs:
            frames, tracks = np.array(frames, np.float32), np.array(tracks, np.float32)
            frames, tracks = self.augmentor.transform([frames/255.0, tracks])
            
            frames, tracks = self.add_photometric_augs((frames*255.0).astype(np.uint8), tracks)
            frames = frames / 255.0
            
            # 50% chance to reverse time
            if np.random.rand(1) > 0.5:
                frames = frames[::-1]
                tracks = tracks[::-1]

            '''
            if np.random.rand(1) > 0.5:
                frac = np.random.rand(1)*0.25 # turn fraction of tracks to zero
                frac = np.maximum(1, int(tracks.shape[1]*frac))
                nullify_points = np.random.choice(tracks.shape[1], frac, replace=False)
                tracks[:, nullify_points] *= 0.0
            '''
            '''
            if np.random.rand(1) > 0.5:
                frac = np.random.rand(1)*0.25 # blackout fraction of frames
                frac = np.maximum(1, int(len(frames)*frac))
                blackout_frames = np.random.choice(len(frames), frac, replace=False)
                frames[blackout_frames] *= 0.0
            '''

            # randomize track order
            tracks = tracks.transpose((1,0,2))
            np.random.shuffle(tracks)
            tracks = tracks.transpose((1,0,2))
        else:
            frames = frames / 255.0

        tracks = np.flip(tracks, -1) # [y,x] => [x,y]
        tracks = tracks * np.array([frames.shape[2], frames.shape[1]])[None,None,:]
        frames, tracks = self.resize_data(frames, tracks)
        frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])
        
        # pad or cut off tracks to same number
        num_tracks = tracks.shape[1]
        if num_tracks < self.traj_per_sample:
            pad_tracks_indices = np.random.choice(num_tracks, self.traj_per_sample-num_tracks)
            pad_tracks = tracks[:, pad_tracks_indices]
            tracks = np.concatenate([tracks, pad_tracks], axis=1)
        else:
            tracks = tracks[:, :self.traj_per_sample]

        trajs = torch.from_numpy(tracks).float()
        frames = torch.from_numpy(np.array(frames, np.float32)).float()

        sample = MyoTrackerData(
            video=frames,
            trajectory=trajs,
            seq_name=seq_name,
        )
        return sample, gotit

    def __len__(self):
        return len(self.seq_names)