import os
import cv2
import torch
import random
import numpy as np

from skimage.transform import resize
from scipy.ndimage import rotate, shift, zoom
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform

class ApplyToKey:
    def __init__(self, transform, key):
        self.key = key
        self.transform_to_apply = transform

    def __call__(self, inputs):
        frames, tracks = inputs
        if self.key == "all":
            # wrap it up (transform will take first element)
            return self.transform_to_apply((frames, tracks))

        if self.key == 0 or self.key == "frames":
            frames = self.transform_to_apply(frames)
        elif self.key == 1 or self.key == "tracks":
            tracks = self.transform_to_apply(tracks)
        return (frames, tracks)


class SaveExample:
    # Make and save videos from first N samples,
    # do nothing on later applications.

    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.saved = 0

    def save_video(self, overlays, filename, fps=15):
        T, H, W, _ = overlays.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=True)
        for frame in overlays:
            writer.write(frame)
        writer.release()

    def __call__(self, inputs):
        frames, tracks = inputs
        if self.saved >= self.num_samples:
            return (frames, tracks)

        np_frames = frames.cpu().numpy().transpose((0,2,3,1))[..., 0]
        np_frames = np.round(np_frames*255.0).astype(np.uint8)
        np_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in np_frames])
        np_tracks = np.round(tracks.cpu().numpy())

        np_frames, np_tracks = np_frames.astype(np.uint8), np_tracks.astype(np.int32)
        overlays = []
        
        for i in range(np_frames.shape[0]):
            frame = np_frames[i].copy()
            for p in range(np_tracks.shape[1]):
                pt = (np_tracks[i, p, 1], np_tracks[i, p, 0])  # (x, y)
                cv2.circle(frame, pt, radius=2, color=(0,255,0), thickness=-1)
            overlays.append(frame)
        overlays = np.array(overlays, dtype=np.uint8)

        save_path = "augmented_samples"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_video(overlays, f"{save_path}/augmented_sample_{self.saved+1}.mp4")
        self.saved += 1
        return (frames, tracks)


# ----------------- Transforms for frames only ----------------------- #
class RandomBlackout(Transform):
    # Turn a number of frames to 0 (black)

    def __init__(self, max_ratio=0.25):
        super().__init__()
        self.max_ratio = max_ratio

    def _transform(self, frames, params=None):
        T = frames.shape[0]
        random_ratio = random.random()*self.max_ratio
        num_affected = int(round(T * random_ratio))
        affected_frames = random.sample(range(T), k=num_affected)
        frames[affected_frames] *= 0
        return frames

class BlockErase(Transform):
    # Turn pixel blocks to 0 (consistent between frames)

    def __init__(self, min_h=2, max_h=32, min_w=2, max_w=32, max_blocks=10):
        super().__init__()
        self.min_h, self.max_h = min_h, max_h
        self.min_w, self.max_w = min_w, max_w
        self.max_blocks = max_blocks

    def _transform(self, frames, params=None):
        # frames: (T,H,W) or (T,C,H,W), dtype float or uint8
        h, w = frames.shape[-2], frames.shape[-1]
        num_blocks = random.randint(1, self.max_blocks)

        for _ in range(num_blocks):
            bh = random.randint(self.min_h, self.max_h)
            bw = random.randint(self.min_w, self.max_w)

            yc = random.randint(bh//2, h - bh//2 - 1)
            xc = random.randint(bw//2, w - bw//2 - 1)

            y1, y2 = yc - bh//2, yc + bh//2
            x1, x2 = xc - bw//2, xc + bw//2

            frames[..., y1:y2, x1:x2] = 0
        return frames


class BlockSwap(Transform):
    # Swap pixel blocks with each other

    def __init__(self, height=(2,32), width=(2,32), max_blocks=10):
        super().__init__()
        self.height = height
        self.width = width
        self.max_blocks = max_blocks

    def _transform(self, frames, params=None):
        # frames shape: (T,H,W) or (T,C,H,W)
        h, w = frames.shape[-2], frames.shape[-1]
        num_blocks = random.randint(1, self.max_blocks)

        for _ in range(num_blocks):
            bh = random.randint(self.height[0], self.height[1])
            bw = random.randint(self.width[0], self.width[1])

            # sample centers
            sy = random.randint(bh//2, h - bh//2 - 1)
            sx = random.randint(bw//2, w - bw//2 - 1)
            dy = random.randint(bh//2, h - bh//2 - 1)
            dx = random.randint(bw//2, w - bw//2 - 1)

            src = frames[..., sy-bh//2:sy+bh//2, sx-bw//2:sx+bw//2].clone()
            dst = frames[..., dy-bh//2:dy+bh//2, dx-bw//2:dx+bw//2].clone()

            frames[..., dy-bh//2:dy+bh//2, dx-bw//2:dx+bw//2] = src
            frames[..., sy-bh//2:sy+bh//2, sx-bw//2:sx+bw//2] = dst
        return frames


# ----------------- Transforms for both frames & tracks (used with ApplyToKey 'all') ----------------------- #
class TemporalReverse:
    def __init__(self):
        pass

    def __call__(self, inputs):
        frames, tracks = inputs
        return (frames.flip(dims=[0]), tracks.flip(dims=[0]))


class Translation:
    """
        Translate by fraction of image size in either direction.
    """
    def __init__(self, height=0.2, width=0.2):
        self.max_h_shift = height
        self.max_w_shift = width

    def __call__(self, inputs):
        frames, tracks = inputs

        h_shift = (random.random() * 2 - 1.0) * self.max_h_shift
        w_shift = (random.random() * 2 - 1.0) * self.max_w_shift
        h_shift_px = int(round(frames.shape[-2] * h_shift))
        w_shift_px = int(round(frames.shape[-1] * w_shift))

        # Shift using scipy and convert back to torch
        np_frames = frames.cpu().numpy()
        for i in range(np_frames.shape[0]):
            for c in range(np_frames.shape[1]):
                np_frames[i, c] = shift(
                    np_frames[i, c], 
                    shift=(h_shift_px, w_shift_px), 
                    order=1, mode='constant', cval=0
                )
        frames = torch.from_numpy(np_frames).to(dtype=frames.dtype, device=frames.device)

        tracks[..., 0] += h_shift_px
        tracks[..., 1] += w_shift_px
        return (frames, tracks)


class Rotation:
    """
    Rotate by [-max_angle:max_angle] degrees.
    """

    def __init__(self, max_angle=30):
        self.max_angle = max_angle

    def __call__(self, inputs):
        frames, tracks = inputs
        H, W = frames.shape[-2:]
        angle = random.randint(-self.max_angle, self.max_angle)

        # Rotate using scipy and convert back to torch
        np_frames = frames.cpu().numpy()
        for i in range(np_frames.shape[0]):
            for c in range(np_frames.shape[1]):
                np_frames[i, c] = rotate(np_frames[i, c], angle, order=1, reshape=False)
        frames = torch.from_numpy(np_frames).to(dtype=frames.dtype, device=frames.device)

        theta = np.radians(-angle)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s), (s, c)))

        np_tracks = tracks.cpu().numpy()
        np_tracks = np_tracks / np.array([H,W])[None, None, :]
        np_tracks = (np_tracks*2.0-1.0).dot(rot_mat)
        np_tracks = (np_tracks+1.0)/2.0 * np.array([H,W])[None, None, :]
        tracks = torch.from_numpy(np_tracks).to(dtype=tracks.dtype, device=tracks.device)
        return (frames, tracks)


class Zoom:
    """Zoom in/out horizontally and vertically by given factors."""

    def __init__(self, height=(0.5, 1.5), width=(0.5, 1.5)):
        self.height = height
        self.width = width

    # Calculate cropping or padding indices
    def _get_slices(self, orig, new):
        slice_new, slice_orig = slice(None), slice(None)
        if new < orig: # Pad
            pad_before = (orig - new) // 2
            pad_after = orig - new - pad_before
            slice_orig = slice(pad_before, pad_before + new)
        else: # Crop
            crop_before = (new - orig) // 2
            slice_new = slice(crop_before, crop_before + orig)
        return slice_orig, slice_new

    def __call__(self, inputs):
        frames, tracks = inputs
        _, _, H, W = frames.shape

        h_ratio = random.random() * (self.height[1]-self.height[0]) + self.height[0]
        w_ratio = random.random() * (self.width[1]-self.width[0]) + self.width[0]
        zoom_factors = (1, 1, h_ratio, w_ratio)
        
        zoomed = zoom(frames.cpu().numpy(), zoom_factors, order=1)
        _, _, new_H, new_W = zoomed.shape
                
        h_slice_orig, h_slice_zoom = self._get_slices(H, new_H)
        w_slice_orig, w_slice_zoom = self._get_slices(W, new_W)
        
        np_frames = frames.cpu().numpy()*0
        np_frames[:, :, h_slice_orig, w_slice_orig] = zoomed[:, :, h_slice_zoom, w_slice_zoom]
        frames = torch.from_numpy(np_frames).to(dtype=frames.dtype, device=frames.device)
        
        tracks = tracks / torch.tensor([H, W])[None, None, :]
        tracks = tracks * 2.0 - 1.0
        tracks = tracks * torch.tensor([h_ratio, w_ratio])[None, None, :]
        tracks = (tracks + 1.0) / 2.0 * torch.tensor([H, W])[None, None, :]
        return (frames, tracks)