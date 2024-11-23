import numpy as np
import cv2
import PIL
import copy

from io import BytesIO
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline, griddata
from skimage.util import random_noise, img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.ndimage import rotate, shift
from skimage.transform import resize
from typing import Callable
from functools import wraps, partial
from abc import ABC, abstractmethod

from mlmia.utilities import gaussian
from mlmia.config import TaskType, is_height_width_oriented
from mlmia.engine import DataArray
from mlmia.utilities.data_utilities import image_resize, flow_resize, segmentation_resize, find_best_padding_cval


class DataAugmentor(object):
    """
    Class for augmenting data.

    Parameters
    ----------
    augmentation_chance: float
        Chance of augmentation
    """

    def __init__(self, task_type: TaskType = None, **kwargs):
        self.method_container = {}
        self.task_type = task_type
        self.inputs_pattern = kwargs.get("inputs_pattern", None)
        self.targets_pattern = kwargs.get("targets_pattern", None)
        self.flow_iterator = None

    def __call__(self, data, batch_input=False):
        return self.transform(data, batch_input)

    def transform(self, data, batch_input=False):
        if self.method_container:
            data = [d.copy() for d in data]
            methods = copy.deepcopy(self.method_container)

            for (method_name, method) in methods.items():
                if batch_input:
                    data = self._transform_batch(method, data)
                else:
                    data = self._transform_single(method, data)
        return data

    def _transform_single(self, method, data):
        if method.augmentation_chance >= np.random.random():
            method.randomize()
            transformed_data = []
            for d_idx, d in enumerate(data):
                if method.apply_to == 'all' or d_idx in method.apply_to:
                    td = self._apply_method(method, d, d_idx)
                else:
                    td = d
                transformed_data.append(td)
        else:
            transformed_data = data

        return transformed_data

    def _transform_batch(self, method, batch):
        transformed_batch = []
        for batch_entry in zip(*batch):
            transformed_data = self._transform_single(method, batch_entry)
            transformed_batch.append(transformed_data)
        transformed_batch = list(map(np.array, list(zip(*transformed_batch))))
        return transformed_batch

    def _apply_method(self, method, data, data_index=None):
        if self._is_sequence_input(data, data_index) or len(data.shape)==4:
            if method.sequence_input:
                new_data = method(data, data_index=data_index)
            else:
                method = partial(method, data_index=data_index)
                new_data = [method(d, sequence_index=idx) for idx, d in enumerate(data)]
        else:
            new_data = method(data, data_index=data_index)
        return DataArray(new_data, layout=data.layout if hasattr(data, 'layout') else None)

    def _is_sequence_input(self, data, data_index=None):
        cond_a = hasattr(data, 'layout') and data.layout is not None and "sequence" in data.layout
        cond_b = False

        if data_index is not None:
            if self.inputs_pattern:
                cond_b = any([cond_b, ("sequence" in self.inputs_pattern and data_index == 0)])
            if self.targets_pattern:
                cond_b = any([cond_b, ("sequence" in self.targets_pattern and data_index == 1)])
        return any([cond_a, cond_b])

    def set_task_type(self, task_type: TaskType = None):
        self.task_type = task_type

    def set_inputs(self, pattern):
        self.inputs_pattern = pattern

    def set_targets(self, pattern):
        self.targets_pattern = pattern

    def set_loader_config(self, config):
        self.inputs_pattern = config.get("inputs_pattern", self.inputs_pattern)
        self.targets_pattern = config.get("targets_pattern", self.targets_pattern)
        self.task_type = config.get("task_type", self.task_type)

    def add_method(self, method_func: Callable[..., tuple]):
        self.method_container[method_func.__name__] = method_func

    def clear_methods(self):
        self.method_container = {}

    def add_contrast_scaling(self, min_scale=0.25, max_scale=1.75, **kwargs):
        method = ContrastScale(min_scale=min_scale, max_scale=max_scale, **kwargs)
        self.add_method(method)

    def add_limit(self, apply_to=(1,), **kwargs):
        method = Limit(apply_to=apply_to, **kwargs)
        self.add_method(method)

    def add_blurring(self, sigma_min=0., sigma_max=1., **kwargs):
        method = Blur(sigma_min=sigma_min, sigma_max=sigma_max, **kwargs)
        self.add_method(method)

    def add_point_disturbance(self, max_disturbance=0.01, graph_indices=None, **kwargs):
        method = PointDisturbance(max_disturbance, graph_indices=graph_indices, **kwargs)
        self.add_method(method)

    def add_point_removal(self, max_removal=0.25, **kwargs):
        method = PointRemoval(max_removal=max_removal, **kwargs)
        self.add_method(method)
        
    def add_point_shuffle(self, size=64, **kwargs):
        method = PointShuffle(size=size, **kwargs)
        self.add_method(method)

    def add_horisontal_flip(self, apply_to='all', flow_indices=None, graph_indices=None, **kwargs):
        method = FlipHorizontal(apply_to=apply_to, flow_indices=flow_indices,
            graph_indices=graph_indices, **kwargs)
        self.add_method(method)

    def add_vertical_flip(self, apply_to='all', flow_indices=None, graph_indices=None, **kwargs):
        method = FlipVertical(apply_to=apply_to, flow_indices=flow_indices,
            graph_indices=graph_indices, **kwargs)
        self.add_method(method)

    def add_elastic_deformation(self, apply_to='all', **kwargs):
        method = ElasticDeformation(apply_to=apply_to, **kwargs)
        self.add_method(method)

    def add_nonlinear_colormap(self, **kwargs):
        method = NonLinearMap(**kwargs)
        self.add_method(method)

    def add_gaussian_shadow(self, sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8), **kwargs):
        method = GaussianShadow(sigma_x=sigma_x, sigma_y=sigma_y, strength=strength, **kwargs)
        self.add_method(method)

    def add_gamma_transformation(self, low=0.25, high=1.7, **kwargs):
        method = GammaTransform(low=low, high=high, **kwargs)
        self.add_method(method)

    def add_brightness_transform(self, max_scale=0.2, **kwargs):
        method = BrightnessTransform(max_scale=max_scale, **kwargs)
        self.add_method(method)

    def add_resolution_deterioration(self, min_scale=0.1, max_scale=4, **kwargs):
        method = ResolutionDeterioration(min_scale=min_scale, max_scale=max_scale, **kwargs)
        self.add_method(method)

    def add_depth_attenuation(self, min_attenuation=0.1, max_attenuation=0.7, min_saturation_point=0.5,
                              max_saturation_point=1.0, input_orientation="HW", **kwargs):
        method = DepthAttenuation(min_attenuation=min_attenuation, max_attenuation=max_attenuation,
                                  min_saturation_point=min_saturation_point, max_saturation_point=max_saturation_point,
                                  input_orientation=input_orientation, **kwargs)
        self.add_method(method)

    def add_haze_application(self, center=(0.2, 0.6), extent=(0.05, 0.15), strength=(0.1, 0.4), input_orientation="HW",
                             **kwargs):
        method = HazeApplication(min_center=center[0], max_center=center[1], min_extent=extent[0], max_extent=extent[1],
                                 min_strength=strength[0], max_strength=strength[1],
                                 input_orientation=input_orientation, **kwargs)
        self.add_method(method)

    def add_jpeg_compression(self, compression_range=(0, 95), **kwargs):
        method = JPEGCompression(min_compression=compression_range[0], max_compression=compression_range[1], **kwargs)
        self.add_method(method)

    def add_random_noise(self, method=None, **kwargs):
        method = RandomNoise(method=method, **kwargs)
        self.add_method(method)

    def add_rotation(self, max_angle=10, apply_to='all', flow_indices=None, 
                     segmentation_indices=None, graph_indices=None, **kwargs):
        method = Rotation(max_angle=max_angle, apply_to=apply_to, flow_indices=flow_indices,
                          segmentation_indices=segmentation_indices, graph_indices=graph_indices, **kwargs)
        self.add_method(method)

    def add_translation(self, apply_to='all', max_width=0.2, max_height=0.2, segmentation_indices=None,
                        graph_indices=None, **kwargs):
        method = Translation(max_width=max_width, max_height=max_height, apply_to=apply_to,
                             segmentation_indices=segmentation_indices, graph_indices=graph_indices, **kwargs)
        self.add_method(method)

    def add_sequence_reverse(self, apply_to='all', flow_indices=None, **kwargs):
        method = SequenceReverse(flow_indices=flow_indices, apply_to=apply_to, **kwargs)
        self.add_method(method)

    def add_sequence_skip(self, apply_to='all', min_skip=0.25, max_skip=0.75, **kwargs):
        method = SequenceSkip(min_skip, max_skip, apply_to=apply_to, **kwargs)
        self.add_method(method)

    def add_scale(self, min_scale_rate=0.9, max_scale_rate=2.0, apply_to='all', flow_indices=None,
                  segmentation_indices=None, graph_indices=None, **kwargs):
        method = Scale(min_scale_rate=min_scale_rate, max_scale_rate=max_scale_rate, flow_indices=flow_indices,
                       segmentation_indices=segmentation_indices, graph_indices=graph_indices, apply_to=apply_to, **kwargs)
        self.add_method(method)

    def add_speckle_reduction(self, diameter_range=(1, 3), color_range=(45, 55), coord_range=(45, 55), **kwargs):
        method = SpeckleNoiseReduction(min_diameter=diameter_range[0], max_diameter=diameter_range[1],
                                       min_color=color_range[0], max_color=color_range[1],
                                       min_coord=coord_range[0], max_coord=coord_range[1],
                                       **kwargs)
        self.add_method(method)

    def add_nonlocal_means(self, **kwargs):
        method = NonLocalMeans(**kwargs)
        self.add_method(method)

    def add_random_crop(self, width, height, **kwargs):
        method = RandomCrop(width=width, height=height, **self.__dict__, **kwargs)
        self.add_method(method)
        
    def add_blackout(self, max_discard, apply_to='all', **kwargs):
        method = Blackout(max_discard=max_discard, apply_to=apply_to, **kwargs)
        self.add_method(method)

    def add_random_sector_crop(self, min_sector_width=60, max_sector_width=60,
                               min_sector_tilt=0, max_sector_tilt=0, input_orientation="HW", **kwargs):
        method = RandomSectorCrop(min_sector_width=min_sector_width, max_sector_width=max_sector_width,
                                  min_sector_tilt=min_sector_tilt, max_sector_tilt=max_sector_tilt,
                                  input_orientation=input_orientation, **kwargs)
        self.add_method(method)

    def add_radial_dropouts(self, sector_width=90,
                            min_depth_splits: int = 3, max_depth_splits: int = 3,
                            min_width_splits: int = 3, max_width_splits: int = 3,
                            min_dropout_ratio: float = 0.2, max_dropout_ratio: float = 0.2,
                            input_orientation="HW", **kwargs):
        method = RadialDropouts(sector_width=sector_width,
                                min_depth_splits=min_depth_splits, max_depth_splits=max_depth_splits,
                                min_width_splits=min_width_splits, max_width_splits=max_width_splits,
                                min_dropout_ratio=min_dropout_ratio, max_dropout_ratio=max_dropout_ratio,
                                input_orientation=input_orientation, **kwargs)
        self.add_method(method)

    def get_config(self):
        config = {'inputs_pattern': self.inputs_pattern,
                  'targets_pattern': self.targets_pattern,
                  'task_type': self.task_type,
                  'methods': dict((value.__class__.__name__, value.get_config()) for (key, value) in
                                  self.method_container.items())}
        return config


class AugmentationMethod(ABC):
    def __init__(self, **kwargs):
        self.apply_to = kwargs.get("apply_to", (0,))
        self.augmentation_chance = kwargs.get("augmentation_chance", 0.5)
        self.flow_indices = kwargs.get("flow_indices", None)
        self.sequence_input = kwargs.get("sequence_input", False)
        self.task_type = kwargs.get("task_type", None)
        self.inputs_pattern = kwargs.get("inputs_pattern", None)
        self.targets_pattern = kwargs.get("targets_pattern", None)

    @abstractmethod
    def transform(self, data, *args, **kwargs):
        ...

    @abstractmethod
    def randomize(self):
        ...

    def __call__(self, data, *args, **kwargs):
        return self.transform(data, *args, **kwargs)

    def __name__(self):
        return self.__class__.__name__

    def ensure_dtype(func):
        @wraps(func)
        def wrapped(cls, data, *args, **kwargs):
            init_dtype = data.dtype
            data = img_as_float(data).astype(np.float32)
            augmented_data = func(cls, data, *args, **kwargs)
            if isinstance(augmented_data, (tuple, list)):
                if init_dtype == np.uint8:
                    return type(augmented_data)(img_as_ubyte(d) for d in augmented_data)
                return type(augmented_data)(d.astype(init_dtype) for d in augmented_data)
            else:
                if init_dtype == np.uint8:
                    return img_as_ubyte(augmented_data)
                return augmented_data.astype(init_dtype)
        return wrapped

    def get_config(self):
        """Returns a dictionary of all non-protected attributes in class."""
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    @staticmethod
    def is_grayscale(data):
        if len(data.shape) == 2:
            return all([data.shape[0] > 0, data.shape[1] > 0])
        elif len(data.shape) == 3:
            return data.shape[-1] == 1
        else:
            return False

    @staticmethod
    def is_rgb(data):
        if len(data.shape) == 3:
            return data.shape[-1] == 3
        else:
            return False


class Blur(AugmentationMethod):
    def __init__(self, sigma_min=0.0, sigma_max=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self._random_sigma = None

    def randomize(self):
        self._random_sigma = np.random.uniform(self.sigma_min, self.sigma_max)

    def transform(self, data, *args, **kwargs):
        return gaussian_filter(data, self._random_sigma) * np.array(data > 0)


class GammaTransform(AugmentationMethod):
    def __init__(self, low=0.25, high=1.7, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high
        self._random_gamma = None

    def randomize(self):
        self._random_gamma = np.random.uniform(self.low, self.high)

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        return np.clip(np.power(data, self._random_gamma), 0, 1)


class RandomNoise(AugmentationMethod):
    def __init__(self, method=None, **kwargs):
        super().__init__(**kwargs)
        self.available_methods = ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"]

        assert method is None or method in self.available_methods, \
            "method argument must be None or one of {}".format(self.available_methods)

        self._random_method = method
        self._random_seed = None

        self.noise_mean = kwargs.get("noise_mean", None)
        self.noise_variance = kwargs.get("noise_variance", None)

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        return random_noise(data, mode=self._random_method, seed=self._random_seed)

    def randomize(self):
        if self._random_method is None:
            self._random_method = np.random.choice(self.available_methods)
        self._random_seed = np.random.randint(2**32, dtype=np.uint)


class BrightnessTransform(AugmentationMethod):
    def __init__(self, max_scale=0.2, **kwargs):
        super().__init__(**kwargs)
        self.max_scale = max_scale
        self._random_scale = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        return np.clip(data+self._random_scale, 0.0, 1.0)

    def randomize(self):
        self._random_scale = np.random.uniform(-self.max_scale, self.max_scale)


class ContrastScale(AugmentationMethod):
    def __init__(self, min_scale=0.25, max_scale=1.75, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._random_scale = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        return np.clip(data*self._random_scale, 0, 1)

    def randomize(self):
        self._random_scale = np.random.uniform(self.min_scale, self.max_scale)

class PointDisturbance(AugmentationMethod): # randomly disturb all keypoints by <1% in x and y
    def __init__(self, max_disturbance=0.01, graph_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.max_disturbance = max_disturbance
        self.graph_indices = graph_indices

    @AugmentationMethod.ensure_dtype    
    def transform(self, data, data_index=None, *args, **kwargs):
        if self.graph_indices is not None and data_index in self.graph_indices:
            data = np.copy(data)

            random_disturbance = (np.random.rand(*data.shape)*2.0) - 1.0
            data = data + random_disturbance*self.max_disturbance
            data = np.clip(data, 0.0, 1.0)
        return data

    def randomize(self):
        pass

class PointRemoval(AugmentationMethod): # randomly relocate some of the points
    def __init__(self, max_removal=0.25, **kwargs):
        super().__init__(**kwargs)
        self.max_removal = max_removal
        self._random_removal = None

    @AugmentationMethod.ensure_dtype    
    def transform(self, data, data_index=None, *args, **kwargs):
        to_remove = np.random.rand(data.shape[0]) < self._random_removal
        for i in [0,31,32,63]:
            to_remove[i] = False
        remove_positions = np.argwhere(to_remove)
        data[remove_positions] = data[remove_positions-1]
        return data

    def randomize(self):
        self._random_removal = np.random.uniform(0.0, self.max_removal)

class PointShuffle(AugmentationMethod): # reshuffle points
    def __init__(self, size=64, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.new_order = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        data = data[self.new_order]
        return data

    def randomize(self):
        self.new_order = np.random.permutation(self.size)

class Limit(AugmentationMethod):
    """
        Clip values.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        return np.clip(data, 0.0, 1.0)

    def randomize(self):
        pass

class FlipVertical(AugmentationMethod):
    """ Flips the data vertically. For flow data, it also multiplies the vy-axis with (-1).
            a b c     g h i
            d e f ==> d e f
            g h i     a b c
    """
    def __init__(self, flow_indices=None, graph_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.flow_indices = flow_indices
        self.graph_indices = graph_indices

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        if len(data.shape) > 3:
          return np.array([np.flipud(data[i]) for i in range(len(data))])
        if self.flow_indices is not None and data_index in self.flow_indices:
            return np.flipud(data)*[1, -1]
        if self.graph_indices is not None and data_index in self.graph_indices:
            data[0] = 1.0 - data[0]
            return data
        return np.flipud(data)

    def randomize(self):
        pass


class FlipHorizontal(AugmentationMethod):
    """ Flips the data horizontally. For flow data, it also multiplies the vx-axis with (-1).
            a b c     c b a
            d e f ==> f e d
            g h i     i h g
    """
    def __init__(self, flow_indices=None, graph_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.flow_indices = flow_indices
        self.graph_indices = graph_indices

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        if len(data.shape) > 3:
          return np.array([np.fliplr(data[i]) for i in range(len(data))])
        if self.flow_indices is not None and data_index in self.flow_indices:
            return np.fliplr(data)*[-1, 1]
        if self.graph_indices is not None and data_index in self.graph_indices:
            data[1] = 1.0 - data[1]
            return data
        return np.fliplr(data)

    def randomize(self):
        pass


class SequenceSkip(AugmentationMethod):
    def __init__(self, min_skip=0.25, max_skip=0.75, **kwargs):
        super().__init__(**kwargs)
        self.sequence_input = True
        self.min_skip = 0.25
        self.max_skip = 0.75
        self.random_skip = None

    def transform(self, data, *args, **kwargs):
        for i in range(len(data)):
            if np.random.uniform() < self.random_skip:
                data[i] = data[i]*0.0
        return data

    def randomize(self):
        self.random_skip = np.random.uniform(self.min_skip, self.max_skip)


class SequenceReverse(AugmentationMethod):
    """ Reverse sequence. Mainly made for optical flow.

    Ref: Computing Inverse Optical Flow by J. Sanchez et al.
    """
    def __init__(self, flow_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.sequence_input = True
        self.flow_indices = flow_indices
        self._input_data = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        self._input_data = data
        data = data[::-1]


        if self.flow_indices is not None and data_index in self.flow_indices:
            inverse_flow = self.estimate_inverse_flow(data)
            return inverse_flow
        return data

    def randomize(self):
        pass

    @staticmethod
    def estimate_inverse_flow(flow):
        x_grid, y_grid = np.meshgrid(np.arange(flow.shape[0]), np.arange(flow.shape[1]), indexing='ij')
        grid = np.stack((x_grid, y_grid), axis=-1)
        grid = grid + flow
        flow_non_zero = np.nonzero(flow)[:2]
        x_min, x_max = np.min(flow_non_zero[0]), np.max(flow_non_zero[0])
        y_min, y_max = np.min(flow_non_zero[1]), np.max(flow_non_zero[1])
        inverse_flow = griddata(grid[x_min:x_max, y_min:y_max].reshape(-1, 2),
                                -flow[x_min:x_max, y_min:y_max].reshape(-1, 2), (x_grid, y_grid), fill_value=0)
        return inverse_flow


class JPEGCompression(AugmentationMethod):
    def __init__(self, min_compression=0, max_compression=95, **kwargs):
        super().__init__(**kwargs)
        self.min_compression = min_compression
        self.max_compression = max_compression
        self._random_compression = None

    def transform(self, data, *args, **kwargs):
        if self.is_grayscale(data):
            image = PIL.Image.fromarray((data[..., 0]*255).astype(np.uint8), mode='L')
        elif self.is_rgb(data):
            image = PIL.Image.fromarray((data*255).astype(np.uint8), mode='RGB')
        else:
            raise ValueError('Unsupported nr of channels in JPEGCompression transform')

        with BytesIO() as f:
            image.save(f, format='JPEG', quality=100-self._random_compression)
            f.seek(0)
            image_jpeg = np.asarray(PIL.Image.open(f)).astype(np.float32)/255.0

        return image_jpeg.reshape(data.shape)

    def randomize(self):
        self._random_compression = np.random.randint(self.min_compression, self.max_compression)


class HazeApplication(AugmentationMethod):
    def __init__(self, min_center=0.20, max_center=0.60, min_extent=0.05, max_extent=0.15, min_strength=0.1,
                 max_strength=0.5, input_orientation="HW", random_seed=None, **kwargs):
        super().__init__(**kwargs)

        self.input_orientation = input_orientation.lower()
        self.min_center, self.max_center = min_center, max_center
        self.min_extent, self.max_extent = min_extent, max_extent
        self.min_strength, self.max_strength = min_strength, max_strength
        self.random_seed = random_seed

        self._random_center = None
        self._random_extent = None
        self._random_strength = None
        self._random_seed = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, return_map=False, *args, **kwargs):
        w, h, ch = data.shape

        if self.input_orientation == "hw":
            h, w = w, h

        r_axis = np.linspace(0, h, 256, endpoint=True)
        theta_axis = np.linspace(-np.pi / 4, np.pi / 4, 256, endpoint=True)

        gauss_func = self._random_strength * gaussian(np.arange(len(r_axis)), self._random_center*256,
                                                     self._random_extent*256)
        gradient_image = np.tile(gauss_func, (len(theta_axis), 1))
        grid_interpolator = RegularGridInterpolator((theta_axis, r_axis), gradient_image, bounds_error=False,
                                                    fill_value=0)

        x = np.linspace(np.sin(theta_axis[0]), np.sin(theta_axis[-1]), w) * (r_axis[-1])
        z = np.linspace(r_axis[0] * min(np.cos(theta_axis[0]), np.cos(theta_axis[-1])), r_axis[-1], h)

        if self.input_orientation == "hw":
            x_grid, z_grid = np.meshgrid(x, z)
        else:
            x_grid, z_grid = np.meshgrid(x, z, indexing='ij')

        coords = np.dstack((np.arctan2(x_grid, z_grid), np.hypot(x_grid, z_grid)))
        haze_map = random_noise(grid_interpolator(coords)[..., None], mode='speckle', seed=self._random_seed)
        data = np.clip(data + haze_map, 0, 1)
        if return_map:
            return data, haze_map
        return data

    def randomize(self):
        self._random_center = np.random.uniform(self.min_center, self.max_center)
        self._random_extent = np.random.uniform(self.min_extent, self.max_extent)
        self._random_strength = np.random.uniform(self.min_strength, self.max_strength)
        self._random_seed = np.random.randint(2**32, dtype=np.uint) if self.random_seed is None else self.random_seed


class ElasticDeformation(AugmentationMethod):
    def __init__(self, min_sigma=0.1, max_sigma=0.2, mode="opencv", return_distortion_map=False, **kwargs):
        super().__init__(**kwargs)

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.mode = mode
        self.return_distortion_map = return_distortion_map

        self._random_seed = None
        self._random_sigma = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        np.random.seed(self._random_seed)

        init_data_shape = data.shape
        w, h, _ = data.shape

        sigma = np.random.uniform(w * 0.10, h * 0.20)
        alpha = (w, h)

        if self.mode == 'opencv':
            x_distorted, y_distorted = self.estimate_distortion(w, h, sigma, alpha)
            distorted_image = np.transpose([cv2.remap(d, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR)
                                            for d in data.transpose(2, 1, 0)], (1, 2, 0))
        else:
            dx = gaussian_filter((np.random.rand(w, h) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(w, h) * 2 - 1), sigma) * alpha

            x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
            x_distorted = x + dx
            y_distorted = y + dy
            indices = np.reshape(x_distorted, (-1, 1)), np.reshape(y_distorted, (-1, 1))
            distorted_image = map_coordinates(data, indices, order=1, mode='reflect').reshape((w, h))

        distorted_image = distorted_image.reshape(init_data_shape)
        return distorted_image

    def randomize(self):
        self._random_seed = np.random.randint(2**32, dtype=np.uint)

    def estimate_distortion(self, width, height, sigma, alpha):
        np.random.seed(self._random_seed)

        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur(np.random.rand(width, height) * 2 - 1, ksize=(blur_size, blur_size), sigmaX=sigma) * alpha[0]
        dy = cv2.GaussianBlur(np.random.rand(width, height) * 2 - 1, ksize=(blur_size, blur_size), sigmaX=sigma) * alpha[1]

        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        x_distorted = (x + dx).astype(np.float32)
        y_distorted = (y + dy).astype(np.float32)
        return x_distorted, y_distorted


class SpeckleNoiseReduction(AugmentationMethod):
    def __init__(self, min_diameter=1, max_diameter=3, min_color=45, max_color=55, min_coord=45, max_coord=55,
                 **kwargs):
        super().__init__(**kwargs)

        self.min_diameter, self.max_diameter = min_diameter, max_diameter
        self.min_color, self.max_color = min_color, max_color
        self.min_coord, self.max_coord = min_coord, max_coord

        self._random_diameter = None
        self._random_color = None
        self._random_coord = None

    def transform(self, data, *args, **kwargs):
        data = data.squeeze()  # TODO: Fix to work with 3 channels
        return cv2.bilateralFilter(data, self._random_diameter, self._random_color, self._random_coord)[..., None]

    def randomize(self):
        self._random_diameter = np.random.randint(self.min_diameter, self.max_diameter+1)
        self._random_color = np.random.randint(self.min_color, self.max_color+1)
        self._random_coord = np.random.randint(self.min_coord, self.max_coord+1)


class NonLocalMeans(AugmentationMethod):
    def __init__(self, min_patch_size=1, max_patch_size=5, min_patch_distance=3, max_patch_distance=9,
                 min_sigma_scale=1, max_sigma_scale=30, **kwargs):
        super().__init__(**kwargs)

        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.min_patch_distance = min_patch_distance
        self.max_patch_distance = max_patch_distance
        self.min_sigma_scale = min_sigma_scale
        self.max_sigma_scale = max_sigma_scale

        self._random_patch_size = None
        self._random_patch_distance = None
        self._random_sigma_scale = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        sigma = self._random_sigma_scale*np.mean(estimate_sigma(data, multichannel=True))
        data = denoise_nl_means(data, h=0.8*sigma, sigma=sigma, fast_mode=True, multichannel=True)
        return data[..., None]

    def randomize(self):
        self._random_patch_size = np.random.randint(self.min_patch_size, self.max_patch_size)
        self._random_patch_distance = np.random.randint(self.min_patch_distance, self.max_patch_distance)
        self._random_sigma_scale = np.random.randint(self.min_sigma_scale, self.max_sigma_scale)


class GaussianShadow(AugmentationMethod):
    """
    Parameters
    ----------
    sigma_x: tuple,
        Sigma value in x-direction with minmax as tuple, (min, max)
    sigma_y: tuple
        Sigma value in y-direction with minmax as tuple, (min, max)
    strength: tuple
        Signal strength with minmax as tuple, (min, max)
    location: tuple, optional
        Force location of shadow at specific location (x, y). Location (x, y) is given as a fraction of the image size,
        i.e. between 0 and 1.
    """

    def __init__(self, sigma_x: tuple, sigma_y: tuple, strength: tuple, location: tuple = None, **kwargs):
        super().__init__(**kwargs)

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.strength = strength
        self.location = location

        self._random_sigma_x = None
        self._random_sigma_y = None
        self._random_strength = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, return_map=False, *args, **kwargs):
        x, y = np.meshgrid(np.linspace(-1, 1, data.shape[0], dtype=np.float32),
                           np.linspace(-1, 1, data.shape[1], dtype=np.float32), copy=False, indexing='ij')

        if self.location:
            x_mu, y_mu = self.location[0]*2-1, self.location[1]*2-1
        else:
            x_mu = np.random.uniform(-1.0, 1.0, 1).astype(np.float32)
            y_mu = np.random.uniform(-1.0, 1.0, 1).astype(np.float32)
        g = 1.0 - self._random_strength * np.exp(-((x - x_mu) ** 2 / (2.0 * self._random_sigma_x ** 2)
                                                   + (y - y_mu) ** 2 / (2.0 * self._random_sigma_y ** 2)))

        augmented_data = np.copy(data)
        if len(augmented_data.shape) > 2:
            augmented_data = augmented_data*g[..., None]
        else:
            augmented_data = augmented_data * g

        if return_map:
            return augmented_data, g
        return augmented_data

    def randomize(self):
        self._random_sigma_x = np.random.uniform(self.sigma_x[0], self.sigma_x[1], 1).astype(np.float32)
        self._random_sigma_y = np.random.uniform(self.sigma_y[0], self.sigma_y[1], 1).astype(np.float32)
        self._random_strength = np.random.uniform(self.strength[0], self.strength[1], 1).astype(np.float32)


class Translation(AugmentationMethod):
    """
    Translates image arbitrary in bounds [-max_width%-max_width%] and [-max_height%, max_height]

    Parameters
    ----------
    max_width: float

    """

    def __init__(self, max_width=0.2, max_height=0.2, mode="constant", flow_indices=None, segmentation_indices=None,
                 graph_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.max_width = max_width
        self.max_height = max_height
        self.flow_indices = flow_indices
        self.segmentation_indices = segmentation_indices
        self.graph_indices = graph_indices
        self.mode = mode

        self._random_width = None
        self._random_height = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        data = np.copy(data)
        '''
        if len(data.shape) > 3:
            t, h, w, ch = data.shape
            for channel in range(0, data.shape[-1]):
                if self.segmentation_indices is not None and data_index in self.segmentation_indices:
                    cval = self.find_best_padding_cval(data[..., channel])
                    data[..., channel] = shift(data[..., channel], [0, h*self._random_height, w*self._random_width],
                                               order=0, cval=cval, mode=self.mode)
                else:
                    data[..., channel] = shift(data[..., channel], [0, h*self._random_height, w*self._random_width],
                                               order=1, mode=self.mode)
        '''
        if self.graph_indices is not None and data_index in self.graph_indices:
            data_reshaped = data
            
            data_reshaped[...,0] = data_reshaped[...,0] + self._random_height
            data_reshaped[...,1] = data_reshaped[...,1] + self._random_width
            #data[data > 1.0] = 1.0
            #data[data < 0.0] = 0.0
            data = data_reshaped

        elif len(data.shape) > 2:
            h, w, ch = data.shape
            for channel in range(0, data.shape[2]):
                if self.segmentation_indices is not None and data_index in self.segmentation_indices:
                    cval = find_best_padding_cval(data[..., channel])
                    data[..., channel] = shift(data[..., channel], [h*self._random_height, w*self._random_width],
                                               order=0, cval=cval, mode=self.mode)
                else:
                    data[..., channel] = shift(data[..., channel], [h*self._random_height, w*self._random_width],
                                               order=1, mode=self.mode)
        else:
            h, w = data.shape
            data = shift(data, [h*self._random_height, w*self._random_width], order=1)
        return data

    def randomize(self):
        self._random_width = np.random.uniform(-self.max_width, self.max_width)
        self._random_height = np.random.uniform(-self.max_height, self.max_height)


class Rotation(AugmentationMethod):
    """
    Rotates image with arbitrary angle in bounds [-max_angle:max_angle]

    Parameters
    ----------
    max_angle: number
        Maximum angle in degrees.
    """

    def __init__(self, max_angle=15, flow_indices=None, segmentation_indices=None, graph_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.max_angle = max_angle
        self._random_angle = None
        self.flow_indices = flow_indices
        self.segmentation_indices = segmentation_indices
        self.graph_indices = graph_indices

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        data = np.copy(data)
        
        '''
        if len(data.shape) > 3:
            for channel in range(0, data.shape[-1]):
                if self.segmentation_indices is not None and data_index in self.segmentation_indices:
                    cval = self.find_best_padding_cval(data[..., channel])
                    data[..., channel] = rotate(data[..., channel], self._random_angle, order=0, reshape=False,
                                                axes=(2,1), cval=cval)
                else:
                    data[..., channel] = rotate(data[..., channel], self._random_angle, order=1, axes=(2,1), reshape=False)
        '''
        if self.graph_indices is not None and data_index in self.graph_indices:
            #if len(data.shape) > 2:
            #    data_reshaped = np.concatenate([data[...,0], data[...,1]], axis=0)
            #else:
            data_reshaped = data

            theta = np.radians(-self._random_angle)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array(((c, -s), (s, c)))

            data_reshaped = data_reshaped*2.0-1.0
            data_reshaped = data_reshaped.dot(rot_mat)
            data_reshaped = (data_reshaped+1.0)/2.0
            #if len(data.shape) > 2:
            #    data = np.stack([data_reshaped[:data.shape[0]], data_reshaped[data.shape[0]:]], axis=-1)
            #else:
            data = data_reshaped

        elif len(data.shape) > 2:
            for channel in range(0, data.shape[2]):
                if self.segmentation_indices is not None and data_index in self.segmentation_indices:
                    cval = find_best_padding_cval(data[..., channel])
                    data[..., channel] = rotate(data[..., channel], self._random_angle, order=0, reshape=False,
                                                cval=cval)
                else:
                    data[..., channel] = rotate(data[..., channel], self._random_angle, order=1, reshape=False)
        else:
            data = rotate(data, self._random_angle, order=1, reshape=False)
        
        if self.flow_indices is not None and data_index in self.flow_indices:
            theta = np.radians(-self._random_angle)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array(((c, -s), (s, c)))
            data = data.dot(rot_mat)
        return data

    def randomize(self):
        self._random_angle = np.random.randint(-self.max_angle, self.max_angle)


class Scale(AugmentationMethod):
    def __init__(self,
        min_scale_rate=0.9, max_scale_rate=2.0, flow_indices=None, segmentation_indices=None, graph_indices=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_scale_rate = min_scale_rate
        self.max_scale_rate = max_scale_rate
        self.flow_indices = flow_indices
        self.segmentation_indices = segmentation_indices
        self.graph_indices = graph_indices
        self._random_scale_rate = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        if self.graph_indices is not None and data_index in self.graph_indices:
            result = data*2.0-1.0
            result = result*self._random_scale_rate
            result = (result+1.0)/2.0
            #result[result>1.0] = 1.0
            #result[result<0.0] = 0.0
        
        else:
            height, width, _ = data.shape
            new_height, new_width = int(height * self._random_scale_rate), int(width * self._random_scale_rate)

            x_min, y_min = max(0, new_width - width) // 2, max(0, new_height - height) // 2
            x_max, y_max = x_min + width, y_min + height
            bbox = np.array([x_min, y_min, x_max, y_max])

            bbox = (bbox / self._random_scale_rate).astype(int)
            x_min, y_min, x_max, y_max = bbox
            cropped_img = data[y_min:y_max, x_min:x_max]

            resize_width, resize_height = min(new_width, width), min(new_height, height)
            pad_width_min, pad_height_min = (width - resize_width) // 2, (height - resize_height) // 2
            pad_width_max, pad_height_max = (width - resize_width)-pad_width_min, (height - resize_height)-pad_height_min
            pad_spec = [(pad_height_min, pad_height_max), (pad_width_min, pad_width_max)] + [(0, 0)] * (data.ndim - 2)

            if self.flow_indices is not None and data_index in self.flow_indices:
                result = flow_resize(cropped_img, (resize_height, resize_width))
                result = np.pad(result, pad_spec, mode='constant')
            elif self.segmentation_indices is not None and data_index in self.segmentation_indices:
                result = []
                pad_spec = [(pad_height_min, pad_height_max), (pad_width_min, pad_width_max)]
                for channel in range(0, cropped_img.shape[2]):
                    result_channel = segmentation_resize(cropped_img[..., channel], (resize_height, resize_width))
                    fill_value = find_best_padding_cval(result_channel)
                    result_channel = np.pad(result_channel, pad_spec, mode='constant', constant_values=fill_value)
                    result.append(result_channel)
                result = np.array(result).transpose((1, 2, 0))   
            else:
                result = image_resize(cropped_img, (resize_height, resize_width))
                result = np.pad(result, pad_spec, mode='constant')
        return result

    def randomize(self):
        self._random_scale_rate = np.random.uniform(self.min_scale_rate, self.max_scale_rate)


class NonLinearMap(AugmentationMethod):
    def __init__(self, alpha=0.544559, beta=1.686562, gamma=5.598193, delta=0.638681, y0=0.002457387314, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.y0 = y0

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        data = self.alpha * np.exp(-np.exp(self.beta - self.gamma * data) + self.delta * data) - self.y0
        return np.clip(data, a_min=0, a_max=1)

    def randomize(self):
        pass


class ResolutionDeterioration(AugmentationMethod):
    def __init__(self, min_scale=0.1, max_scale=4, **kwargs):
        super().__init__(**kwargs)

        self.available_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC,
                                  cv2.INTER_LANCZOS4]
        self.min_scale = min_scale
        self.max_scale = max_scale

        self._random_scale = None
        self._random_method_away = None
        self._random_method_back = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, *args, **kwargs):
        new_shape = (int(data.shape[0] * self._random_scale[0]), int(data.shape[1] * self._random_scale[1]))
        resized_data = cv2.resize(data, new_shape, interpolation=self._random_method_away)
        backsized_image = cv2.resize(resized_data, data.shape[:2][::-1], interpolation=self._random_method_back)
        return np.clip(backsized_image[..., None], a_min=0, a_max=1)

    def randomize(self):
        self._random_scale = np.random.uniform(self.min_scale, self.max_scale, 2)
        self._random_method_away = np.random.choice(self.available_methods)
        self._random_method_back = np.random.choice(self.available_methods)


class DepthAttenuation(AugmentationMethod):
    def __init__(self, min_attenuation=0.1, max_attenuation=0.7, min_saturation_point=0.9,
                 max_saturation_point=1.0, input_orientation="HW", **kwargs):
        super().__init__(**kwargs)
        self.input_orientation = input_orientation.lower()
        self.min_attenuation = min_attenuation
        self.max_attenuation = max_attenuation
        self.min_saturation_pt = min_saturation_point
        self.max_saturation_pt = max_saturation_point

        self._random_attenuation = None
        self._random_saturation_point = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, return_map=False, *args, **kwargs):
        w, h, ch = data.shape

        if self.input_orientation == "hw":
            h, w = w, h

        r_axis = np.linspace(0, h, 256, endpoint=True)
        theta_axis = np.linspace(-np.pi / 4, np.pi / 4, 256, endpoint=True)

        xp = np.array([0., self._random_saturation_point, 1.0])
        yp = np.array([1.0, 1-self._random_attenuation,  1-self._random_attenuation])

        spl = UnivariateSpline(xp, yp, k=2)
        xs = np.linspace(xp.min(), xp.max(), len(r_axis))
        ys = np.clip(spl(xs), 0, 1)

        gradient_image = np.tile(ys, (len(theta_axis), 1))
        grid_interpolator = RegularGridInterpolator((theta_axis, r_axis), gradient_image, bounds_error=False,
                                                    fill_value=0)
        x = np.linspace(np.sin(theta_axis[0]), np.sin(theta_axis[-1]), w) * (r_axis[-1])
        z = np.linspace(r_axis[0] * min(np.cos(theta_axis[0]), np.cos(theta_axis[-1])), r_axis[-1], h)
        if self.input_orientation == "hw":
            x_grid, z_grid = np.meshgrid(x, z)
        else:
            x_grid, z_grid = np.meshgrid(x, z, indexing='ij')

        coords = np.dstack((np.arctan2(x_grid, z_grid), np.hypot(x_grid, z_grid)))
        depth_attenuation_map = grid_interpolator(coords)
        augmented_data = data * depth_attenuation_map[..., None]

        if return_map:
            return augmented_data, depth_attenuation_map
        return augmented_data

    def randomize(self):
        self._random_attenuation = np.random.uniform(self.min_attenuation, self.max_attenuation, 1)[0]
        self._random_saturation_point = np.random.uniform(self.min_saturation_pt,
                                                          self.max_saturation_pt-np.finfo(dtype=np.float32).eps, 1)[0]


class Reverberation(AugmentationMethod):
    def __init__(self, max_width=0.15, max_height=0.05, max_tilt=15, max_offset=0.15, **kwargs):
        super().__init__(**kwargs)
        self.max_width = max_width
        self.max_tilt = max_tilt
        self.max_height = max_height
        self.max_offset = max_offset

        self._random_rev_center = None
        self._random_rev_width = None
        self._random_rev_height = None
        self._random_tilt = None
        self._random_offset = None
        self._random_rev_strength = None
        self._random_rev_offset_strength_ratio = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, return_map=False, *args, **kwargs):
        x, y = np.meshgrid(np.linspace(0, 1, data.shape[0], dtype=np.float32),
                           np.linspace(0, 1, data.shape[1], dtype=np.float32), copy=False, indexing='ij')

        x_mu = self._random_rev_center[0]
        y_mu = self._random_rev_center[1]
        offset = self._random_offset
        width = 2*self._random_rev_width**2
        height = 2*self._random_rev_height**2

        reverberation_map = self._random_rev_strength*np.exp(-((x - x_mu) ** 2 / height + (y - y_mu) ** 2 / width))
        offset_strength = self._random_rev_strength*self._random_rev_offset_strength_ratio

        if np.random.random() > 0.5:
            reverberation_map += offset_strength*np.exp(-((x - x_mu - offset) ** 2 / height + (y - y_mu) ** 2 / width))
            reverberation_map += offset_strength*np.exp(-((x - x_mu + offset) ** 2 / height + (y - y_mu) ** 2 / width))

        reverberation_map = rotate(reverberation_map, self._random_tilt, reshape=False)
        data_mask = data > 0.0
        reverberation_map = reverberation_map*data_mask[..., 0]
        augmented_data = np.clip(data+reverberation_map[..., None], 0, data.max())

        if return_map:
            return augmented_data, reverberation_map
        return augmented_data

    def randomize(self):
        self._random_rev_center = np.random.uniform(0, 1, 2)
        self._random_rev_width = np.random.uniform(0.01, self.max_width, 1)
        self._random_rev_height = np.random.uniform(0, self.max_height, 1)
        self._random_tilt = np.random.uniform(-self.max_tilt, self.max_tilt)
        self._random_offset = np.random.uniform(0.02, self.max_offset, 1)
        self._random_rev_strength = np.random.uniform(0.5, 1.0, 1)
        self._random_rev_offset_strength_ratio = np.random.uniform(0.1, 0.5, 1)


class RandomSectorCrop(AugmentationMethod):
    """
        Crop the sector to simulate different combinations of beam width and tilt. Input images must have correct aspect
        ratio.
        Set min_sector_tilt = -max_sector_tilt to get symmetrical tilt probability.
        Specific for phased array images.
        Also assumes that the beam depth of the augmented images starts at 0.

        Parameters
        ----------
        min_sector_width: number
            Narrowest sector width to generate in degrees
        max_sector_width: number
            Widest sector width to generate in degrees
        min_sector_tilt: number
            Minimal tilt angle in degrees
        max_sector_tilt: number
            Maximal tilt angle in degrees
    """

    def __init__(self, min_sector_width=60, max_sector_width=60, min_sector_tilt=0,
                 max_sector_tilt=0, crop_bottom_edge=True, input_orientation="HW", **kwargs):
        super().__init__(**kwargs)
        self.input_orientation = input_orientation.lower()
        self.min_sector_width = min_sector_width
        self.max_sector_width = max_sector_width
        self.min_sector_tilt = min_sector_tilt
        self.max_sector_tilt = max_sector_tilt
        self.crop_bottom_edge = crop_bottom_edge
        self._masks_cache = None
        self._random_sector_width = None
        self._random_sector_tilt = None
        self._h = None
        self._w = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, return_map=False, *args, **kwargs):
        w, h, ch = data.shape

        if self.input_orientation == "hw":
            h, w = w, h

        self._h = h
        self._w = w
        mask = self.masks_cache[self._random_sector_width][self._random_sector_tilt]

        if self.input_orientation == "hw":
            mask = mask.transpose([1, 0, 2])

        augmented_data = data * mask

        if return_map:
            return augmented_data, mask
        return augmented_data

    @property
    def masks_cache(self):
        if self._masks_cache is None:
            self._masks_cache = {}
            for sector_width in range(self.min_sector_width, self.max_sector_width + 1):
                tilt_masks = {}
                for sector_tilt in range(self.min_sector_tilt, self.max_sector_tilt + 1):
                    tilt_masks[sector_tilt] = self.generate_mask(sector_width, sector_tilt)
                self._masks_cache[sector_width] = tilt_masks
            return self._masks_cache
        else:
            return self._masks_cache

    def generate_mask(self, sector_width, sector_tilt):
        mask = np.ones((self._w, self._h, 1), dtype=np.float32)
        # Side edges
        left_half_sector_rad = np.deg2rad(-sector_width / 2 + sector_tilt)
        right_half_sector_rad = np.deg2rad(sector_width / 2 + sector_tilt)
        for i in range(self._h):
            border_left = np.round(self._w / 2 + np.tan(left_half_sector_rad) * i).astype(np.int)
            border_right = np.round(self._w / 2 + np.tan(right_half_sector_rad) * i).astype(np.int)
            if (border_left < 0) and (border_right > self._w):
                break
            mask[0:max(0, border_left), i] = 0.
            mask[min(self._w, border_right):self._w, i] = 0.
        # Bottom edge
        if self.crop_bottom_edge:
            for i in range(self._w):
                border_bottom = np.int(np.round(np.sqrt(self._h ** 2 - (i - self._w / 2) ** 2)))
                mask[i, border_bottom:] = 0.
        return mask

    def randomize(self):
        self._random_sector_width = np.random.randint(self.min_sector_width, self.max_sector_width + 1)
        self._random_sector_tilt = np.random.randint(self.min_sector_tilt, self.max_sector_tilt + 1)


class RadialDropouts(AugmentationMethod):
    """
        US images have radial coordinate system, so we apply radial dropout instead of the usual cartesian dropout.
        Specific for phased array images.
        Also assumes that the beam depth of the augmented images starts at 0.

        The sector will be divided into depth_splits * width_split areas. dropout_ratio of the total number of areas
        will be set to 0 at augmentation time.

        Parameters
        ----------
        sector_width: number
            Sector width in degrees of the images. If you don't know what is the width of your images (or have images
            with a variable width), use a slightly overestimated width here to avoid consistently removing the sides of
            the sector.
        min_depth_splits: int
            Minimal number of splits to divide the depth axis to generate the dropout areas
        max_depth_splits: int
            Maximal number of splits to divide the depth axis to generate the dropout areas
        min_width_splits: int
            Minimal number of splits to divide the width axis to generate the dropout areas
        max_width_splits: int
            Maximal number of splits to divide the width axis to generate the dropout areas
        min_dropout_ratio: float
            Minimal dropout probability
        max_dropout_ratio: float
            Maximal dropout probability
    """

    def __init__(self, sector_width=90,
                 min_depth_splits: int = 3, max_depth_splits: int = 3,
                 min_width_splits: int = 3, max_width_splits: int = 3,
                 min_dropout_ratio: float = 0.2, max_dropout_ratio: float = 0.2,
                 input_orientation="HW", **kwargs):
        super().__init__(**kwargs)
        self.input_orientation = input_orientation.lower()
        self.sector_width = sector_width
        self.min_depth_splits = min_depth_splits
        self.min_width_splits = min_width_splits
        self.min_dropout_ratio = min_dropout_ratio
        self.max_depth_splits = max_depth_splits
        self.max_width_splits = max_width_splits
        self.max_dropout_ratio = max_dropout_ratio

        self._random_depth_splits = None
        self._random_width_splits = None
        self._random_dropout_ratio = None

    @AugmentationMethod.ensure_dtype
    def transform(self, data, return_map=False, *args, **kwargs):
        h, w, ch = data.shape

        if self.input_orientation == "wh":
            h, w = w, h

        dropout_image_size = (50, 50)
        r_axis = np.linspace(0, h, dropout_image_size[1], endpoint=True)
        theta_axis = np.linspace(-np.deg2rad(self.sector_width / 2), np.deg2rad(self.sector_width / 2),
                                 dropout_image_size[0],
                                 endpoint=True)

        dropout_probs = np.random.uniform(size=(self._random_width_splits, self._random_depth_splits))
        dropout_image = np.where(dropout_probs < self._random_dropout_ratio, 0, 1)
        dropout_image = resize(dropout_image, output_shape=dropout_image_size, order=0, preserve_range=True)

        grid_interpolator = RegularGridInterpolator((theta_axis, r_axis), dropout_image, bounds_error=False,
                                                    fill_value=0, method='nearest')
        w_bis = int(2 * h * np.sin(np.deg2rad(self.sector_width / 2)))
        x = np.linspace(np.sin(theta_axis[0]), np.sin(theta_axis[-1]), w_bis) * (r_axis[-1])
        z = np.linspace(r_axis[0] * min(np.cos(theta_axis[0]), np.cos(theta_axis[-1])), r_axis[-1], h)

        x_grid, z_grid = np.meshgrid(x, z, indexing='ij')

        coords = np.dstack((np.arctan2(x_grid, z_grid), np.hypot(x_grid, z_grid)))
        dropout_map = grid_interpolator(coords)
        if self.input_orientation == "hw":
            dropout_map = dropout_map.T
            padding = data.shape[1] - dropout_map.shape[1]
            if padding < 0:
                dropout_map = dropout_map[:, -padding // 2: -padding // 2 + data.shape[1]]
            else:
                dropout_map = np.pad(dropout_map, ((0, 0), (padding // 2, padding - padding // 2)))
        else:
            padding = data.shape[0] - dropout_map.shape[0]
            if padding < 0:
                dropout_map = dropout_map[-padding // 2: -padding // 2 + data.shape[0], ...]
            else:
                dropout_map = np.pad(dropout_map, ((padding // 2, padding - padding // 2), (0, 0)))

        augmented_data = data * dropout_map[..., None]

        if return_map:
            return augmented_data, dropout_map
        return augmented_data

    def randomize(self):
        self._random_depth_splits = np.random.randint(self.min_depth_splits, self.max_depth_splits + 1)
        self._random_width_splits = np.random.randint(self.min_width_splits, self.max_width_splits + 1)
        self._random_dropout_ratio = np.random.uniform(self.min_dropout_ratio, self.max_dropout_ratio)


class Blackout(AugmentationMethod):
    def __init__(self, max_discard=0.5, flow_indices=None, segmentation_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.max_discard = max_discard
        self.flow_indices = flow_indices
        self.apply_to = 'all'
        self.segmentation_indices = segmentation_indices

    @AugmentationMethod.ensure_dtype
    def transform(self, data, data_index=None, *args, **kwargs):
        data = np.copy(data)
        if len(data.shape) > 3:
            for i in range(len(data)):
              if np.random.rand((1)) <= self._random_discard:
                data[i] = data[i]*0
        return data

    def randomize(self):
        self._random_discard = np.random.rand((1)) * self.max_discard

class RandomCrop(AugmentationMethod):
    def __init__(self, width, height, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.apply_to = 'all'

        self._random_x_start_scale = None
        self._random_y_start_scale = None

    def transform(self, data, *args, **kwargs):
        if self.inputs_pattern is None or is_height_width_oriented(self.inputs_pattern):
            x, y = self.height, self.width
        else:
            x, y = self.height, self.width

        x_start_max = data.shape[0] - x if x <= data.shape[0] else 0
        y_start_max = data.shape[1] - y if y <= data.shape[1] else 0
        x_min = int(self._random_x_start_scale*x_start_max)
        y_min = int(self._random_y_start_scale*y_start_max)
        return data[x_min:x_min+x, y_min:y_min+y]

    def randomize(self):
        self._random_x_start_scale = np.random.uniform(0, 1)
        self._random_y_start_scale = np.random.uniform(0, 1)

