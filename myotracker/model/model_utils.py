# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-6


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def reduce_masked_mean(inputs, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(inputs)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def bilinear_sampler(inputs, coords):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    The coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
    Returns:
        Tensor: sampled points.
    """
    sizes = inputs.shape[2:]
    coords = coords * torch.tensor([2.0/sizes[1], 2.0/sizes[0]], dtype=coords.dtype, device=coords.device) - 1.0
    return F.grid_sample(inputs, coords, align_corners=False)


def sample_features4d(inputs, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    2)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = inputs.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(inputs, coords)
    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C
