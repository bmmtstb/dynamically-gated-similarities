"""
definition of regularly used types
"""

from typing import Callable, Union

import torch
from easydict import EasyDict
from torch import nn
from torch.types import Device as TorchDevice
from torchvision.tv_tensors import Image as tv_Image, Mask as tv_Mask, Video as tv_Video

# Configuration
Config = Union[dict[str, any], EasyDict]  # is actually an EasyDict but can't use that as variable type hint
"""A nested configuration, describing a module or the whole tracker."""

NodePath = list[str]
"""A list of key-names used for traversing through a Config."""

FilePath = str
"""The path to a file or directory. Can be project-local or global."""

FilePaths = tuple[FilePath, ...]
"""Multiple FilePaths as tuple."""

Validator = Callable[[any, any], bool]
"""A function for validating a value of a Config object. Accepts up to two values and returns a boolean."""

Validations = dict[str, list[Union[str, tuple[str, any], Validator, None]]]
"""A dictionary of validations, mapping a value of a given Config to some sort of validation."""

# Data Handling
DataGetter = Callable[["DataSample"], tuple[Union[torch.Tensor, any], ...]]
"""Function to extract specific data from a DataSample."""

# Torch
Device = Union[TorchDevice, str]
"""Torch device, either descriptive string (e.g. "cpu" or "cuda:0") or torch.device object."""

Metric = nn.Module
"""A module or function that computes a metric."""

Loss = nn.Module
"""A module or function that computes a loss."""

# Images
TVImage = tv_Image
"""Torchvision tv_tensor.Image"""
TVVideo = tv_Video
"""Torchvision tv_tensor.Video"""
ByteImage = torch.ByteTensor
"""Torch.ByteTensor otherwise known as dtype=torch.uint8 with values in range 0..255"""
FloatImage = torch.FloatTensor
"""Torch.FloatTensor with values in range ..."""
Image = Union[TVImage, ByteImage, FloatImage]
"""A tensor based image with shape ``[(B x) C x H x W]``"""

ImgShape = tuple[int, int]
"""Shape of an image as (h, w)"""

Heatmap = tv_Mask
"""Heatmap as mask with shape ``[(B x) J x h x w]``"""
