"""
definition of regularly used types
"""

from typing import Callable, Union

import torch
from easydict import EasyDict
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device as TorchDevice
from torchvision.tv_tensors import Image as tv_Image, Mask as tv_Mask, Video as tv_Video

# Configuration
Config = Union[dict[str, any], EasyDict]  # is actually an EasyDict but can't use that as variable type hint
"""A nested configuration, describing a module or the whole tracker."""
NodePath = list[str]
"""A list of key-names used for traversing through a config."""

# File handling
FilePath = str
"""The path to a file or directory. Can be project-local or global."""
FilePaths = tuple[FilePath, ...]
"""Multiple FilePaths as tuple."""

# Validation
Validator = Callable[[any, any], bool]
"""A function for validating a value of a Config object. Accepts up to two values and returns a boolean."""
Validations = dict[str, list[Union[str, type, tuple[str, any], Validator]]]
"""A dictionary of validations, mapping a value of a given Config to some sort of validation."""

# Data Handling
DataGetter = Callable[["DataSample"], tuple[Union[torch.Tensor, any], ...]]
"""Function to extract specific data from a :class:`DataSample`."""

# Modules
Instance = Union[str, type]
"""An instance to be loaded is either the name of that instance or a class-type."""
Metric = nn.Module
"""A module or function that computes a metric."""
Loss = nn.Module
"""A module or function that computes a loss."""
Scheduler = LRScheduler
"""A Scheduler used to update the learning rate during training."""

# Torch
Device = Union[TorchDevice, str]
"""Torch device, either descriptive string (e.g. "cpu" or "cuda:0") or a regular torch.device object."""

# Images
Video = Union[tv_Video, torch.Tensor]
"""Torchvision tv_tensor.Video or regular tensor"""
Image = Union[tv_Image, torch.Tensor]
"""A tensor based image with shape ``[(B x) C x H x W]``"""
Heatmap = Union[tv_Mask, torch.Tensor]
"""Heatmap as mask with shape ``[(B x) J x h x w]``"""
ImgShape = tuple[int, int]
"""Shape of an image as (h, w)"""
