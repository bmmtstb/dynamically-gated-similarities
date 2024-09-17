"""
definition of regularly used types
"""

from typing import Callable, TypeAlias, Union

import torch as t
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device as TorchDevice
from torchvision.tv_tensors import Image as tv_Image, Mask as tv_Mask, Video as tv_Video

# Configuration
Config: TypeAlias = dict[str, any]
"""A nested configuration, describing a module or the whole tracker."""
NodePath: TypeAlias = list[str]
"""A list of key-names used for traversing through a config."""

# File handling
FilePath: TypeAlias = str
"""The path to a file or directory. Can be project-local or global."""
FilePaths: TypeAlias = tuple[FilePath, ...]
"""Multiple FilePaths as tuple."""

# Validation
Validator: TypeAlias = Callable[[any, any], bool]
"""A function for validating a value of a Config object. Accepts up to two values and returns a boolean."""
Validations: TypeAlias = dict[str, list[Union[str, type, tuple[str, any], Validator]]]
"""A dictionary of validations, mapping a value of a given Config to some sort of validation."""

# Data Handling
DataGetter = Callable[["State"], tuple[Union[t.Tensor, any], ...]]
"""Function to extract specific data from a :class:`State`."""

# Modules
Instance: TypeAlias = Union[str, type]
"""An instance to be loaded is either the name of that instance or a class-type."""
Metric: TypeAlias = nn.Module
"""A module or function that computes a metric."""
Loss: TypeAlias = nn.Module
"""A module or function that computes a loss."""
Scheduler: TypeAlias = LRScheduler
"""A Scheduler used to update the learning rate during training."""
Results: TypeAlias = dict[str, any]
"""A dictionary of results, as name->metric or name->iterable of metrics."""

# Torch
Device: TypeAlias = Union[TorchDevice, str]
r"""A Torch device, either descriptive string (e.g. 'cpu') or a regular :class:`torch.device` object."""

# Images
Video: TypeAlias = Union[tv_Video, t.Tensor]
"""A tensor based video, either torchvision or regular tensor. Shape ``[B x C x H x W]``."""
Image: TypeAlias = Union[tv_Image, t.Tensor]
"""A tensor based image with shape ``[B x C x H x W]``."""
Images: TypeAlias = list[Image]
"""A list with length ``B``, containing Images with shape ``[1 x C x H x W]``."""
Heatmap: TypeAlias = Union[tv_Mask, t.Tensor]
"""Heatmap as mask with shape ``[(B x) J x h x w]``."""
ImgShape: TypeAlias = tuple[int, int]
"""Shape of an image as tuple like ``(height, width)``."""
