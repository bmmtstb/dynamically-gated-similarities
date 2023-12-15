"""
definition of regularly used types
"""
from typing import Callable, Union

import torch
from easydict import EasyDict
from torchvision.tv_tensors import Image as tv_Image, Video as tv_Video

# Configuration
Config = dict[str, any] | EasyDict  # is actually an EasyDict but can't use that as variable type hint
NodePath = list[str]
FilePath = str
FilePaths = tuple[FilePath, ...]
Validator = Callable[[any, any], bool]  # function for validating a value
Validations = dict[str, list[str | tuple[str, any] | Validator | None]]

# Pose State
PoseStateTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""Current PoseState as tuple like (pose, jcs, bbox)"""


# Torch
Device = Union[torch.device, str]

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
