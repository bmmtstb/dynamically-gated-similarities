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
Validator = Callable[[any, any], bool]  # function for validating a value
Validations = dict[str, list[str | tuple[str, any] | Validator | None]]

# Pose State
PoseStateTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Torch
Device = Union[torch.device, str]

# Images
TVImage = tv_Image
TVVideo = tv_Video
ByteImage = torch.ByteTensor
FloatImage = torch.FloatTensor
Image = Union[TVImage, ByteImage, FloatImage]

TVImages = tv_Image
ByteImages = torch.ByteTensor
FloatImages = torch.FloatTensor
Images = Union[TVImages, ByteImages, FloatImages]

ImgShape = tuple[int, int]  # (w, h) as target width and height of the image
