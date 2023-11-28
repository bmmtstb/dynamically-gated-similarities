"""
definition of regularly used types
"""
from typing import Callable, Union

import torch
from easydict import EasyDict

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
