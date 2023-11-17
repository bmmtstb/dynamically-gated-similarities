"""
definition of regularly used types
"""
from pathlib import Path
from typing import Callable

import torch
from easydict import EasyDict

# Configuration
Config = dict[str, any] | EasyDict  # is actually an EasyDict but can't use that as variable type hint
NodePath = list[str]
FilePath = str | Path
Validator = Callable[[any, any], bool]  # function for validating a value
Validations = dict[str, list[str | tuple[str, any] | Validator | None]]

# Pose State
PoseStateTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Torch
Device = torch.device | str
