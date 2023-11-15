"""
definition of regularly used types
"""
from pathlib import Path

import torch
from easydict import EasyDict

# Configuration
Config = dict[str, any] | EasyDict  # is actually an EasyDict but can't use that as variable type hint
NodePath = list[str]
FilePath = str | Path


# Pose State
PoseStateTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Torch
Device = torch.device | str
