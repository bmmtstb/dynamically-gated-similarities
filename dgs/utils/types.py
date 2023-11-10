"""
definition of regularly used types
"""
import torch

# Configuration
Config = dict[str, bool | str | int | float | dict]
Path = list[str]


# Pose State
PoseStateTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Torch
Device = torch.device | str
