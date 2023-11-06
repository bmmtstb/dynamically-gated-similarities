"""
definition of regularly used types
"""
import torch

Config = dict[str, bool | str | int | float | dict]
Path = list[str]

PoseState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
