"""
Helper methods for loading and manipulating data, Datasets, and Dataloader.
"""

from typing import Callable, Type, Union

import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision import tv_tensors

from dgs.utils.states import DataSample


def collate_devices(batch: list[torch.device], *_args, **_kwargs) -> torch.device:
    """Collate a batch of devices into a single device."""
    return batch[0]


def collate_tensors(batch: list[torch.Tensor], *_args, **_kwargs) -> torch.Tensor:
    """Collate a batch of tensors into a single one.

    Will use torch.cat() if the first dimension has a shape of one, otherwise torch.stack()
    """
    if len(batch[0].shape) > 0 and batch[0].shape[0] == 1:
        return torch.cat(batch)
    return torch.stack(batch)


def collate_bboxes(batch: list[tv_tensors.BoundingBoxes], *_args, **_kwargs) -> tv_tensors.BoundingBoxes:
    """Collate a batch of bounding boxes into a single one.
    It is expected that all bounding boxes have the same canvas size and format.

    Raises:
        ValueError: If the batch of bounding boxes has different attributes.
    """
    bb_format: tv_tensors.BoundingBoxFormat = batch[0].format
    canvas_size = batch[0].canvas_size

    return tv_tensors.BoundingBoxes(
        torch.cat(batch),
        canvas_size=canvas_size,
        format=bb_format,
    )


def collate_tvt_tensors(
    batch: list[Union[tv_tensors.Image, tv_tensors.Mask, tv_tensors.Video]], *_args, **_kwargs
) -> Union[tv_tensors.Image, tv_tensors.Mask, tv_tensors.Video]:
    """Collate a batch of tv_tensors into a batched version of it."""
    return tv_tensors.wrap(torch.cat(batch), like=batch[0])


def collate_data_samples(batch: Union[list[DataSample], DataSample]) -> DataSample:
    """Collate function for multiple DataSamples, to flatten / squeeze the shapes and keep the tv_tensors classes.

    The default collate function messes up a few of the dimensions and removes custom tv_tensor classes.
    Therefore, add custom collate functions for the tv_tensors classes.
    Additionally, custom torch tensor collate, which stacks tensors only if first dimension != 1, cat otherwise.

    Args:
        batch: A list of `DataSamples`, each `DataSample` containing a single sample or bounding box.

    Returns:
        One single `DataSample` object, containing a batch of samples or bounding boxes.
    """
    if isinstance(batch, DataSample):
        return batch

    custom_collate_map: dict[Type, Callable] = default_collate_fn_map.copy()
    custom_collate_map.update(
        {
            str: lambda str_batch, *args, **kwargs: tuple(s for s in str_batch),
            tuple: lambda t_batch, *args, **kwargs: sum(t_batch, ()),
            tv_tensors.BoundingBoxes: collate_bboxes,
            (tv_tensors.Image, tv_tensors.Video, tv_tensors.Mask): collate_tvt_tensors,
            torch.device: collate_devices,
            torch.Tensor: collate_tensors,  # override regular tensor collate to *not* add another dimension
        }
    )
    c_batch: dict[str, any] = collate(batch, collate_fn_map=custom_collate_map)

    # shouldn't need validation, because every single DataSample has been validated before.
    return DataSample(**c_batch, validate=False)
