"""
Module for handling data loading and preprocessing using torch Datasets.
"""
import os
from abc import abstractmethod
from typing import Callable, Type, Union

import torch
import torchvision.transforms.v2 as tvt
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision import tv_tensors

from dgs.models.module import BaseModule
from dgs.models.states import DataSample
from dgs.utils.files import is_project_dir, is_project_file, to_abspath
from dgs.utils.image import CustomCropResize, CustomResize, CustomToAspect, load_image
from dgs.utils.types import Config, FilePath, NodePath, Validations  # pylint: disable=unused-import
from dgs.utils.validation import validate_bboxes, validate_images, validate_key_points

base_dataset_validations: Validations = {
    "crop_mode": ["str", ("in", CustomToAspect.modes)],
    "crop_size": [("instance", tuple), ("len", 2), lambda x: x[0] > 0 and x[1] > 0],
    "dataset_path": ["str", "folder exists in project"],
}


def collate_devices(batch: list[torch.device], *_args, **_kwargs) -> torch.device:
    """Collate a batch of devices into a single device."""
    device = batch[0]
    if any(d != device for d in batch):
        raise ValueError("Devices should be equal for each sample in a batch.")
    return device


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
    if any(bb.format != bb_format or bb.canvas_size != canvas_size for bb in batch):
        raise ValueError("Canvas size and format has to be equal for all bounding boxes in a batch.")

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


def collate_data_samples(batch: list[DataSample]) -> DataSample:
    """Collate function for multiple DataSamples, to flatten / squeeze the shapes and keep the tv_tensors classes.

    The default collate function messes up a few of the dimensions and removes custom tv_tensor classes.
    Therefore, add custom collate functions for the tv_tensors classes.
    Additionally, custom torch tensor collate, which stacks tensors only if first dimension != 1, cat otherwise.

    Args:
        batch: A list of `DataSamples`, each `DataSample` containing a single sample or bounding box.

    Returns:
        One single `DataSample` object, containing a batch of samples or bounding boxes.
    """
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


class BaseDataset(BaseModule, TorchDataset):
    """Base class for custom datasets.

    Using the Bounding Box as Index
    -------------------------------

    The BaseDataset assumes that one sample of dataset (one :meth:`self.__getitem__` call)
    contains one single bounding box.
    It should be possible to work with a plain dict,
    but to have a few quality-of-life features, the DataSample class was implemented.

    Why not use the Image ID as Index?
    ----------------------------------

    This is **not** chosen since the batch-size might vary when using the image index to create batches,
    because every image can have a different number of detections.
    With a batch size of B, we obtain B original images.
    Every one of those has a specific number of detections N, ranging from zero to an arbitrary number.

    This means that for obtaining batches of a constant size, it is necessary to 'flatten' the inputs.
    Thus creating a mapping from image name and bounding box to the respective dataset.
    With that in place, the DataLoader can retrieve batches with the same batch size.
    The detections of one image might be split into different batches.

    The other option is to have batches with slightly different sizes.
    The DataLoader loads a fixed batch of images, the Dataset computes the resulting detections and returns those.
    """

    data: list
    """Arbitrary data, which will be converted using :meth:`self.arbitrary_to_ds()`"""

    def __call__(self, *args, **kwargs) -> any:
        """Has to override call from BaseModule"""
        raise NotImplementedError("Dataset can't be called.")

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

    def __len__(self) -> int:
        """Override len() functionality for torch.

        The Length of the dataset is technically just len(data), but might be obtained otherwise.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> DataSample:
        """Retrieve data at index from given dataset.

        Args:
            idx: index of the dataset object. Is a reference to the same object as len().

        Returns:
            The pre-computed backbone outputs.
        """
        sample: DataSample = self.arbitrary_to_ds(self.data[idx]).to(self.device)
        if "image_crop" not in sample or "local_coordinates" not in sample:
            self.get_image_crop(sample)
        return sample

    @abstractmethod
    def arbitrary_to_ds(self, a) -> DataSample:
        """Given a single arbitrary data sample, convert it to a DataSample object."""
        raise NotImplementedError

    def get_image_crop(self, ds: DataSample):
        """Add image crop and local keypoints to given sample"""
        ds.to(self.device)

        structured_input = {
            "image": validate_images(load_image(ds.filepath)),
            "box": validate_bboxes(ds.bbox),
            "keypoints": validate_key_points(ds.keypoints),
            "output_size": self.params["crop_size"],
            "mode": self.params["crop_mode"],
        }
        new_sample = self.transform_crop_resize()(structured_input)
        ds.image_crop = new_sample["image"]
        ds.keypoints_local = new_sample["keypoints"]

    @staticmethod
    def transform_resize_image() -> tvt.Compose:
        """Given an image, bboxes, and key-points, resize them with custom modes.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {
            "image": tv_tensors.Image,
            "box": tv_tensors.BoundingBoxes,
            "keypoints": torch.Tensor,
            "output_size": ImgShape,
            "mode": str,
        }

        Returns:
            A composed torchvision function that accepts a dict as input.
        """
        return tvt.Compose(
            [
                CustomToAspect(),  # make sure the image has the correct aspect ratio for the backbone model
                CustomResize(),
                tvt.ClampBoundingBoxes(),  # keep bboxes in their canvas_size
                # tvt.SanitizeBoundingBoxes(),  # clean up bboxes if available
                tvt.ToDtype({tv_tensors.Image: torch.float32}, scale=True),
            ]
        )

    @staticmethod
    def transform_crop_resize() -> tvt.Compose:
        """Given one single image, with its corresponding bounding boxes and key-points,
        obtain a cropped image for every bounding box with localized key-points.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {
            "image": tv_tensors.Image,
            "box": tv_tensors.BoundingBoxes,
            "keypoints": torch.Tensor,
            "output_size": ImgShape,
            "mode": str,
        }

        Returns:
            A composed torchvision function that accepts a dict as input.

            After calling this transform function, some values will have different shapes:

            image
                Now contains the image crops as tensor of shape ``[N x C x H x W]``.
            bboxes
                Zero, one, or multiple bounding boxes for this image as tensor of shape ``[N x 4]``.
                And the bounding boxes got transformed into the `XYWH` format.
            coordinates
                Now contains the joint coordinates of every detection in local coordinates in shape ``[N x J x 2|3]``.

        """
        return tvt.Compose(
            [
                tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),  # xyxy to easily obtain ltrb
                tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
                # tvt.SanitizeBoundingBoxes(),  # clean up bboxes
                CustomCropResize(),  # crop the image at the four corners specified in bboxes
                tvt.ClampBoundingBoxes(),  # duplicate ?
                # tvt.SanitizeBoundingBoxes(),  # duplicate ?
            ]
        )

    def get_path_in_dataset(self, path: FilePath) -> FilePath:
        """Given an arbitrary file- or directory-path, return its absolute path.

        1. check whether the path is a valid absolute path
        2. check whether the path is a valid project path
        3. check whether the path is an existing path within self.params["dataset_path"]

        Returns:
            The absolute found path to the file or directory.

        Raises:
            FileNotFoundError: If the path is not found.
        """
        if os.path.exists(path):
            return os.path.normpath(path)
        if is_project_file(path) or is_project_dir(path):
            return to_abspath(path)
        dataset_path = os.path.join(self.params["dataset_path"], str(path))
        if is_project_file(dataset_path) or is_project_dir(dataset_path):
            return to_abspath(dataset_path)
        raise FileNotFoundError(f"Could not find a path to file or directory at {path}")
