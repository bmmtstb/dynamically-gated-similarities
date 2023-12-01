"""
Module for handling data loading and preprocessing using torch Datasets.
"""
from abc import abstractmethod

import numpy as np
import torch
import torchvision.transforms.v2 as tvt
from torch.utils.data import Dataset as TorchDataset
from torchvision import tv_tensors

from dgs.models.module import BaseModule
from dgs.models.states import BackboneOutput, BackboneOutputs
from dgs.utils.image import CustomCropResize, CustomToAspect
from dgs.utils.types import Config, ImgShape, NodePath


class BaseDataset(TorchDataset, BaseModule):
    """Custom Dataset"""

    filenames: np.ndarray
    """(np.ndarray) store list of filenames as numpy array, to reduce memory usage on multiple devices"""

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        # structured_input: dict[str, any] = {
        # "image": None, "bboxes": None, "coordinates": None, "mode": None, "output_size": None,
        # }

    @abstractmethod
    def __len__(self) -> int:
        """Override len() functionality for torch."""
        return len(self.filenames)

    @abstractmethod
    def __getitem__(self, idx: int) -> BackboneOutput:
        """Retrieve data at index from given dataset.

        Args:
            idx: index of the data object. Is a reference to the same object as len().

        Returns:
            Precomputed backbone output
        """
        raise NotImplementedError

    def __getitems__(self, indices: list[int]) -> BackboneOutputs:
        """Get batch of data given list of indices.

        This method accepts a list of filename indices within this batch and returns a list of samples.
        Subclasses can optionally implement __getitems__() for speeding up batched samples loading.

        Args:
            indices: A list of filename indices to retrieve the image data from
        """
        raise NotImplementedError

    @staticmethod
    def transform_resize_image(backbone_size: ImgShape) -> tvt.Compose:
        """Given an image, bboxes, and key-points, resize them with custom modes.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {\
            "image": tv_tensors.Image,\
            "bboxes": tv_tensors.BoundingBoxes,\
            "coordinates": tv_tensors.Mask,\
            "output_size": ImgShape,\
            "mode": str,\
        }

        Args:
            backbone_size: (w, h) as width and height of the image that is put as input into backbone

        Returns:
            A composed torchvision function that accepts a dict as input
        """
        return tvt.Compose(
            [
                tvt.ToDtype(
                    {tv_tensors.Image: torch.uint8}, scale=True
                ),  # kind of optional, because either load image returns uint8 or another dtype was wanted
                CustomToAspect(),  # make sure the image has the correct aspect ratio for the backbone model
                tvt.Resize(
                    backbone_size, antialias=True
                ),  # make sure the image has the correct input size for the backbone model
                tvt.ClampBoundingBoxes(),  # keep bboxes in their canvas_size
                tvt.SanitizeBoundingBoxes(labels_getter="bboxes"),  # clean up bboxes if available
                tvt.ToDtype({tv_tensors.Image: torch.float32}, scale=True),
            ]
        )

    @staticmethod
    def transform_crop_resize() -> tvt.Compose:
        """Given one single image, with its corresponding bounding boxes and key-points,
        obtain a cropped image for every bounding box with localized key-points.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {\
            "image": tv_tensors.Image,\
            "bboxes": tv_tensors.BoundingBoxes,\
            "coordinates": tv_tensors.Mask,\
            "output_size": ImgShape,\
            "mode": str,\
        }

        Returns:
            A composed torchvision function that accepts a dict as input.

            After calling this transform function, some values will have different shapes:

            image
                Now contains the image crops as tensor of shape ``[N x C x H x W]``

            bboxes
                Zero, one, or multiple bounding boxes for this image as tensor of shape ``[N x 4]``

                And the bounding boxes got transformed into the XYWH format.

            coordinates:
                Now contains the joint coordinates of every detection in local coordinates in shape ``[N x J x 2|3]``
        """
        return tvt.Compose(
            [
                tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),  # xyxy to easily obtain ltrb
                tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
                tvt.SanitizeBoundingBoxes(),  # clean up bboxes
                CustomCropResize(),  # crop the image at the four corners specified in bboxes
                tvt.ClampBoundingBoxes(),  # duplicate ?
                tvt.SanitizeBoundingBoxes(),  # duplicate ?
            ]
        )
