"""
Module for handling data loading and preprocessing using torch Datasets.
"""

import torch
import torchvision.transforms.v2 as tvt
from torch.utils.data import Dataset as TorchDataset
from torchvision import tv_tensors

from dgs.models.module import BaseModule
from dgs.models.states import DataSample
from dgs.utils.image import CustomCropResize, CustomResize, CustomToAspect, load_image
from dgs.utils.types import Config, ImgShape, NodePath, Validations

base_dataset_validations: Validations = {
    # "resize_mode": ["str", ("in", CustomToAspect.modes)],
    # "backbone_size": [("or",
    #   ("and", ("instance", list), ("len", 2), lambda x: x[0] > 0 and x[1] > 0),
    #   ("None", ...))
    # ],
    "crop_mode": ["str", ("in", CustomToAspect.modes)],
    "crop_size": [("instance", list), ("len", 2), lambda x: x[0] > 0 and x[1] > 0],
}


class BaseDataset(TorchDataset, BaseModule):
    """Base class for custom datasets.

    Using the Bounding Box as Index
    -------------------------------
    The BaseDataset assumes that one sample of dataset (one __getitem__ call) contains one single bounding box.
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

    Methods:
        self.transform_resize_image
        self.transform_crop_resize
    """

    data: list[DataSample]
    """Generator or List of all the dataset samples. Indexed per bounding box instead of per image.
    
    Every dict contains a single bounding box sample.
    """

    cropped_shape: ImgShape
    """Shape of the cropped image"""

    def __call__(self, *args, **kwargs) -> any:
        """Has to override call from BaseModule"""
        raise NotImplementedError("Dataset can't be called.")

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

    def __len__(self) -> int:
        """Override len() functionality for torch."""
        return len(self.data)

    def __getitem__(self, idx: int) -> DataSample:
        """Retrieve data at index from given dataset.

        Args:
            idx: index of the dataset object. Is a reference to the same object as len().

        Returns:
            Precomputed backbone output
        """
        sample = self.data[idx]
        if "image_crop" not in sample or "local_coordinates" not in sample:
            structured_input = {
                "image": load_image(sample.filepath),
                "bboxes": sample.bbox,
                "coordinates": sample.keypoints,
                "output_size": self.params["crop_size"],
                "mode": self.params["crop_mode"],
            }
            new_sample = self.transform_crop_resize()(structured_input)
            sample.image_crop = new_sample["image_crop"]
            sample.local_keypoints = new_sample["local_coordinates"]

        return sample

    @staticmethod
    def transform_resize_image() -> tvt.Compose:
        """Given an image, bboxes, and key-points, resize them with custom modes.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {\
            "image": tv_tensors.Image,\
            "bboxes": tv_tensors.BoundingBoxes,\
            "coordinates": torch.Tensor,\
            "output_size": ImgShape,\
            "mode": str,\
        }

        Returns:
            A composed torchvision function that accepts a dict as input
        """
        return tvt.Compose(
            [
                CustomToAspect(),  # make sure the image has the correct aspect ratio for the backbone model
                CustomResize(),
                tvt.ClampBoundingBoxes(),  # keep bboxes in their canvas_size
                tvt.SanitizeBoundingBoxes(),  # clean up bboxes if available
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
            "coordinates": torch.Tensor,\
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

                And the bounding boxes got transformed into the XYWH box_format.

            coordinates:
                Now contains the joint coordinates of every detection in local coordinates in shape ``[N x J x 2|3]``
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
