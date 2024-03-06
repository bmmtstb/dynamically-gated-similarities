"""
General utility functions.
"""

import os.path
import sys
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.io import write_jpeg
from torchvision.transforms import v2 as tvt
from torchvision.transforms.functional import convert_image_dtype

from dgs.utils.files import mkdir_if_missing
from dgs.utils.image import CustomCropResize, load_image
from dgs.utils.types import Device, FilePath, FilePaths, Image


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Clone and convert torch tensor to numpy.

    Args:
        t: Torch tensor on arbitrary hardware.

    Returns:
        A single numpy array with the same shape and type as the original tensor.
    """
    # Detach creates a new tensor with the same data, so it is important to clone.
    # May not be necessary if moving from GPU to CPU, but better safe than sorry
    return t.detach().cpu().clone().numpy()


def extract_crops_from_images(
    img_fps: Union[list[FilePath], FilePaths],
    new_fps: Union[list[FilePath], FilePaths],
    boxes: tv_tensors.BoundingBoxes,
    key_points: torch.Tensor = None,
    **kwargs,
) -> tuple[Image, torch.Tensor]:
    """Given a list of original image paths and a list of target crops paths,
    use the given bounding boxes to extract their content as image crops and save them as new images.

    Does only work if the images have the same size, because otherwise the bounding-boxes would not match anymore.

    Notes:
        It is expected that ``img_fps``, ``new_fps``, and ``boxes`` have the same length.

    Args:
        img_fps: An iterable of absolute paths pointing to the original images.
        new_fps: An iterable of absolute paths pointing to the image crops.
        boxes: The bounding boxes as tv_tensors.BoundingBoxes of arbitrary format.
        key_points (torch.Tensor, optional): Key points of the respective images.
            The key points will be transformed with the images. Default None just means that a placeholder is passed.

    Keyword Args:
        crop_size (ImgShape): The target shape of the image crops. Defaults to ``(256, 256)``.
        device (Device): Device to run the cropping on. Defaults to "cuda" if available "cpu" otherwise.
        transform (tvt.Compose): A torchvision transform given as Compose to get the crops from the original image.
            Defaults to a version of CustomCropResize.
        transform_mode (str): Defines the resize mode in the transform function.
            Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`. Default "zero-pad".
        quality (int): The quality to save the jpegs as. Default 90. Default of torchvision is 75.

    Returns:
        crops, key_points: The computed image crops and their respective key points on the device specified in kwargs.
        The image-crops are saved already, which means in most cases the return values can be ignored.

    Raises:
        ValueError: If input lengths don't match.
    """
    if not len(img_fps) == len(new_fps) == boxes.shape[-2] or (
        key_points is not None and not len(key_points) == len(img_fps)
    ):
        raise ValueError("There has to be an equal amount of image-, crop-paths, boxes, and key points if present.")

    # extract kwargs
    device: Device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    transform = kwargs.get(
        "transform",
        tvt.Compose(  # default
            [
                tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),
                tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
                CustomCropResize(),
            ]
        ),
    )

    imgs: Image = load_image(
        filepath=tuple(img_fps),
        device=device,
    )

    # pass original images through CustomResizeCrop transform and get the resulting image crops on the cpu
    res = transform(
        {
            "image": imgs,
            "box": boxes,
            "keypoints": key_points if key_points is not None else torch.zeros((imgs.shape[-4], 1, 2), device=device),
            "mode": kwargs.get("transform_mode", "zero-pad"),
            "output_size": kwargs.get("crop_size", (256, 256)),
        }
    )
    crops: torch.Tensor = res["image"].cpu()
    kps: torch.Tensor = res["keypoints"].cpu()

    for fp, crop, kp in zip(new_fps, crops, kps):
        mkdir_if_missing(os.path.dirname(fp))
        write_jpeg(input=convert_image_dtype(crop, torch.uint8), filename=fp, quality=kwargs.get("quality", 90))
        if key_points is not None:
            torch.save(kp, str(fp).replace(".jpg", ".pt"))

    return crops.to(device=device), res["keypoints"]


class HidePrint:
    """Safely disable printing for a block of code.

    Source: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print/45669280#45669280

    Examples:
        >>> with HidePrint():
        ...     print("Hello")
        ... print("Bye")
        Bye
    """

    _original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def ids_to_one_hot(ids: Union[torch.Tensor, torch.Tensor], nof_classes: int) -> torch.Tensor:
    """Given a tensor containing the class ids as LongTensor, return the one hot representation as LongTensor."""
    return F.one_hot(ids.long(), nof_classes)  # pylint: disable=not-callable
