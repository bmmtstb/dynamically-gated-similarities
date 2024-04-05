"""
General utility functions.
"""

import os.path
import sys
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import tv_tensors as tvte
from torchvision.io import write_jpeg
from torchvision.transforms import v2 as tvt
from torchvision.transforms.functional import convert_image_dtype

from dgs.utils.config import DEF_CONF
from dgs.utils.files import mkdir_if_missing
from dgs.utils.image import CustomCropResize, load_image_list
from dgs.utils.types import Device, FilePath, FilePaths, Image, Images


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
    imgs: Images, bboxes: tvte.BoundingBoxes, kps: torch.Tensor = None, **kwargs
) -> tuple[Image, Union[torch.Tensor, None]]:
    """Given a list of images, use the bounding boxes to extract the respective image crops.
    Additionally, compute the local key-point coordinates if global key points are given.

    Args:
        imgs: A list containing one or multiple tv_tensors.Image tensors.
        bboxes: The bounding boxes as tv_tensors.BoundingBoxes of arbitrary format.
        kps: The key points of the respective images.
            The key points will be transformed with the images to obtain the local key point coordinates.
            Default None just means that a placeholder is passed and returned.

    Keyword Args:
        crop_size (ImgShape): The target shape of the image crops.
            Defaults to `DEF_CONF.images.crop_size`.
        transform (tvt.Compose): A torchvision transform given as Compose to get the crops from the original image.
            Defaults to a version of CustomCropResize.
        crop_mode (str): Defines the resize mode in the transform function.
            Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`.
            Default `DEF_CONF.images.mode`.
    """
    if len(imgs) == 0:
        return tvte.Image(torch.empty(0, 3, 1, 1)), None

    if len(imgs) != len(bboxes):
        raise ValueError(f"Expected length of imgs {len(imgs)} and number of bounding boxes {len(bboxes)} to match.")

    transform = kwargs.get(
        "transform",
        tvt.Compose([tvt.ConvertBoundingBoxFormat(format="XYWH"), tvt.ClampBoundingBoxes(), CustomCropResize()]),
    )
    res = transform(
        {
            "images": imgs,
            "box": bboxes,
            "keypoints": kps if kps is not None else torch.zeros((len(imgs), 1, 2), device=imgs[0].device),
            "mode": kwargs.get("crop_mode", DEF_CONF.images.crop_mode),
            "output_size": kwargs.get("crop_size", DEF_CONF.images.crop_size),
        }
    )
    return res["image"], None if kps is None else res["keypoints"]


def extract_crops_and_save(
    img_fps: Union[list[FilePath], FilePaths],
    boxes: tvte.BoundingBoxes,
    new_fps: Union[list[FilePath], FilePaths],
    key_points: torch.Tensor = None,
    **kwargs,
) -> tuple[Image, torch.Tensor]:
    """Given a list of original image paths and a list of target image-crops paths,
    use the given bounding boxes to extract the image content as image crops and save them as new images.

    Does only work if the images have the same size, because otherwise the bounding-boxes would not match anymore.

    Notes:
        It is expected that ``img_fps``, ``new_fps``, and ``boxes`` have the same length.

    Args:
        img_fps: An iterable of absolute paths pointing to the original images.
        boxes: The bounding boxes as tv_tensors.BoundingBoxes of arbitrary format.
        new_fps: An iterable of absolute paths pointing to the image crops.
        key_points (torch.Tensor, optional): Key points of the respective images.
            The key points will be transformed with the images. Default None just means that a placeholder is passed.

    Keyword Args:
        crop_size (ImgShape): The target shape of the image crops.
            Defaults to `DEF_CONF.images.crop_size`.
        transform (tvt.Compose): A torchvision transform given as Compose to get the crops from the original image.
            Defaults to a version of CustomCropResize.
        crop_mode (str): Defines the resize mode in the transform function.
            Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`.
            Default `DEF_CONF.images.mode`.
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

    imgs: Images = load_image_list(filepath=tuple(img_fps), device=device)

    crops, loc_kps = extract_crops_from_images(imgs=imgs, bboxes=boxes, kps=key_points, **kwargs)

    for i, (fp, crop) in enumerate(zip(new_fps, crops.cpu())):
        mkdir_if_missing(os.path.dirname(fp))
        write_jpeg(input=convert_image_dtype(crop, torch.uint8), filename=fp, quality=kwargs.get("quality", 90))
        if key_points is not None:
            torch.save(loc_kps[i].unsqueeze(0), str(fp).replace(".jpg", ".pt"))

    return crops.to(device=device), None if key_points is None else loc_kps


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
