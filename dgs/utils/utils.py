"""
General utility functions.
"""
import os.path
from typing import Union

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.io import write_jpeg
from torchvision.transforms import v2 as tvt

from dgs.utils.files import mkdir_if_missing
from dgs.utils.image import CustomCropResize, load_image
from dgs.utils.types import Device, FilePath, FilePaths, TVImage


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
    key_points: Union[torch.Tensor, None] = None,
    **kwargs,
) -> tuple[TVImage, torch.Tensor]:
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
    if not len(img_fps) == len(new_fps) == boxes.shape[-2]:
        raise ValueError("There has to be an equal amount of image paths, crop paths and boxes.")

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

    imgs: TVImage = load_image(
        filepath=tuple(img_fps),
        device=device,
        requires_grad=False,
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

    for fp, crop in zip(new_fps, crops):
        mkdir_if_missing(os.path.dirname(fp))
        write_jpeg(input=crop, filename=fp, quality=kwargs.get("quality", 90))

    return crops.to(device=device), res["keypoints"]
