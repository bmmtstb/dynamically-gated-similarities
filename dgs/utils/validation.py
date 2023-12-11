"""
Utilities for validating recurring data types.
"""
from typing import Union

import torch
from torchvision import tv_tensors

from dgs.utils.exceptions import InvalidPathException
from dgs.utils.files import is_file, project_to_abspath
from dgs.utils.types import FilePath, Image, TVImage


def validate_bboxes(
    bboxes: tv_tensors.BoundingBoxes,
    dims: Union[int, None] = 2,
    box_format: Union[tv_tensors.BoundingBoxFormat, None] = None,
) -> tv_tensors.BoundingBoxes:
    """Given a tensor of bounding boxes, validate them and return them as torchvision-tensor bounding-boxes.

    Args:
        bboxes: tv_tensor.BoundingBoxes object
        dims: Number of dimensions bboxes should have.
            Use None to not force any number of dimensions.
            Defaults to two dimensions with the bounding box dimensions as ``[B x 4]``.
        box_format: If present, validates whether the tv_tensors.BoundingBoxFormat matches the one of bboxes.
            Default None, and therefore no validation of the format.

    Returns:
        bounding boxes as tv_tensor.BoundingBoxes object with exactly dims dimensions.

    Raises:
        TypeError
            If the bboxes input is not a Tensor.
        ValueError
            If the bboxes have the wrong shape or the bboxes have the wrong format.
    """
    if not isinstance(bboxes, tv_tensors.BoundingBoxes):
        raise TypeError(f"Bounding boxes should be torch tensor or tv_tensor Bounding Boxes but is {type(bboxes)}")

    if bboxes.shape[-1] != 4:
        raise ValueError(f"Bounding boxes should have a size of four in the last dimension, but is {bboxes.shape[-1]}")

    if box_format is not None and box_format != bboxes.format:
        raise ValueError(f"Bounding boxes are expected to be in format {box_format} but are in format {bboxes.format}")

    if dims is None:
        return bboxes

    new_bboxes = validate_dimensions(bboxes, dims)

    return tv_tensors.wrap(new_bboxes, like=bboxes)


def validate_dimensions(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Given a tensor, make sure he has the correct number of dimensions.

    Args:
        tensor: any torch tensor
        dims: Number of dimensions the tensor should have.

    Returns:
        The tensor with the correct number of dimensions.

    Raises:
        TypeError
            If the key point input is not a torch tensor.
        ValueError
            If the length of the key points is bigger than 'dims' and cannot be unsqueezed.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"The input should be a torch tensor but is {type(tensor)}")

    if len(tensor.shape) > dims:
        tensor.squeeze_()
        if len(tensor.shape) > dims:
            raise ValueError(
                f"The length of tensor.shape should be {dims} but shape is {tensor.shape}. "
                f"Unsqueezing did not work."
            )
    while len(tensor.shape) < dims:
        tensor.unsqueeze_(0)

    return tensor


def validate_images(images: Image, dims: Union[int, None] = 4) -> TVImage:
    """Given one single or multiple images, validate them and return a torchvision-tensor image.

    Args:
        images: torch tensor or tv_tensor.Image object
        dims: Number of dimensions img should have.
            Use None to not force any number of dimensions.
            Defaults to four dimensions with the image dimensions as ``[B x C x H x W]``.


    Returns:
        image as tv_tensor.Image object with exactly dims dimensions.

    Raises:
        TypeError
            If the image is not a Tensor.
        ValueError
            If the image channel has the wrong dimensionality.
    """
    if not isinstance(images, (torch.ByteTensor, torch.FloatTensor, tv_tensors.Image)):
        raise TypeError(f"Image should be torch tensor or tv_tensor Image but is {type(images)}")

    if dims is not None:
        images = validate_dimensions(images, dims)

    if len(images.shape) < 3:
        raise ValueError(f"Image should have at least 3 dimensions. Shape: {images.shape}")

    if images.shape[-3] not in [1, 3, 4]:
        raise ValueError(
            f"Image should either be RGB, RGBA or depth. But a dimensionality {images.shape[-3]} is unknown."
        )

    return tv_tensors.Image(images)


def validate_filepath(file_path: FilePath) -> FilePath:
    """Validate the file path.

    Args:
        file_path: Path to the file as a string or a file object.

    Returns:
        The validated file path.

    Raises:
        InvalidPathException
            If the file path does not exist.
    """
    if not is_file(file_path):
        raise InvalidPathException(filepath=file_path)

    return project_to_abspath(filepath=file_path)


def validate_key_points(
    key_points: torch.Tensor, dims: Union[int, None] = 3, nof_joints: int = None, joint_dim: int = None
) -> tv_tensors.Mask:
    """Given a tensor of key points, validate them and return them as torchvision tensor mask.

    Args:
        key_points: torch tensor or tv_tensor.Mask object
        dims: Number of dimensions key_points should have.
            Use None to not force any number of dimensions.
            Defaults to three dimensions with the key point dimensions as ``[B x J x 2|3]``.
        nof_joints: The number of joints the key points should have.
            Default None does not validate the number of joints at all.
        joint_dim: The dimensionality the joints should have.
            Default None does not validate the dimensionality additionally to being two or three.

    Returns:
        key points as tv_tensor.Mask object with exactly dims dimensions.

    Raises:
        TypeError
            If the key point input is not a Tensor.
        ValueError
            If the key points are neither two- nor three-dimensional.
    """
    if not isinstance(key_points, (torch.Tensor, tv_tensors.Mask)):
        raise TypeError(f"Key points should be torch tensor or tv_tensor Mask but is {type(key_points)}")

    if not 2 <= key_points.shape[-1] <= 3:
        raise ValueError(
            f"The key points should be two- or three-dimensional, but they have a shape of {key_points.shape[-1]}"
        )

    if joint_dim is not None and key_points.shape[-1] != joint_dim:
        raise ValueError(f"The dimensionality of the joints should be {joint_dim} but is {key_points.shape[-1]}")

    if nof_joints is not None and key_points.shape[-2] != nof_joints:
        raise ValueError(f"The number of joints should be {nof_joints} but is {key_points.shape[-2]}")

    if dims is not None:
        key_points = validate_dimensions(key_points, dims)

    return tv_tensors.Mask(key_points)
