"""
Utilities for validating recurring data types.
"""
import os
from collections.abc import Iterable, Sized
from typing import Union

import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException, ValidationException
from dgs.utils.files import is_file, to_abspath
from dgs.utils.types import FilePath, FilePaths, Heatmap, Image, TVImage, Validator

VALIDATIONS: dict[str, Validator] = {
    "optional": (lambda _x, _d: True),
    # types
    "None": (lambda x, _: x is None),
    "not None": (lambda x, _: x is not None),
    "bool": (lambda x, _: isinstance(x, bool)),
    "str": (lambda x, _: isinstance(x, str)),
    "int": (lambda x, _: isinstance(x, int)),
    "float": (lambda x, _: isinstance(x, float)),
    "dict": (lambda x, _: isinstance(x, dict)),
    "list": (lambda x, _: isinstance(x, list)),
    "set": (lambda x, _: isinstance(x, set)),
    "tuple": (lambda x, _: isinstance(x, tuple)),
    "number": (lambda x, _: isinstance(x, int | float)),
    "callable": (lambda x, _: callable(x)),
    "instance": isinstance,  # alias
    "isinstance": isinstance,
    "iterable": (lambda x, _: isinstance(x, Iterable)),
    "sized": (lambda x, _: isinstance(x, Sized)),
    # number
    "gt": (lambda x, d: isinstance(d, int | float) and x > d),
    "gte": (lambda x, d: isinstance(d, int | float) and x >= d),
    "lt": (lambda x, d: isinstance(d, int | float) and x < d),
    "lte": (lambda x, d: isinstance(d, int | float) and x <= d),
    "between": (lambda x, d: isinstance(x, int | float) and isinstance(d, tuple) and len(d) == 2 and d[0] < x < d[1]),
    "within": (lambda x, d: isinstance(x, int | float) and isinstance(d, tuple) and len(d) == 2 and d[0] <= x <= d[1]),
    "outside": (
        lambda x, d: isinstance(x, int | float) and isinstance(d, tuple) and len(d) == 2 and x < d[0] or x > d[1]
    ),
    # lists and other iterables
    "len": (lambda x, d: hasattr(x, "__len__") and len(x) == d),
    "shorter": (lambda x, d: hasattr(x, "__len__") and len(x) <= d),
    "longer": (lambda x, d: hasattr(x, "__len__") and len(x) >= d),
    "in": (lambda x, d: hasattr(d, "__contains__") and x in d),
    "not in": (lambda x, d: hasattr(x, "__contains__") and x not in d),
    "contains": (lambda x, d: hasattr(x, "__contains__") and d in x),
    "not contains": (lambda x, d: hasattr(x, "__contains__") and d not in x),
    "all type": (lambda x, d: hasattr(x, "__len__") and isinstance(d, type) and all(isinstance(xi, d) for xi in x)),
    "all in": (lambda x, d: hasattr(x, "__len__") and hasattr(d, "__contains__") and all(xi in d for xi in x)),
    # string
    "startswith": (lambda x, d: isinstance(x, str) and (isinstance(d, str) or bool(str(d))) and x.startswith(d)),
    "endswith": (lambda x, d: isinstance(x, str) and (isinstance(d, str) or bool(str(d))) and x.endswith(d)),
    # file and folder
    "file exists": (lambda x, _: isinstance(x, FilePath) and os.path.isfile(x)),
    "file exists in project": (lambda x, _: isinstance(x, FilePath) and os.path.isfile(os.path.join(PROJECT_ROOT, x))),
    "file exists in folder": (
        lambda x, f: isinstance(x, FilePath) and isinstance(f, FilePath) and os.path.isfile(os.path.join(f, x))
    ),
    "folder exists": (lambda x, _: isinstance(x, FilePath) and os.path.isdir(x)),
    "folder exists in project": (lambda x, _: isinstance(x, FilePath) and os.path.isdir(os.path.join(PROJECT_ROOT, x))),
    "folder exists in folder": (
        lambda x, f: isinstance(x, FilePath) and isinstance(f, FilePath) and os.path.isdir(os.path.join(f, x))
    ),
    # logical operators, including nested validations
    "eq": (lambda x, d: x == d),
    "neq": (lambda x, d: x != d),
    "not": (lambda x, d: not VALIDATIONS[d[0]](x, d[1])),
    "and": (lambda x, d: all(VALIDATIONS[d[i][0]](x, d[i][1]) for i in range(len(d)))),
    "or": (lambda x, d: any(VALIDATIONS[d[i][0]](x, d[i][1]) for i in range(len(d)))),
    "xor": (lambda x, d: bool(VALIDATIONS[d[0][0]](x, d[0][1])) != bool(VALIDATIONS[d[1][0]](x, d[1][1]))),
}
"""A list of default validations to check values using :meth:`validate_value`."""


def validate_bboxes(
    bboxes: tv_tensors.BoundingBoxes,
    dims: Union[int, None] = 2,
    box_format: Union[tv_tensors.BoundingBoxFormat, None] = None,
) -> tv_tensors.BoundingBoxes:
    """Given a torchvision tensor of bounding boxes,
    validate them and return them as a torchvision-tensor of bounding-boxes.

    Args:
        bboxes: `tv_tensor.BoundingBoxes` object with an arbitrary shape, most likely ``[B x 4]``.
        dims: Number of dimensions bboxes should have.
            Use None to not force any number of dimensions.
            Defaults to two dimensions with the bounding box dimensions as ``[B x 4]``.
        box_format: If present, validates whether the tv_tensors.BoundingBoxFormat matches the one of bboxes.
            Default None, and therefore no validation of the format.

    Returns:
        Bounding boxes as `tv_tensor.BoundingBoxes` object with exactly `dims` dimensions.

    Raises:
        TypeError: If the `bboxes` input is not a Tensor.
        ValueError: If the `bboxes` have the wrong shape or the `bboxes` have the wrong format.
    """
    if not isinstance(bboxes, tv_tensors.BoundingBoxes):
        raise TypeError(f"Bounding boxes should be torch tensor or tv_tensor Bounding Boxes but is {type(bboxes)}")

    if box_format is not None and box_format != bboxes.format:
        raise ValueError(f"Bounding boxes are expected to be in format {box_format} but are in format {bboxes.format}")

    if dims is None:
        return bboxes

    new_bboxes = validate_dimensions(bboxes, dims)

    return tv_tensors.wrap(new_bboxes, like=bboxes)


def validate_dimensions(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Given a tensor, make sure he has the correct number of dimensions.

    Args:
        tensor: Any `torch.tensor` or other object that can be converted to one.
        dims: Number of dimensions the tensor should have.

    Returns:
        A `torch.tensor` with the correct number of dimensions.

    Raises:
        TypeError: If the `tensor` input is not a `torch.tensor` or cannot be cast to one.
        ValueError: If the length of the `tensor` is bigger than `dims` and cannot be unsqueezed.
    """
    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.tensor(tensor)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"The input should be a torch tensor or a type that can be converted to one. "
                f"But `tensor` is {type(tensor)}"
            ) from e

    if tensor.ndim > dims:
        tensor.squeeze_()
        if tensor.ndim > dims:
            raise ValueError(
                f"The length of tensor.shape should be {dims} but shape is {tensor.shape}. "
                f"Unsqueezing did not work."
            )
    while tensor.ndim < dims:
        tensor.unsqueeze_(0)

    return tensor


def validate_filepath(file_paths: Union[FilePath, Iterable[FilePath]]) -> FilePaths:
    """Validate the file path.

    Args:
        file_paths: Path to the file as a string or a file object.

    Returns:
        FilePaths: The validated file path.

    Raises:
        InvalidPathException: If at least one of the paths in `file_paths` does not exist.
    """
    if isinstance(file_paths, (list, tuple)):
        return tuple(validate_filepath(file_path)[0] for file_path in file_paths)
    if not is_file(file_paths):
        raise InvalidPathException(filepath=file_paths)

    return tuple([to_abspath(filepath=file_paths)])


def validate_heatmaps(
    heatmaps: Union[torch.Tensor, Heatmap], dims: Union[int, None] = 4, nof_joints: int = None
) -> Heatmap:
    """Validate a given tensor of heatmaps, whether it has the correct format and shape.

    Args:
        heatmaps: tensor-like object
        dims: Number of dimensions heatmaps should have.
            Use None to not force any number of dimensions.
            Defaults to four dimensions with the heatmap dimensions as ``[B x J x w x h]``.
        nof_joints: The number of joints the heatmap should have.
            Default None does not validate the number of joints at all.

    Returns:
        Heatmap: The validated heatmaps as tensor with the correct type.

    Raises:
        TypeError: If the `heatmaps` input is not a Tensor or cannot be cast to one.
        ValueError: If the `heatmaps` are neither two- nor three-dimensional.
    """
    if not isinstance(heatmaps, (Heatmap, torch.Tensor)):
        raise TypeError(f"heatmaps should be a Heatmap or torch tensor but are {type(heatmaps)}.")

    if nof_joints is not None and (heatmaps.ndim < 3 or heatmaps.shape[-3] != nof_joints):
        raise ValueError(f"The number of joints should be {nof_joints} but is {heatmaps.shape[-2]}.")

    if dims is not None:
        heatmaps = validate_dimensions(heatmaps, dims)

    return tv_tensors.Mask(heatmaps)


def validate_ids(ids: Union[int, torch.Tensor]) -> torch.IntTensor:
    """Validate a given tensor or single integer value.

    Args:
        ids: Arbitrary torch tensor to check.

    Returns:
        torch.IntTensor: Torch integer tensor with one dimension.

    Raises:
        TypeError: If `ids` is not a `torch.IntTensor`.
    """
    if isinstance(ids, int):
        ids = torch.tensor([ids], dtype=torch.int)

    if not isinstance(ids, torch.Tensor) or (isinstance(ids, torch.Tensor) and ids.dtype != torch.int):
        raise TypeError(f"The input should be a torch int tensor but is {type(ids)}")

    ids.squeeze_()

    if ids.ndim == 0:
        ids.unsqueeze_(-1)
    elif ids.ndim != 1:
        raise ValueError(f"IDs should have only one dimension, but shape is {ids.shape}")

    return ids.to(dtype=torch.int32)


def validate_images(images: Union[Image, torch.Tensor], dims: Union[int, None] = 4) -> TVImage:
    """Given one single or multiple images, validate them and return a torchvision-tensor image.

    Args:
        images: torch tensor or tv_tensor.Image object
        dims: Number of dimensions img should have.
            Use None to not force any number of dimensions.
            Defaults to four dimensions with the image dimensions as ``[B x C x H x W]``.

    Returns:
        TVImage: The images as `tv_tensor.Image` object with exactly `dims` dimensions.

    Raises:
        TypeError: If `images` is not a Tensor or cannot be cast to one.
        ValueError: If the dimension of the `images` channels is wrong.
    """
    if not isinstance(images, (torch.Tensor, torch.ByteTensor, torch.FloatTensor, tv_tensors.Image)) or not (
        isinstance(images, torch.Tensor) and images.dtype in [torch.float32, torch.uint8]  # iff tensor, check dtype
    ):
        raise TypeError(f"Image should be torch tensor or tv_tensor Image but is {type(images)}.")

    if dims is not None:
        images = validate_dimensions(images, dims)

    if images.ndim < 3:
        raise ValueError(f"Image should have at least 3 dimensions. Shape: {images.shape}.")

    if images.shape[-3] not in [1, 3, 4]:
        raise ValueError(
            f"Image should either be RGB, RGBA or depth. But a dimensionality {images.shape[-3]} is unknown."
        )

    return tv_tensors.Image(images)


def validate_key_points(
    key_points: torch.Tensor, dims: Union[int, None] = 3, nof_joints: int = None, joint_dim: int = None
) -> torch.Tensor:
    """Given a tensor of key points, validate them and return them as torch tensor of the correct shape.

    Args:
        key_points: One `torch.tensor` or any similarly structured data.
        dims: The number of dimensions `key_points` should have.
            Use `None` to not force any number of dimensions.
            Defaults to three dimensions with the key point dimensions as ``[B x J x 2|3]``.
        nof_joints: The number of joints `key_points` should have.
            Default `None` does not validate the number of joints at all.
        joint_dim: The dimensionality the joint dimension should have.
            Default `None` does not validate the dimensionality additionally to being two or three.

    Returns:
        torch.Tensor: The key points as a single `torch.tensor` with exactly the requested number of dimensions like
        ``[... x nof_joints x joint_dim]``.

    Raises:
        TypeError: If the key point input is not a Tensor.
        ValueError: If the key points or joints have the wrong dimensionality.
    """
    if not isinstance(key_points, torch.Tensor):
        raise TypeError(f"Key points should be torch tensor but is {type(key_points)}.")

    if joint_dim is None and not 2 <= key_points.shape[-1] <= 3:
        raise ValueError(
            f"By default, the key points should be two- or three-dimensional, "
            f"but they have a shape of {key_points.shape[-1]}"
        )

    if joint_dim is not None and key_points.shape[-1] != joint_dim:
        raise ValueError(f"The dimensionality of the joints should be {joint_dim} but is {key_points.shape[-1]}.")

    if nof_joints is not None and key_points.shape[-2] != nof_joints:
        raise ValueError(f"The number of joints should be {nof_joints} but is {key_points.shape[-2]}.")

    if dims is not None:
        key_points = validate_dimensions(key_points, dims)

    return key_points


def validate_value(value: any, data: any, validation: str) -> bool:
    """Check a single value against a given predefined validation, possibly given some additional data.

    Args:
        value: The value to validate.
        data: Possibly additional data needed for validation, is ignored otherwise.
        validation: The name of the validation to perform.

    Returns:
        bool: Whether the given `value` is valid given the `validation` and possibly more `data`.

    Raises:
        KeyError: If the given `validation` does not exist.
    """
    if validation not in VALIDATIONS:
        raise KeyError(f"Validation {validation} does not exist.")
    try:
        return VALIDATIONS[validation](value, data)
    except Exception as e:
        raise ValidationException(
            f"Could not validate value {value} with data {data} for validation {validation}"
        ) from e
