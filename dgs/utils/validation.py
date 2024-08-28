"""
Utilities for validating recurring data types.
"""

import os
from collections.abc import Iterable, Sized
from typing import Union

import torch as t
from torchvision import tv_tensors as tvte

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException, ValidationException
from dgs.utils.files import is_dir, is_file, is_project_dir, mkdir_if_missing, to_abspath
from dgs.utils.types import FilePath, FilePaths, Heatmap, Image, Images, Validator

VALIDATIONS: dict[str, Validator] = {
    "optional": (lambda _x, _d: True),
    # types
    "None": (lambda x, _: x is None),
    "not None": (lambda x, _: x is not None),
    "number": (lambda x, _: isinstance(x, (int, float))),
    "callable": (lambda x, _: callable(x)),
    "iterable": (lambda x, _: isinstance(x, Iterable)),
    "sized": (lambda x, _: isinstance(x, Sized)),
    "instance": isinstance,
    "isinstance": isinstance,
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
    "shorter": (lambda x, d: hasattr(x, "__len__") and len(x) < d),
    "longer": (lambda x, d: hasattr(x, "__len__") and len(x) > d),
    "shorter eq": (lambda x, d: hasattr(x, "__len__") and len(x) <= d),
    "longer eq": (lambda x, d: hasattr(x, "__len__") and len(x) >= d),
    "in": (lambda x, d: hasattr(d, "__contains__") and x in d),
    "not in": (lambda x, d: hasattr(x, "__contains__") and x not in d),
    "contains": (lambda x, d: hasattr(x, "__contains__") and d in x),
    "not contains": (lambda x, d: hasattr(x, "__contains__") and d not in x),
    # string
    "startswith": (lambda x, d: isinstance(x, str) and (isinstance(d, str) or bool(str(d))) and x.startswith(d)),
    "endswith": (lambda x, d: isinstance(x, str) and (isinstance(d, str) or bool(str(d))) and x.endswith(d)),
    # file and folder
    "file exists": (
        lambda x, _: isinstance(x, str)
        and (VALIDATIONS["file exists absolute"](x, _) or VALIDATIONS["file exists in project"](x, _))
    ),
    "file exists absolute": (lambda x, _: isinstance(x, str) and os.path.isfile(x)),
    "file exists in project": (lambda x, _: isinstance(x, str) and os.path.isfile(os.path.join(PROJECT_ROOT, x))),
    "file exists in folder": (
        lambda x, f: isinstance(x, str) and isinstance(f, str) and os.path.isfile(os.path.join(f, x))
    ),
    "folder exists": (
        lambda x, b: isinstance(x, str)
        and (VALIDATIONS["folder exists absolute"](x, b) or VALIDATIONS["folder exists in project"](x, b))
    ),
    "folder exists absolute": (
        lambda x, b: isinstance(x, str) and (is_dir(x) if not b else mkdir_if_missing(x) and True)
    ),
    "folder exists in project": (
        lambda x, b: isinstance(x, str)
        and (is_project_dir(x) if not b else is_project_dir(x) or mkdir_if_missing(x) and True)
    ),
    "folder exists in folder": (
        lambda x, f: isinstance(x, str) and isinstance(f, str) and os.path.isdir(os.path.join(f, x))
    ),
    # logical operators, including nested validations
    "eq": (lambda x, d: x == d),
    "neq": (lambda x, d: x != d),
    "not": (lambda x, d: not VALIDATIONS["all"](x, d)),
    "forall": (
        lambda x, data: (
            VALIDATIONS["iterable"](x, None)
            and (
                all(VALIDATIONS[data[0]](x_i, data[1]) for x_i in x)
                if isinstance(data, tuple)
                else (
                    all(VALIDATIONS[data](x_i, None) for x_i in x)
                    if isinstance(data, str)
                    else (
                        all(isinstance(x_i, data) for x_i in x)
                        if isinstance(data, type)
                        else (
                            all(VALIDATIONS["all"](x_i, d_i) for d_i in data for x_i in x)
                            if isinstance(data, list)
                            else False
                        )
                    )
                )
            )
        )
    ),
    "all": (
        lambda x, data: (
            (len(data) == 2 and VALIDATIONS[data[0]](x, data[1]))
            if isinstance(data, tuple)
            else (
                VALIDATIONS[data](x, None)
                if isinstance(data, str)
                else (
                    isinstance(x, data)
                    if isinstance(data, type)
                    else (
                        (len(data) and all(VALIDATIONS["all"](x, sub_item) for sub_item in data))
                        if isinstance(data, list)
                        else False
                    )
                )
            )
        )
    ),
    "any": (
        lambda x, data: (
            (len(data) == 2 and VALIDATIONS[data[0]](x, data[1]))
            if isinstance(data, tuple)
            else (
                VALIDATIONS[data](x, None)
                if isinstance(data, str)
                else (
                    isinstance(x, data)
                    if isinstance(data, type)
                    else (
                        (len(data) and any(VALIDATIONS["any"](x, sub_item) for sub_item in data))
                        if isinstance(data, list)
                        else False
                    )
                )
            )
        )
    ),
    "xor": (
        lambda x, d: isinstance(d, list)
        and len(d) == 2
        and bool(VALIDATIONS["all"](x, d[0])) != bool(VALIDATIONS["all"](x, d[1]))
    ),
}


def validate_bboxes(
    bboxes: tvte.BoundingBoxes,
    length: int = None,
    dims: Union[int, None] = 2,
    box_format: Union[tvte.BoundingBoxFormat, None] = None,
) -> tvte.BoundingBoxes:
    """Given a torchvision tensor of bounding boxes,
    validate them and return them as a torchvision-tensor of bounding-boxes.

    Args:
        bboxes: `tv_tensor.BoundingBoxes` object with an arbitrary shape, most likely ``[B x 4]``.
        length: The number of items or batch-size the tensor should have.
            Default `None` does not validate the length.
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
    if not isinstance(bboxes, tvte.BoundingBoxes):
        raise TypeError(f"Bounding boxes should be torch tensor or tv_tensor Bounding Boxes but is {type(bboxes)}")

    if box_format is not None and box_format != bboxes.format:
        raise ValueError(f"Bounding boxes are expected to be in format {box_format} but are in format {bboxes.format}")

    saved = bboxes

    if dims is not None:
        bboxes = validate_dimensions(tensor=bboxes, dims=dims, length=length)
    elif length is not None and len(bboxes) != length:
        raise ValidationException(f"Bounding box length is expected to be {length} but got {len(bboxes)}")

    return tvte.wrap(bboxes, like=saved)


def validate_dimensions(tensor: t.Tensor, dims: int, *_, length: int = None) -> t.Tensor:
    """Given a tensor, make sure he has the correct number of dimensions.

    Args:
        tensor: Any `torch.tensor` or other object that can be converted to one.
        dims: Number of dimensions the tensor should have.
        length: The number of items or batch-size the tensor should have.
            Default `None` does not validate the length.

    Returns:
        A `torch.tensor` with the correct number of dimensions.

    Raises:
        TypeError: If the `tensor` input is not a `torch.tensor` or cannot be cast to one.
        ValueError: If the length of the `tensor` is bigger than `dims` and cannot be unsqueezed.
    """
    if not isinstance(tensor, t.Tensor):
        try:
            tensor = t.tensor(tensor)
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

    if length is not None and length != len(tensor):
        raise ValidationException(f"length is expected to be {length} but got {len(tensor)}")

    return tensor


def validate_filepath(file_paths: Union[FilePath, Iterable[FilePath], FilePaths], length: int = None) -> FilePaths:
    """Validate the file path.

    Args:
        file_paths: Path to the file as a string or a file object.
        length: The length a :class:`FilePaths` object should have.
            Except for a length of 1 not applicable for :class:`FilePath`.

    Returns:
        FilePaths: The validated file path.

    Raises:
        InvalidPathException: If at least one of the paths in `file_paths` does not exist.
    """
    if isinstance(file_paths, (list, tuple)):
        if length is not None and len(file_paths) != length:
            raise ValidationException(f"Expected {length} paths but got {len(file_paths)}.")
        return tuple(validate_filepath(file_path)[0] for file_path in file_paths)

    if isinstance(file_paths, str) and length is not None and length != 1:
        raise ValidationException(f"Expected {length} paths but got a single path {file_paths}.")

    file_paths = str(file_paths)
    if not is_file(file_paths):
        raise InvalidPathException(filepath=file_paths)

    return tuple([to_abspath(filepath=file_paths)])


def validate_heatmaps(
    heatmaps: Union[t.Tensor, Heatmap], length: int = None, dims: Union[int, None] = 4, nof_joints: int = None
) -> Heatmap:
    """Validate a given tensor of heatmaps, whether it has the correct format and shape.

    Args:
        heatmaps: tensor-like object
        length: The number of items or batch-size the tensor should have.
            Default `None` does not validate the length.
        dims: Number of dimensions heatmaps should have.
            Use None to not force any number of dimensions.
            Defaults to four dimensions with the heatmap dimensions as ``[B x J x w x h]``.
        nof_joints: The number of joints the heatmap should have (``J``).
            Default None does not validate the number of joints at all.

    Returns:
        Heatmap: The validated heatmaps as tensor with the correct type.

    Raises:
        TypeError: If the `heatmaps` input is not a Tensor or cannot be cast to one.
        ValueError: If the `heatmaps` are neither two- nor three-dimensional.
    """
    if not isinstance(heatmaps, (Heatmap, t.Tensor)):
        raise TypeError(f"heatmaps should be a Heatmap or torch tensor but are {type(heatmaps)}.")

    if nof_joints is not None and (heatmaps.ndim < 3 or heatmaps.shape[-3] != nof_joints):
        raise ValueError(f"The number of joints should be {nof_joints} but is {heatmaps.shape[-2]}.")

    if dims is not None:
        heatmaps = validate_dimensions(tensor=heatmaps, dims=dims, length=length)
    elif length is not None and len(heatmaps) != length:
        raise ValidationException(f"Heatmap length is expected to be {length} but got {len(heatmaps)}")

    return tvte.Mask(heatmaps)


def validate_ids(ids: Union[int, t.Tensor], length: int = None) -> t.Tensor:
    """Validate a given tensor or single integer value.

    Args:
        ids: Arbitrary torch tensor to check.
        length: The number of items or batch-size the tensor should have.
            Default `None` does not validate the length.

    Returns:
        torch.Tensor: Torch integer tensor with one dimension.

    Raises:
        TypeError: If `ids` is not a `torch.Tensor`.
    """
    if isinstance(ids, int):
        ids = t.tensor([ids], dtype=t.int)

    if not isinstance(ids, t.Tensor) or ids.is_floating_point() or ids.is_complex():
        raise TypeError(f"The input should be an integer or an whole numbered torch.Tensor but is {type(ids)}")

    ids.squeeze_()

    if ids.ndim == 0:
        ids.unsqueeze_(-1)
    elif ids.ndim != 1:
        raise ValueError(f"IDs should have only one dimension, but shape is {ids.shape}")

    if length is not None and ids.size(0) != length:
        raise ValidationException(f"IDs length is expected to be {length} but got {ids.size(0)}")

    return ids.long()


def validate_image(images: Union[Image, t.Tensor], length: int = None, dims: Union[int, None] = 4) -> Image:
    """Given one single image or a stacked batch images, validate them and return a torchvision-tensor image.

    Args:
        images: torch tensor or tv_tensor.Image object
        length: The number of items or batch-size the tensor should have.
            Default `None` does not validate the length.
        dims: Number of dimensions img should have.
            Use None to not force any number of dimensions.
            Defaults to four dimensions with the image dimensions as ``[B x C x H x W]``.

    Returns:
        Image: The images as `tv_tensor.Image` object with exactly `dims` dimensions.

    Raises:
        TypeError: If `images` is not a Tensor or cannot be cast to one.
        ValueError: If the dimension of the `images` channels is wrong.
    """
    if not isinstance(images, (t.Tensor, t.Tensor, t.Tensor, tvte.Image)) or not (
        isinstance(images, t.Tensor) and images.dtype in [t.float32, t.uint8]  # iff tensor, check dtype
    ):
        raise TypeError(f"Image should be torch tensor or tv_tensor Image but is {type(images)}.")

    if dims is not None:
        images = validate_dimensions(tensor=images, dims=dims, length=length)
    elif length is not None and len(images) != length:
        raise ValidationException(f"Image length is expected to be {length} but got {len(images)}")

    if images.ndim < 3:
        raise ValueError(f"Image should have at least 3 dimensions. Shape: {images.shape}.")

    if images.shape[-3] not in [1, 3, 4]:
        raise ValueError(
            f"Image should either be RGB, RGBA or depth. But a dimensionality {images.shape[-3]} is unknown."
        )

    return tvte.Image(images)


def validate_images(images: list[Union[Image, t.Tensor]]) -> Images:
    """Given one single or multiple images, validate them and return a torchvision-tensor image.

    Args:
        images: A list containing :class:`~torch.Tensor` or :class:`.tv_tensor.Image` objects.

    Returns:
        The images as a list containing :class:`.tv_tensor.Image` objects, each with exactly 4 dimensions.

    Raises:
        TypeError: If `images` is not a list.
    """
    if not isinstance(images, (list, tuple)):
        raise TypeError(f"Expected images to be a list, got {type(images)}.")

    return [validate_image(img, length=1, dims=4) for img in images]


def validate_key_points(
    key_points: t.Tensor,
    length: int = None,
    dims: Union[int, None] = 3,
    nof_joints: int = None,
    joint_dim: int = None,
) -> t.Tensor:
    """Given a tensor of key points, validate them and return them as torch tensor of the correct shape.

    Args:
        key_points: One `torch.tensor` or any similarly structured data.
        length: The number of items or batch-size the tensor should have.
            Default `None` does not validate the length.
        dims: The number of dimensions `key_points` should have.
            Use `None` to not force any number of dimensions.
            Defaults to three dimensions with the key point dimensions as ``[B x J x 2|3]``.
        nof_joints: The number of joints ``key_points`` should have (``J``).
            Default `None` does not validate the number of joints at all.
        joint_dim: The dimensionality the joint dimension should have (``2|3``).
            Default `None` does not validate the dimensionality additionally to being two or three.

    Returns:
        torch.Tensor: The key points as a single `torch.tensor` with exactly the requested number of dimensions like
        ``[... x nof_joints x joint_dim]``.

    Raises:
        TypeError: If the key point input is not a Tensor.
        ValueError: If the key points or joints have the wrong dimensionality.
    """
    if not isinstance(key_points, t.Tensor):
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
        key_points = validate_dimensions(tensor=key_points, dims=dims, length=length)
    elif length is not None and len(key_points) != length:
        raise ValidationException(f"Key-point length is expected to be {length} but got {len(key_points)}")

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
    if isinstance(validation, type):
        return isinstance(value, validation)
    if validation not in VALIDATIONS:
        raise KeyError(f"Validation '{validation}' does not exist.")
    try:
        return VALIDATIONS[validation](value, data)
    except Exception as e:
        raise ValidationException(
            f"Could not validate value {value} with data {data} for validation {validation}"
        ) from e
