"""
.. _image_util_page:

Utility for handling images in pytorch.

Loading, saving, manipulating of RGB-images.

Within pytorch an image is a Byte-, Uint8-, or Float-Tensor with a shape of ``[C x h x w]``.
Within torchvision an image is a tv_tensor.Image object with the same shape.
A Batch of torch images therefore has a shape of ``[B x C x h x w]``.
Within pytorch and torchvision, the images have channels in order of RGB.
The size / shape of an image is given as tuple (and sometimes list) of ints in the form of (h, w).

RGB Images in cv2 have a shape of ``[h x w x C]`` and the channels are in order GBR.
Grayscale Images in cv2 have a shape of ``[h x w]``.
"""

from typing import Iterable, Union

import torch
import torchvision.transforms.v2 as tvt
from torch.nn import Module as Torch_NN_Module
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image, read_video
from torchvision.transforms.v2.functional import (
    center_crop as tvt_center_crop,
    crop as tvt_crop,
    pad as tvt_pad,
    resize as tvt_resize,
)

from dgs.utils.exceptions import ValidationException
from dgs.utils.files import to_abspath
from dgs.utils.types import FilePath, FilePaths, Image, ImgShape, Video
from dgs.utils.validation import validate_bboxes, validate_filepath, validate_key_points


def load_image(
    filepath: Union[FilePath, FilePaths],
    force_reshape: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu",
    read_mode: ImageReadMode = ImageReadMode.RGB,
    **kwargs,
) -> Image:
    """Load an image or multiple images given a single or multiple filepaths.

    Notes:
        To be able to return a single torch tensor while loading multiple images,
        make sure that either the images have the same shape, or ``force_reshape`` is set to ``True``.

    Notes:
        To be able to compute gradients, the dtype has to be ``torch.float32``.
        Therefore, the ``dtype`` and ``requires_grad`` of the model are partially correlated.

    Args:
        filepath: Single string or list of absolute or local filepaths to the image.
        force_reshape: Whether to reshape the image(s) to a target shape.
            The mode and size can be specified in the kwargs.
            Default False.
        dtype: The dtype of the image, most likely one of uint8, byte, or float32.
            Default torch.float32.
        device: Device the image should be on.
            Default "cpu"
        read_mode: Which ImageReadMode to use while loading the images.
            Default 'ImageReadMode.RGB'.

    Keyword Args:
        mode: If ``force_reshape`` is true, defines the resize mode, has to be in the modes of
            :class:`~dgs.utils.image.CustomToAspect`. Default "zero-pad".
        output_size: If ``force_reshape`` is true, defines the height and width of the returned images.
            Default (256, 256).

    Examples:
        >>> img = load_image("./tests/test_data/866-200x300.jpg")
        >>> print(img.shape)
        torch.Size([1, 3, 300, 200])

        >>> multiple_images = ["./tests/test_data/866-200x300.jpg", "./tests/test_data/866-1000x1000.jpg"]
        >>> imgs = load_image(multiple_images)
        Traceback (most recent call last):
            ...
        RuntimeError: All images should have the same shape.

        >>> imgs = load_image(multiple_images, force_reshape=True, output_size=(300, 300))
        >>> print(imgs.shape)
        torch.Size([2, 3, 300, 300])

    Raises:
        RuntimeError: If images have different shapes but ``force_reshape`` is ``False``.

    Returns:
        Torch uint8 / byte tensor with its original shape of ``[B x C x H x W]`` if force_reshape is false,
        otherwise the returned shape depends on the ``output_size``.
        The returned image will always have four dimensions.
    """
    paths: FilePaths = validate_filepath(filepath)

    # load images
    images = [read_image(path, mode=read_mode).to(device=device) for path in paths]

    transform_dtype = tvt.ToDtype({tv_tensors.Image: dtype, "others": None}, scale=True)
    # if multiple images are loaded, reshape them to a given output_size
    if force_reshape:
        transform = tvt.Compose([CustomToAspect(), CustomResize(), transform_dtype])
        new_images: list[Image] = []
        mode: str = kwargs.pop("mode", "zero-pad")
        output_size: ImgShape = kwargs.pop("output_size", (512, 512))

        for img in images:
            data = {
                "image": tv_tensors.Image(img.detach().clone()),
                "box": tv_tensors.BoundingBoxes(torch.zeros((1, 4)), format="XYWH", canvas_size=(1, 1)),
                "keypoints": torch.zeros((1, 1, 2)),
                "mode": mode,
                "output_size": output_size,
                **kwargs,
            }
            new_images.append(transform(data)["image"])
        images = new_images

    if not all(img.shape[-3:] == images[0].shape[-3:] for img in images):
        raise ValueError(f"All images should have the same shape, but shapes are: {[img.shape for img in images]}")

    images = tv_tensors.Image(torch.stack(images), device=device)

    return transform_dtype(images)


def load_image_list(
    filepath: Union[FilePath, FilePaths],
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu",
    read_mode: ImageReadMode = ImageReadMode.RGB,
) -> list[Image]:
    """Load multiple images with possibly different sizes as a list of tv_tensor images.

    Args:
        filepath: Single string or list of absolute or local filepaths to the image.
        dtype: The dtype of the image, most likely one of uint8, byte, or float32.
            Default torch.float32.
        device: Device the image should be on.
            Default "cpu"
        read_mode: Which ImageReadMode to use while loading the images.
            Default 'ImageReadMode.RGB'.

    Returns:
        A list of tv_tensor images with the provided dtype on the device.
    """
    paths: FilePaths = validate_filepath(filepath)
    transform_dtype = tvt.ToDtype(dtype, scale=True)

    return [
        tv_tensors.Image(transform_dtype(read_image(path, mode=read_mode)), dtype=dtype).to(device=device)
        for path in paths
    ]


def load_video(filepath: FilePath, **kwargs) -> Video:  # pragma: no cover
    """Load a video from a given filepath.

    Returns:
        A batch of torch uint8 / byte images with their original shape of ``[T x C x H x W]``.
        With T being the number of frames in the video.
    """
    fp: FilePath = to_abspath(filepath)

    dtype = kwargs.get("dtype", torch.uint8)
    device = kwargs.get("device", "cpu")

    # read video, save frames and discard audio
    frames, *_ = read_video(fp, output_format="TCHW", pts_unit="sec")

    return tv_tensors.Video(frames, dtype=dtype, device=device)


def compute_padding(old_w: int, old_h: int, target_aspect: float) -> list[int]:
    """Given the width and height of an old and image,
    compute the size of a padding around the old image such that the aspect ratio matches a target.

    Args:
        old_w: Width of the old image
        old_h: Height of the old image
        target_aspect: Aspect the new image should have

    Returns:
        A list of integers as paddings for the left, top, right, and bottom side respectively.
    """
    old_aspect: float = old_w / old_h

    if old_aspect == target_aspect:
        return [0, 0, 0, 0]

    height_padding = int(old_w // target_aspect - old_h)
    width_padding = int(target_aspect * old_h - old_w)

    if height_padding > 0 > width_padding:
        # +1 pixel on the bottom if new shape is odd
        return [0, height_padding // 2, 0, height_padding // 2 + (height_padding % 2)]
    if height_padding < 0 < width_padding:
        # +1 pixel on the right if new shape is odd
        return [width_padding // 2, 0, width_padding // 2 + (width_padding % 2), 0]
    raise ArithmeticError("During computing the sizes for padding, something unexpected happened.")  # pragma: no cover


class CustomTransformValidator:
    """All the custom transforms need to extract and validate values from args and kwargs.
    The values are always the same, therefore, it is possible to create an external validation function.
    """

    # will only have validation functions, and pylint does not like that.
    # pylint: disable=too-few-public-methods

    validators: dict[str, callable] = {
        "image": "_validate_image",
        "box": "_validate_bboxes",
        "keypoints": "_validate_key_points",
        "mode": "_validate_mode",
        "output_size": "_validate_output_size",
    }

    def _validate_inputs(self, *args, necessary_keys: list[str] = None, **kwargs) -> tuple:
        """Validate the inputs of the forward call.

        It can be specified which values have to be validated.

        Args:
            args: Expects the first value of args to be a structured dict containing all the values.
            necessary_keys: A list of strings containing the names of the required keys in `args`.
                Defaults to None.

        Returns:
            Returns the values in the order they appear in `necessary_keys`.

            Default: A tuple containing the values for `(image, box, keypoints, mode, output_size)`.

        Raises:
            TypeError:
            KeyError: If a key from `necessary_keys` is not in `args`.
            ValueError:
        """
        if len(args) != 1 or not isinstance(args[0], dict):
            raise TypeError(f"Invalid args, expected one dict, but got {args}")

        structured_dict: dict[str, any] = args[0]
        structured_dict.update(kwargs)

        if necessary_keys is None:
            necessary_keys = ["image", "box", "keypoints", "output_size", "mode"]

        return_values = []

        for key in necessary_keys:
            if key not in structured_dict:
                raise KeyError(f"Key: {key} was requested but is not in structured_dict.")

            new_value = structured_dict.pop(key)

            if key not in self.validators or not hasattr(self, self.validators[key]):
                raise ValidationException(f"Key: {key} is not in self.validators or there is no validation set.")

            getattr(self, self.validators[key])(new_value, structured_dict)
            return_values.append(new_value)

        # # shape of bboxes and coordinates are dependent
        # if bboxes is not ... and coordinates is not ... and bboxes.shape[-2] != coordinates.shape[-3]:
        #     raise ValueError(
        #         f"Bounding boxes and coordinates have mismatching dimension or shape in batch. "
        #         f"The shapes are - bboxes: {bboxes.shape}, coords: {coordinates.shape}"
        #     )

        return_values.append(structured_dict)

        return tuple(return_values)

    @staticmethod
    def _validate_bboxes(bboxes: tv_tensors.BoundingBoxes, *_args):
        """Validate the bounding boxes"""
        if not isinstance(bboxes, tv_tensors.BoundingBoxes):
            raise TypeError(f"Bounding boxes should be a tv_tensors.BoundingBoxes object but is {type(bboxes)}")
        if bboxes.format != tv_tensors.BoundingBoxFormat.XYWH:
            raise ValueError(f"Bounding boxes should be in XYWH format, but are in {bboxes.format}")

    @staticmethod
    def _validate_key_points(kp: torch.Tensor, *_args):
        """Validate the key points."""
        if not isinstance(kp, torch.Tensor):
            raise TypeError(f"key points should be a torch Tensor object but is {type(kp)}")
        if kp.ndim != 3:
            raise ValueError(f"key points should have three dimensions, but shape is: {kp.shape}")

    @staticmethod
    def _validate_mode(mode: str, structured_dict: dict[str, any], *_args):
        """Validate the mode."""
        if mode not in CustomToAspect.modes:
            raise KeyError(f"Mode: {mode} is not a valid mode.")
        if mode == "fill-pad" and ("fill" not in structured_dict or structured_dict["fill"] is None):
            raise KeyError("In fill-pad mode, fill should be a value of the structured dict and should not be None.")

    @staticmethod
    def _validate_image(img: tv_tensors.Image, *_args):
        """Validate the image."""
        if not isinstance(img, tv_tensors.Image):
            raise TypeError(f"image should be a tv_tensors.Image object but is {type(img)}")

    @staticmethod
    def _validate_output_size(out_s: ImgShape, *_args):
        """Validate the output size."""
        if not isinstance(out_s, (Iterable, tuple)) or len(out_s) != 2:
            raise TypeError(f"output_size is expected to be iterable or tuple of length 2, but is {out_s}")
        if any(not isinstance(v, int) or v <= 0 for v in out_s):
            raise ValueError("output_size should be two positive integers.")


class CustomToAspect(Torch_NN_Module, CustomTransformValidator):
    """Custom torchvision Transform that modifies the image, bboxes, and coordinates simultaneously to match a target
    aspect ratio.

    Notes:
        It is expected that Resize() is called after this transform,
        to not only match the aspect ratio but also the overall size.

        This transforms' default mode is zero-padding.


    The following modes are available for resizing:

    distort
        Skips CustomToAspect entirely and therefore does not change the original aspect ratio at all.
        This will result in a distorted image when using Resize(),
        iff the aspect ratios of the old and new shape aren't close.

    edge-pad
        Uses Pad() to extend the image to the correct aspect ratio.
        The value used for padding_mode of Pad() is `edge`.

    inside-crop
        Uses the target aspect ratio to extract a sub-image out of the original.
        Basically is a center crop with one dimension being as large as possible while maintaining the aspect ratio.

    outside-crop
        Is only available for the CustomCrop() model, but will be passed through.
        Instead of cropping at the exact bounding box, match the aspect ratio by widening one of the dimensions

    fill-pad
        Uses Pad() to extend the image to the correct aspect ratio.
        The value used for padding_mode of Pad() is `constant` and the fill value has to be provided within the kwargs.

    mean-pad
        Uses Pad() to extend the image to the correct aspect ratio.
        The value used for padding_mode of Pad() is `constant` with a fill value as the RGB mean of the image.

    reflect-pad
        Uses Pad() to extend the image to the correct aspect ratio.
        The value used for padding_mode of Pad() is `reflect`.

    symmetric-pad
        Uses Pad() to extend the image to the correct aspect ratio.
        The value used for padding_mode of Pad() is `symmetric`.

    zero-pad
        Uses Pad() to extend the image to the correct aspect ratio.
        The value used for padding_mode of Pad() is `constant` with a value of zero.
    """

    # pylint: disable=too-many-arguments

    modes: list[str] = [
        "distort",
        "edge-pad",
        "inside-crop",
        "fill-pad",
        "mean-pad",
        "outside-crop",
        "reflect-pad",
        "symmetric-pad",
        "zero-pad",
    ]

    H: int
    W: int
    original_aspect: float

    h: int
    w: int
    target_aspect: float

    def forward(self, *args, **kwargs) -> dict[str, any]:
        """Modify the image, bboxes and coordinates to have a given aspect ratio (shape)

        Use module in Compose and pass structured dict as argument.
        This function will then obtain a dictionary as first and most likely only argument.

        Keyword Args:
            image: One or multiple torchvision images either as byte or float image
                with a shape of ``[B x C x H x W]``.
            box: Zero, one, or multiple bounding boxes per image.
                With N detections and a batch size of B, the bounding boxes have a shape of ``[B*N x 4]``.
                Also, keep in mind that bboxes has to be a two-dimensional tensor,
                because every image in this batch can have a different number of detections.
                The ordering of the bounding boxes will stay the same.
            keypoints: Joint-coordinates as key-points with coordinates in relation to the original image.
                With N detections per image and a batch size of B,
                the coordinates have a max shape of ``[B*N x J x 2|3]``.
                Either batch and detections are stacked in one dimension,
                because every image in this batch can have a different number of detections,
                or there is not batched dimension at all.
                The ordering of the coordinates will stay the same.
            output_size: (h, w) as target height and width of the image
            mode: See class description. Default "zero-pad"

            aspect_round_decimals: (int) (optional)
                Before comparing them, round the aspect ratios to the number of decimals. Default 2

            fill: (optional)
                See parameter fill of torchvision.transforms.v2.Pad()

        Returns:
            Structured dict with updated and overwritten image(s), bboxes and coordinates.
            All additional input values are passed down as well.
        """

        image, bboxes, coordinates, output_size, mode, kwargs, *_ = self._validate_inputs(*args, **kwargs)

        self.H, self.W = image.shape[-2:]
        self.original_aspect: float = self.W / self.H

        self.h, self.w = output_size
        self.target_aspect: float = self.w / self.h

        a_r_decimals: int = int(kwargs.get("aspect_round_decimals", 2))

        # Return early if aspect ratios are fairly close. There will not be any noticeable distortion.
        if mode in ["distort", "outside-crop"] or (
            round(self.original_aspect, a_r_decimals) == round(self.target_aspect, a_r_decimals)
        ):
            return {
                "image": image,
                "box": bboxes,
                "keypoints": coordinates,
                "mode": mode,
                "output_size": output_size,
                **kwargs,
            }

        if mode.endswith("-pad"):
            return self._handle_padding(image, bboxes, coordinates, output_size=output_size, mode=mode, **kwargs)

        if mode == "inside-crop":
            return self._handle_inside_crop(image, bboxes, coordinates, output_size=output_size, mode=mode, **kwargs)

        raise NotImplementedError

    def _handle_padding(
        self,
        image: Image,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: torch.Tensor,
        mode: str,
        **kwargs,
    ) -> dict:
        """To keep forward uncluttered, handle all the padding variants separately.

        Mostly taken from: https://github.com/pytorch/vision/issues/6236#issuecomment-1175971587
        """

        if mode == "mean-pad":
            # compute mean of RGB channels over this batch
            # mean needs to receive tensor as float or complex
            # convert and save the mean with the same dtype as the input image (float or uint8)
            padding_fill = tuple(image.mean(dim=[-4, -2, -1], dtype=torch.float32).to(dtype=image.dtype))
            padding_mode = "constant"
        elif mode == "edge-pad":
            padding_fill = None
            padding_mode = "edge"
        elif mode == "fill-pad":
            padding_fill = kwargs.get("fill")
            padding_mode = "constant"
        elif mode == "reflect-pad":
            padding_fill = None
            padding_mode = "reflect"
        elif mode == "symmetric-pad":
            padding_fill = None
            padding_mode = "symmetric"
        else:  # default and mode == "zero-pad"
            padding_fill = 0
            padding_mode = "constant"

        # compute padding value
        padding: list[int] = compute_padding(old_w=self.W, old_h=self.H, target_aspect=self.w / self.h)
        if padding_mode in ["reflect", "symmetric"] and (
            max(padding[0], padding[2]) >= image.shape[-1] or max(padding[1], padding[3]) >= image.shape[-2]
        ):
            raise ValueError("In padding modes reflect and symmetric, the padding can not be bigger than the image.")
        # pad image, bboxes, and coordinates using the computed values
        # for bboxes and coordinates padding mode and fill do not need to be given
        padded_image: tv_tensors.Image = tv_tensors.wrap(
            tvt_pad(image, padding=padding, fill=padding_fill, padding_mode=padding_mode), like=image
        )
        padded_bboxes: tv_tensors.BoundingBoxes = tv_tensors.wrap(tvt_pad(bboxes, padding=padding), like=bboxes)

        diff = [padding[0], padding[1]]

        if coordinates.shape[-1] == 3:
            # 3d coordinates have no padding in the third dimension
            diff.append(0)
        padded_coords: torch.Tensor = coordinates + torch.tensor(diff, device=coordinates.device)

        return {
            "image": padded_image,
            "box": padded_bboxes,
            "keypoints": padded_coords,
            "mode": mode,
            **kwargs,
        }

    def _handle_inside_crop(
        self,
        image: Image,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: torch.Tensor,
        **kwargs,
    ) -> dict:
        """To keep forward uncluttered, handle the inside cropping or extracting separately."""

        # Compute the new height and new width of the inside crop.
        # At least one of both will be equal to the current H or W
        # When W stays the same: W / nh = w / h
        # When H stays the same: nw / H = w / h
        nh = min(int(self.W / self.w * self.h), self.H)
        nw = min(int(self.H / self.h * self.w), self.W)

        cropped_image: tv_tensors.Image = tv_tensors.wrap(tvt_center_crop(image, output_size=[nh, nw]), like=image)

        # W = delta_w + nw, H = delta_h + nh
        delta = [self.W - nw, self.H - nh]

        # use delta to shift bbox, such that the bbox uses local coordinates
        box_diff = torch.div(torch.tensor(delta + [0.0, 0.0], device=coordinates.device, dtype=torch.float32), 2)
        cropped_bboxes: tv_tensors.BoundingBoxes = tv_tensors.wrap(bboxes - box_diff, like=bboxes)

        # use delta to shift the coordinates, such that they use local coordinates
        if coordinates.shape[-1] == 3:
            # 3d coordinates have no crop in the third dimension
            delta.append(0)

        cropped_coords: torch.Tensor = coordinates - torch.div(
            torch.tensor(delta, device=coordinates.device, dtype=torch.float32), 2
        )

        return {
            "image": cropped_image,
            "box": cropped_bboxes,
            "keypoints": cropped_coords,
            **kwargs,
        }


class CustomResize(Torch_NN_Module, CustomTransformValidator):
    """
    Resize image, bbox and key points with this custom transform.

    The image and bbox are resized using regular torch resize transforms.
    """

    H: int
    W: int

    h: int
    w: int

    def forward(self, *args, **kwargs) -> dict[str, any]:
        """Resize image, bbox and key points in one go.

        Keyword Args:
            image: One single image as tv_tensor.Image of shape ``[B x C x H x W]``
            box: tv_tensor.BoundingBoxes in XYWH box_format of shape ``[N x 4]``, with N detections.
            keypoints: The joint coordinates in global frame as ``[N x J x 2|3]``
            output_size: (h, w) as target height and width of the image

        Returns:
            Will overwrite the image, bbox, and key points with the newly computed values.
            Key Points will be in local image coordinates.

            The new shape of the images is ``[B x C x h x w]``.
        """
        image, bboxes, coordinates, output_size, kwargs, *_ = self._validate_inputs(
            necessary_keys=["image", "box", "keypoints", "output_size"], *args, **kwargs
        )

        # extract shapes for padding
        self.H, self.W = image.shape[-2:]
        self.h, self.w = output_size

        image = tv_tensors.wrap(tvt_resize(image, size=list(output_size), antialias=True), like=image)
        bboxes = tv_tensors.wrap(tvt_resize(bboxes, size=list(output_size), antialias=True), like=bboxes)
        if coordinates.shape[-1] == 2:
            coordinates *= torch.tensor(
                [self.w / self.W, self.h / self.H], dtype=torch.float32, device=coordinates.device
            )
        else:
            # fixme: 3d coordinates have 0 in the third dimension ?
            coordinates *= torch.tensor(
                [self.w / self.W, self.h / self.H, 0], dtype=torch.float32, device=coordinates.device
            )

        return {
            "image": image,
            "box": bboxes,
            "keypoints": coordinates,
            "output_size": output_size,
            **kwargs,
        }


class CustomCropResize(Torch_NN_Module, CustomTransformValidator):
    """Extract all bounding boxes of a single torch tensor image as new image crops
    then resize the result to the given output shape, which makes the results stackable again.
    Additionally, the coordinates will be transformed to use the local coordinate system.
    """

    H: int
    W: int

    h: int
    w: int

    transform = tvt.Compose([CustomToAspect(), CustomResize()])

    def forward(self, *args, **kwargs) -> dict[str, any]:
        """Extract bounding boxes out of one or multiple images and resize the crops to the target shape.

        For bboxes and coordinates, N has to be at least 1.

        Either there is exactly one image or exactly as many stacked images as there are bounding boxes.
        If there is one image, then there can be an arbitrary number (N) of bboxes and key points,
        which will all be extracted from this single source image.
        If there are exactly N equally sized images, with N bounding boxes and N key points,
        every box will be extracted from exactly one image.

        Note:
            If you want to extract 3 bounding boxes from img1 and 2 from img2, either call this method twice,
            or create an image as a stacked or expanded version of img1 and img2. This will only work,
            iff img1 and img2 have the same shape!

        Note:
            The bboxes have to be one `tv_tensors.BoundingBoxes` object,
            therefore, all boxes have to have the same format and canvas size.

        Keyword Args:
            image: One single image or a batch of N images as tv_tensor.Image of shape ``[(N x) C x H x W]``
            box: tv_tensor.BoundingBoxes in XYWH box_format of shape ``[N x 4]``, with N detections.
            keypoints: The joint coordinates in global frame as ``[N x J x 2|3]``
            mode: The mode for resizing.
                Similar to the modes of CustomToAspect, except there is one additional case 'outside-crop' available.
            output_size: (h, w) as target height and width of the image

        Returns:
            Will overwrite the content of the 'image' and 'keypoints' keys
            with the values of the newly computed cropped image and the local coordinates.

            The new shape of the images is ``[N x C x h x w]``.

            The shape of the coordinates will stay the same.

            The bounding boxes will not change at all and will therefore still be in global coordinates.
        """
        # pylint: disable=too-many-locals,too-many-arguments
        image, bboxes, coordinates, output_size, mode, kwargs, *_ = self._validate_inputs(*args, **kwargs)

        # extract shapes for padding
        self.H, self.W = image.shape[-2:]
        self.h, self.w = output_size

        img_crops: list[tv_tensors.Image] = []
        img_crop: tv_tensors.Image
        coord_crops: list[torch.Tensor] = []
        coord_crop: torch.Tensor

        if bboxes.shape[0] != coordinates.shape[0]:
            raise ValueError("Expected bounding boxes and key points to have the same number of dimensions.")
        if image.ndim == 4 and image.shape[-4] > 1 and image.shape[-4] != bboxes.shape[0]:
            raise ValueError(
                "If you provide multiple images, it is expected that they have the same length as the bounding boxes."
            )

        # use torch to round and then cast the bboxes to int
        bboxes_corners = bboxes.round().to(dtype=torch.int)

        for i, (corners, coords) in enumerate(zip(bboxes_corners, coordinates)):
            # get current image
            if image.ndim == 4 and image.shape[-4] > 1:
                image_i = tv_tensors.wrap(image[i].unsqueeze(0), like=image)
            else:
                image_i = image

            if mode == "outside-crop":
                img_crop, coord_crop = self._handle_outside_crop(coords, corners, image_i)
            else:
                # use torchvision cropping and modify the coords accordingly
                left, top, width, height = corners
                left = max(left, 0)
                top = max(top, 0)
                width = max(width, 1)  # min width of 1
                height = max(height, 1)  # min height of 1
                img_crop = tv_tensors.wrap(tvt_crop(image_i, top, left, height, width), like=image)
                delta = [left, top]
                if coords.shape[-1] == 3:
                    delta.append(0)
                coord_crop = coords - torch.tensor(delta, device=coords.device)

            # Resize the image and coord crops to make them stackable again.
            # Use CustomToAspect to make the image the correct aspect ratio.
            # Mostly redundant for outside-crop mode, but even there are a few edge cases.
            modified_data: dict[str, any] = self.transform(
                {
                    "image": img_crop,
                    "box": validate_bboxes(tv_tensors.wrap(bboxes[i], like=bboxes)),
                    "keypoints": validate_key_points(coord_crop),
                    "output_size": output_size,
                    "mode": mode if mode != "outside-crop" else kwargs.get("aspect_mode", "zero-pad"),
                    **kwargs,
                }
            )

            # bboxes will always be in global coordinates and will not be cropped!
            img_crops.append(modified_data["image"])
            coord_crops.append(modified_data["keypoints"])

        assert len(img_crops) == len(coord_crops)

        return {
            "image": tv_tensors.Image(torch.cat(img_crops)),
            "box": bboxes,
            "keypoints": torch.cat(coord_crops),
            "output_size": output_size,
            "mode": mode,
            **kwargs,
        }

    def _handle_outside_crop(
        self,
        coordinates: torch.Tensor,
        corners: torch.Tensor,
        image: Image,
    ) -> tuple[Image, torch.Tensor]:
        """Handle method outside crop to keep forward cleaner"""
        # extract corners from current bboxes
        left, top, box_width, box_height = corners
        # We want to know the necessary padding around the cropped image, so it has the same aspect as the bounding box.
        # Therefore, the target aspect is the aspect of the output size,
        # and the old aspect is the one of the bounding box.
        padding = compute_padding(old_w=box_width, old_h=box_height, target_aspect=self.w / self.h)
        # padding contains positive values for ltrb
        # left and top need to subtract those paddings
        # width and height need to add the paddings of both sides
        # all values have to be within the image boundaries
        new_left: int = min(max(left - padding[0], 0), self.W - 1)
        new_top: int = min(max(top - padding[1], 0), self.H - 1)
        new_width: int = max(min(box_width + padding[0] + padding[2], self.W - 1), 0)
        new_height: int = max(min(box_height + padding[1] + padding[3], self.H - 1), 0)

        # Compute the image and coordinate crops
        image_crop = tv_tensors.wrap(
            tvt.functional.crop(image, left=new_left, top=new_top, width=new_width, height=new_height),
            like=image,
        )

        if coordinates.shape[-1] == 2:
            coord_crop = coordinates - torch.tensor([new_left, new_top]).to(
                dtype=coordinates.dtype, device=coordinates.device
            )
        else:
            # fixme: 3d coordinates no cropping in the third dimension ?
            coord_crop = coordinates - torch.tensor([new_left, new_top, 0]).to(
                dtype=coordinates.dtype, device=coordinates.device
            )

        return image_crop, coord_crop
