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
from typing import Iterable

import torch
import torchvision.transforms.v2 as tvt_v2
from torch.nn import Module as Torch_NN_Module
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image, read_video
from torchvision.transforms.v2.functional import (
    center_crop as tvt_center_crop,
    crop as tvt_crop,
    pad as tvt_pad,
    resize as tvt_resize,
)

from dgs.utils.files import project_to_abspath
from dgs.utils.types import FilePath, ImgShape, TVImage, TVVideo


def load_image(filepath: FilePath, **kwargs) -> TVImage:
    """Load an image from a given filepath.

    Args:
        filepath: Absolute or local filepath to the image.

    Keyword Args:
        dtype: The dtype of the image, most likely uint8, byte, or float. Default torch.uint8
        device: Device the image should be on. Default "cpu"
        requires_grad: Whether image tensor should include gradient. Default False

    Returns:
        Torch uint8 / byte tensor with its original shape of ``[C x H x W]``.
    """
    fp: FilePath = project_to_abspath(filepath)

    dtype = kwargs.get("dtype", torch.uint8)
    device = kwargs.get("device", "cpu")
    requires_grad = kwargs.get("requires_grad", False)

    return tv_tensors.Image(
        read_image(fp, mode=ImageReadMode.RGB), dtype=dtype, device=device, requires_grad=requires_grad
    )


def load_video(filepath: FilePath, **kwargs) -> TVVideo:
    """Load a video from a given filepath.

    Returns:
        A batch of torch uint8 / byte images with their original shape of ``[T x C x H x W]``.
        With T being the number of frames in the video.
    """
    fp: FilePath = project_to_abspath(filepath)

    dtype = kwargs.get("dtype", torch.uint8)
    device = kwargs.get("device", "cpu")
    requires_grad = kwargs.get("requires_grad", False)

    # read video, save frames and discard audio
    frames, *_ = read_video(fp, output_format="TCHW", pts_unit="sec")

    return tv_tensors.Video(frames, dtype=dtype, device=device, requires_grad=requires_grad)


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
    raise ArithmeticError("During computing the sizes for padding, something unexpected happened.")


class CustomTransformValidator:
    """All the custom transforms need to extract and validate values from args and kwargs.
    The values are always the same, therefore, it is possible to create an external validation function.
    """

    # will only have validation functions, and pylint does not like that.
    # pylint: disable=too-few-public-methods

    @staticmethod
    def _validate_inputs(
        *args, **kwargs
    ) -> tuple[TVImage, tv_tensors.BoundingBoxes, tv_tensors.Mask, ImgShape, str, dict]:
        """Validate the inputs of the forward call.

        Raises:
            Different errors and exceptions if the arguments are invalid

            TypeError

            KeyError

            ValueError
        """
        if len(args) != 1 or not isinstance(args[0], dict):
            raise TypeError(f"invalid args, expected one dict, but got {args}")

        structured_dict: dict[str, any] = args[0]
        structured_dict.update(kwargs)

        necessary_keys = ["image", "labels", "mode", "output_size"]
        if any(k not in structured_dict for k in necessary_keys):
            raise KeyError(
                f"CustomCropResize got invalid kwargs, expected {necessary_keys} but got {structured_dict.keys()}"
            )

        image: tv_tensors.Image = structured_dict.pop("image")
        labels: tuple = structured_dict.pop("labels")
        if not isinstance(labels, tuple) and len(labels) == 2:
            raise TypeError(f"labels should be tuple of length 2 but is {labels}")
        bboxes, coordinates = labels

        output_size: ImgShape = structured_dict.pop("output_size")
        mode: str = structured_dict.pop("mode")

        del labels, structured_dict

        # mode
        if mode not in CustomToAspect.modes:
            raise KeyError(f"Mode: {mode} is not a valid mode.")
        if mode == "fill-pad" and ("fill" not in kwargs or kwargs["fill"] is None):
            raise KeyError("In fill-pad mode, fill should be a value of kwargs and should not be None.")

        # image, bboxes and coordinates should be tv_tensors
        if not isinstance(image, tv_tensors.Image):
            raise TypeError(f"image should be a tv_tensors.Image object but is {type(image)}")
        if not isinstance(bboxes, tv_tensors.BoundingBoxes):
            raise TypeError(f"bboxes should be a tv_tensors.BoundingBoxes object but is {type(bboxes)}")
        if not isinstance(coordinates, tv_tensors.Mask):
            raise TypeError(f"coordinates should be a tv_tensors.Mask object but is {type(coordinates)}")

        # shapes and formats
        if len(bboxes.shape) != 2 or len(coordinates.shape) != 3 or bboxes.shape[-2] != coordinates.shape[-3]:
            raise ValueError(
                f"Bounding boxes and coordinates have mismatching dimension or shape in batch. "
                f"The shapes are - bboxes: {bboxes.shape}, coords: {coordinates.shape}"
            )
        if bboxes.format != tv_tensors.BoundingBoxFormat.XYWH:
            raise ValueError(f"Bounding boxes should be in XYWH format, but are in {bboxes.format}")

        # output_size
        if not isinstance(output_size, (Iterable, tuple)) or len(output_size) != 2:
            raise TypeError(f"output_size is expected to be iterable or tuple of length 2, but is {output_size}")
        if any(not isinstance(v, int) or v <= 0 for v in output_size):
            raise ValueError("output_size should be two positive integers.")

        return image, bboxes, coordinates, output_size, mode, kwargs


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

    new_h: int
    new_w: int
    target_aspect: float

    def forward(self, *args, **kwargs) -> dict[str, any]:
        """Modify the image, bboxes and coordinates to have a given aspect ratio (shape)

        Use module in Compose and pass structured dict as argument.
        This function will then obtain a dictionary as first and most likely only argument.

        Keyword Args:
            image: One or multiple torchvision images either as byte or float image
                with a shape of ``[B x C x H x W]``.
            bboxes: Zero, one, or multiple bounding boxes per image.
                With N detections and a batch size of B, the bounding boxes have a shape of ``[B*N x 4]``.
                Also, keep in mind that bboxes has to be a two-dimensional tensor,
                because every image in this batch can have a different number of detections.
                The ordering of the bounding boxes will stay the same.
            coordinates: Joint-coordinates as key-points with coordinates in relation to the original image.
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

        image, bboxes, coordinates, output_size, mode, kwargs = self._validate_inputs(*args, **kwargs)

        self.H, self.W = image.shape[-2:]
        self.original_aspect: float = self.W / self.H

        self.new_h, self.new_w = output_size
        self.target_aspect: float = self.new_w / self.new_h

        a_r_decimals: int = int(kwargs.get("aspect_round_decimals", 2))

        # Return early if aspect ratios are fairly close. There will not be any noticeable distortion.
        if mode in ["distort", "outside-crop"] or (
            round(self.original_aspect, a_r_decimals) == round(self.target_aspect, a_r_decimals)
        ):
            return {
                "image": image,
                "bboxes": bboxes,
                "coordinates": coordinates,
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
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: tv_tensors.Mask,
        mode: str,
        **kwargs,
    ) -> dict:
        """To keep forward uncluttered, handle all the padding variants separately.

        Mostly taken from: https://github.com/pytorch/vision/issues/6236#issuecomment-1175971587
        """

        if mode == "mean-pad":
            # compute mean of RGB channels and save it as the same dtype as the image (float or uint8)
            padding_fill = tuple(image.mean(dim=[-2, -1], dtype=image.dtype))
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
        padding: list[int] = compute_padding(old_w=self.W, old_h=self.H, target_aspect=self.new_w / self.new_h)
        # pad image, bboxes, and coordinates using the computed values
        padded_image: tv_tensors.Image = tv_tensors.wrap(
            tvt_pad(image, padding=padding, fill=padding_fill, padding_mode=padding_mode), like=image
        )
        padded_bboxes: tv_tensors.BoundingBoxes = tv_tensors.wrap(
            tvt_pad(bboxes, padding=padding, fill=padding_fill, padding_mode=padding_mode), like=bboxes
        )
        padded_coords: tv_tensors.Mask = tv_tensors.wrap(
            tvt_pad(coordinates, padding=padding, fill=padding_fill, padding_mode=padding_mode), like=coordinates
        )

        return {
            "image": padded_image,
            "bboxes": padded_bboxes,
            "coordinates": padded_coords,
            "mode": mode,
            **kwargs,
        }

    def _handle_inside_crop(
        self,
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: tv_tensors.Mask,
        **kwargs,
    ) -> dict:
        """To keep forward uncluttered, handle the inside cropping or extracting separately."""

        # Compute the differences. Where diff is greater than 0, the dimension needs to be modified.
        height_diff = int(self.new_w / self.original_aspect - self.new_h)
        width_diff = int(self.original_aspect * self.new_h - self.new_w)

        if height_diff > 0:
            new_shape: list[int] = [self.W, self.H - height_diff]
        elif width_diff > 0:
            new_shape: list[int] = [self.W - width_diff, self.H]
        else:
            raise ArithmeticError("During computing the sizes for cropping, something unexpected happened.")

        cropped_image: tv_tensors.Image = tv_tensors.wrap(tvt_center_crop(image, output_size=new_shape), like=image)
        cropped_bboxes: tv_tensors.BoundingBoxes = tv_tensors.wrap(
            tvt_center_crop(bboxes, output_size=new_shape), like=bboxes
        )
        cropped_coords: tv_tensors.Mask = tv_tensors.wrap(
            tvt_center_crop(coordinates, output_size=new_shape), like=coordinates
        )

        return {
            "image": cropped_image,
            "bboxes": cropped_bboxes,
            "coordinates": cropped_coords,
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

    def forward(self, *args, **kwargs) -> dict[str, any]:
        """Extract all bounding boxes out of an image and resize them to the new shape.

        Keyword Args:
            image: One single image as tv_tensor.Image of shape ``[(1 x) C x W x H]``
            labels: tuple with bboxes in first and coordinates as second element.
                For bboxes and coordinates, N has to be at least 1. Possibly add a dummy dimension.
                bboxes: tv_tensor.BoundingBoxes in XYWH box_format of shape ``[N x 4]``, with N detections.
                coordinates: The joint coordinates in global frame as ``[N x J x 2|3]``
            mode: The mode for resizing.
                Similar to the modes of CustomToAspect, except there is one additional case 'outside-crop' available.
            output_size: (h, w) as target height and width of the image

        Todo is it possible to get this to work with batches of images?
            Possibly flatten B and N dimension and keep indices somewhere...

        Returns:
            Will overwrite the image and coordinates keys
            with the values of the newly computed cropped image and the local coordinates.

            The new shape of the images is ``[N x C x w x h]``.

            The shape of the coordinates will stay the same.
        """
        # pylint: disable=too-many-locals,too-many-arguments
        image, bboxes, coordinates, output_size, mode, kwargs = self._validate_inputs(*args, **kwargs)

        # extract shapes for padding
        *_, self.H, self.W = image.shape
        self.h, self.w = output_size

        img_crops: list[tv_tensors.Image] = []
        image_crop: tv_tensors.Image
        coord_crops: list[tv_tensors.Mask] = []
        coord_crop: tv_tensors.Mask

        # use torch to round and then cast the bboxes to int
        bboxes_corners = torch.round(bboxes, decimals=0).to(dtype=torch.int)
        for i, (corners, coords) in enumerate(zip(bboxes_corners, coordinates)):
            if mode == "outside-crop":
                image_crop, coord_crop = self._handle_outside_crop(output_size, i, coords, corners, image, bboxes)
            else:
                # extract corners from current bboxes
                left, top, width, height = corners
                image_crop = tv_tensors.wrap(
                    tvt_crop(image, left=left, top=top, width=width, height=height), like=image
                )

                coord_crop = tv_tensors.wrap(
                    tvt_crop(coords, left=left, top=top, width=width, height=height), like=image
                )
            # resize the image and coord crops to make them stackable again
            img_crops.append(
                tv_tensors.wrap(tvt_resize(image_crop, size=list(output_size), antialias=True), like=image)
            )
            coord_crops.append(
                tv_tensors.wrap(tvt_resize(coord_crop, size=list(output_size), antialias=True), like=coordinates)
            )

        return {
            "image": torch.stack(img_crops),
            "labels": (bboxes, torch.stack(coord_crops)),
            "mode": mode,
            **kwargs,
        }

    def _handle_outside_crop(
        self,
        output_size: ImgShape,
        i: int,
        coordinates: tv_tensors.Mask,
        corners: torch.Tensor,
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
    ) -> tuple[TVImage, tv_tensors.Mask]:
        """Handle method outside crop to keep forward cleaner"""
        # extract corners from current bboxes
        left, top, box_width, box_height = corners
        # reuse compute_padding() to compute the corners
        # the bboxes has the old shape, the output size is the target shape
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
            tvt_v2.functional.crop(image, left=new_left, top=new_top, width=new_width, height=new_height),
            like=image,
        )
        coord_crop = tv_tensors.wrap(
            tvt_v2.functional.crop(coordinates, left=new_left, top=new_top, width=new_width, height=new_height),
            like=coordinates,
        )

        # There are some edge cases, especially if the new values are on the maxima,
        # that the image still does not have the right aspect ratio.
        # Therefore, the result is padded with zeros if the aspect doesn't match.
        modified_data: dict[str, any] = tv_tensors.wrap(
            CustomToAspect(
                image=image_crop,
                bboxes=bboxes[i],
                coordinates=coord_crop,
                output_size=output_size,
                # mode="zero-pad",
            ),
            like=image,
        )
        image_crop = modified_data["image"]
        bboxes[i] = modified_data["bboxes"]  # overwrite bboxes in place
        coord_crop = modified_data["coordinates"]

        return image_crop, coord_crop
