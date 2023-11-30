"""
Utility for handling images in pytorch.

Loading, saving, manipulating of RGB-images.

Within pytorch an image is a FloatTensor with a shape of ``[C x h x w]``.
A Batch of torch images therefore has a shape of ``[B x C x h x w]``.
Withing torch the images have channels in order of RGB.

RGB Images in cv2 have a shape of ``[h x w x C]`` and the channels are in order GBR.
Grayscale Images in cv2 have a shape of ``[h x w]``.
"""
from typing import Iterable

import torch
import torchvision.transforms.v2 as tvt_v2
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image, read_video
from torchvision.transforms.v2.functional import center_crop as tvt_center_crop, pad as tvt_pad

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
    frames, *_ = read_video(fp, output_format="TCHW")

    return tv_tensors.Video(frames, dtype=dtype, device=device, requires_grad=requires_grad)


class CustomToAspect(torch.nn.Module):
    """Custom torchvision Transform that modifies the image, bbox, and coordinates simultaneously to match a target
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

    extract
        Uses the target aspect ratio to extract a sub-image out of the original.
        Basically is a center crop with one dimension being as large as possible while maintaining the aspect ratio.

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
        "extract",
        "fill-pad",
        "mean-pad",
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

    def _validate_inputs(
        self,
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: tv_tensors.Mask,
        output_size: ImgShape,
        mode: str,
        **kwargs,
    ) -> None:
        """Validate the inputs of the forward call.

        Raises:
            Different errors if the arguments are invalid
        """
        # mode
        if mode not in self.modes:
            raise KeyError(f"Mode: {mode} is not a valid mode.")
        if mode == "fill-pad" and ("fill" not in kwargs or kwargs["fill"] is None):
            raise KeyError("In fill-pad mode, fill should be a value of kwargs and should not be None.")

        # image and bboxes should be tv_tensors
        if not isinstance(image, tv_tensors.Image):
            raise TypeError(f"image should be a tv_tensors.Image object but is {type(image)}")
        if not isinstance(bboxes, tv_tensors.BoundingBoxes):
            raise TypeError(f"bboxes should be a tv_tensors.BoundingBoxes object but is {type(bboxes)}")
        if not isinstance(coordinates, tv_tensors.Mask):
            raise TypeError(f"coordinates should be a tv_tensors.Mask object but is {type(coordinates)}")

        # output sizes should be positive integers
        if not isinstance(output_size, Iterable) or len(output_size) != 2:
            raise TypeError("output_size should be an iterable of length 2.")
        if any(not isinstance(v, int) or v <= 0 for v in output_size):
            raise ValueError("output_size should be two positive integers.")

    def forward(
        self,
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: tv_tensors.Mask,
        output_size: ImgShape,
        mode: str = "zero-pad",
        **kwargs,
    ) -> dict:
        """Modify the image, bboxes and coordinates to have a given aspect ratio (shape)

        Args:
            image: The original torchvision image, either as byte or float image with shape of ``[C x H x W]``
            bboxes:
            coordinates: joint-coordinates as key-points with a shape of ``[J x 2|3]``
            output_size: (w, h) as target width and height of the image
            mode: See class description. Default "zero-pad"

        Keyword Args:
            aspect_round_decimals: (int) Before comparing them, round the aspect ratios to the number of decimals.
            Default 2

            fill: See parameter fill of torchvision.transforms.v2.Pad()

        Returns:
            Structured dict with updated image, bboxes and coordinates.
            All additional input values are passed down as well.
        """

        self._validate_inputs(image, bboxes, coordinates, output_size, mode, **kwargs)

        self.H, self.W = image.shape[-2:]
        self.original_aspect: float = self.W / self.H

        self.new_h, self.new_w = output_size
        self.target_aspect: float = self.new_w / self.new_h

        a_r_decimals: int = int(kwargs.get("aspect_round_decimals", 2))

        # Return early if aspect ratios are fairly close. There will not be any noticeable distortion.
        if mode == "distort" or round(self.original_aspect, a_r_decimals) == round(self.target_aspect, a_r_decimals):
            return {
                "image": image,
                "bboxes": bboxes,
                "coordinates": coordinates,
                "mode": mode,
                "output_size": output_size,
                **kwargs,
            }

        if mode.endswith("-pad"):
            return self._handle_padding(image, bboxes, coordinates, output_size, mode, **kwargs)

        if mode == "extract":
            return self._handle_extract(image, bboxes, coordinates, output_size, mode, **kwargs)

        raise NotImplementedError

    def _handle_padding(
        self,
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: tv_tensors.Mask,
        output_size: ImgShape,
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
        height_padding = int(self.W // self.target_aspect - self.H)
        width_padding = int(self.target_aspect * self.H - self.W)

        padding: list[int]  # left, top, right, bottom

        if height_padding > 0 > width_padding:
            # +1 pixel on the bottom if new shape is odd
            padding: list[int] = [0, height_padding // 2, 0, height_padding // 2 + (height_padding % 2)]
        elif height_padding < 0 < width_padding:
            # +1 pixel on the right if new shape is odd
            padding: list[int] = [width_padding // 2, 0, width_padding // 2 + (width_padding % 2), 0]
        else:
            raise ArithmeticError("During computing the sizes for padding, something unexpected happened.")

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
            "output_size": output_size,
            **kwargs,
        }

    def _handle_extract(
        self,
        image: TVImage,
        bboxes: tv_tensors.BoundingBoxes,
        coordinates: tv_tensors.Mask,
        output_size: ImgShape,
        mode: str,
        **kwargs,
    ) -> dict:
        """To keep forward uncluttered, handle extract separately."""

        # Compute the differences. Where diff is greater than 0, the dimension needs to be modified.
        height_diff = int(self.new_w / self.original_aspect - self.new_h)
        width_diff = int(self.original_aspect * self.new_h - self.new_w)

        if height_diff > 0:
            new_shape: list[int] = [self.W, self.H - height_diff]
        elif width_diff > 0:
            new_shape: list[int] = [self.W - width_diff, self.H]
        else:
            raise ArithmeticError("During computing the sizes for extracting, something unexpected happened.")

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
            "mode": mode,
            "output_size": output_size,
            **kwargs,
        }


class CustomCrop(torch.nn.Module):
    """Crop torch tensor image to given corners."""

    @staticmethod
    def _validate_inputs(image: TVImage, bboxes: tv_tensors.BoundingBoxes, **kwargs) -> None:
        """Validate the inputs of the forward call.

        Raises:
            Different errors if the arguments are invalid
        """
        if not isinstance(bboxes, tv_tensors.BoundingBoxes) or bboxes.format != tv_tensors.BoundingBoxFormat.XYXY:
            raise TypeError("bbox should be tv_tensors.BoundingBoxes with format tv_tensors.BoundingBoxFormat.XYXY")

    def forward(self, image: TVImage, bboxes: tv_tensors.BoundingBoxes, **kwargs) -> dict:
        """..."""

        # *_, H, W = image.shape
        for bbox in bboxes:
            left, top, width, height = bbox

            tvt_v2.functional.crop(image, left=left, top=top, width=width, height=height)

        raise NotImplementedError
