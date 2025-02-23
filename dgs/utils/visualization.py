"""
Helpers for visualizing data.

For pytorch, torchvision and cv2 image descriptions, see the :ref:`image file description <image_util_page>`.

Matplotlib uses a different order for the images: `[B x H x W x C]`.
At least, the channel for matplotlib is RGB too.
"""

# pylint: skip-file

from typing import Union

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torchvision import tv_tensors as tvte
from torchvision.transforms.v2 import ToPILImage
from torchvision.transforms.v2.functional import convert_bounding_box_format, to_dtype
from torchvision.utils import draw_bounding_boxes, draw_keypoints, make_grid

from dgs.utils.constants import SKELETONS
from dgs.utils.types import Image


@t.no_grad()
def torch_show_image(imgs: Union[Image, list[Image], t.Tensor, list[t.Tensor]], show: bool = True, **kwargs) -> None:
    """Show a single torch image using matplotlib.

    Args:
        imgs: Some kind of torch image to be shown.
            The image can be a modified image, batch of images or list of images.
            Additionally, modified images like with bboxes, skeleton, or grid of images.
            The shape is only three-dimensional: ``[B x C x H x W]``
        show: Whether to show the image.
            Set show as false to print multiple images at once, or to modify the plt object.
            Default: True
    """
    if isinstance(imgs, t.Tensor) and imgs.ndim == 4:
        imgs = [make_grid(imgs, nrow=kwargs.get("nrow", 8))]
    elif not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach().clone()
        img: t.Tensor = to_dtype(img, t.uint8, scale=True)
        if img.ndim != 3:
            raise ValueError(f"Sth went wrong, img shape is {img.shape}")
        img = ToPILImage(mode="RGB")(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if show:
        plt.show()


@t.no_grad()
def show_image_with_additional(
    img: Union[Image, t.Tensor],
    key_points: t.Tensor = None,
    bboxes: tvte.BoundingBoxes = None,
    show: bool = True,
    kp_connectivity: Union[str, list[tuple[int, int]]] = None,
    # kp_visibility: torch.Tensor = None,
    **kwargs,
) -> t.Tensor:
    """Draw one torch tensor images, potentially adding key points and bounding boxes on top.

    Notes:
        Fixme: I implemented a key-point visibility flag for torchvision,
            but it is not (yet) in the main / stable branch: https://github.com/pytorch/vision/pull/8225
        When the flag is available, it should be safe to draw the key-points and skeletons again.

    Args:
        img: Some kind of torch image to be shown.
            The image can be a tensor or tv_tensors.Image.
            Additionally, the image can already be modified with bboxes, key points, or skeletons.
            The shape has to be three-dimensional: ``[C x H x W]``.
        key_points: If present, key points to be drawn on top of the image.
            Can contain multiple detections as shape ``[N x K x 2]``
        bboxes: If present, the bounding boxes to be drawn as ``tv_tv_tensors.BoundingBoxes`` object.
            The shape will always be two-dimensional: ``[N x 4]``.
        show: Whether to show the plot after drawing the image(s).
        kp_connectivity: If present the keypoint connectivity as a list of tuples (ID start -> ID end) or
            as string (key) value of ``dgs.utils.constants.SKELETONS``.
        # kp_visibility: (Bool) Tensor containing the visibility of every key point.
        #     Default None means that all the key points are visible.
        #     The shape has to be two-dimensional: ``[N x K]``

    Keyword Args:
        bbox_labels: see value `labels` of function :func:`torchvision.utils.draw_bounding_boxes`
        bbox_colors: see value `colors` of function :func:`torchvision.utils.draw_bounding_boxes`
        bbox_fill: see value `fill` of function :func:`torchvision.utils.draw_bounding_boxes`
        bbox_width: see value `width` of function :func:`torchvision.utils.draw_bounding_boxes`
        bbox_font: see value `font` of function :func:`torchvision.utils.draw_bounding_boxes`
        bbox_font_size: see value `font_size` of function :func:`torchvision.utils.draw_bounding_boxes`
        kp_colors: see value `colors` of function :func:`torchvision.utils.draw_keypoints`
        kp_radius: see value `radius` of function :func:`torchvision.utils.draw_keypoints`
        kp_width: see value `width` of function :func:`torchvision.utils.draw_keypoints`

    Returns:
        int_img: The modified torch image with type of uint8 / byte for later usage.

    Raises:
        TypeError: If the image has the wrong type.
        ValueError: If the image has the wrong shape or the connectivity or given color is faulty.
    """
    if isinstance(img, (tvte.Image, t.Tensor)) and img.ndim == 3:
        pass
    elif isinstance(img, (tvte.Image, t.Tensor)) and img.ndim == 4:
        img = img.squeeze(0)
        if img.ndim != 3:
            raise ValueError(f"image could not be squeezed to correct shape, got: {img.shape}")
    else:
        raise TypeError(f"image is neither tensor nor image. Got {type(img)}")

    int_img: t.Tensor = to_dtype(img, t.uint8, scale=True)

    if bboxes is not None:
        # get key point params
        bbox_params: dict[str, any] = {}
        if "bbox_labels" in kwargs:
            bbox_params["labels"] = kwargs["bbox_labels"]
        if "bbox_colors" in kwargs:
            bbox_params["colors"] = kwargs["bbox_colors"]
        if "bbox_fill" in kwargs:
            bbox_params["fill"] = kwargs["bbox_fill"]
        if "bbox_width" in kwargs:
            bbox_params["width"] = kwargs["bbox_width"]
        if "bbox_font" in kwargs:
            bbox_params["font"] = kwargs["bbox_font"]
        if "bbox_font_size" in kwargs:
            bbox_params["font_size"] = kwargs["bbox_font_size"]
        # draw bboxes
        int_img = draw_bounding_boxes(
            image=int_img,
            boxes=convert_bounding_box_format(inpt=bboxes, new_format=tvte.BoundingBoxFormat.XYXY),
            **bbox_params,
        )

    if key_points is not None:
        # get key point params
        kp_params: dict[str, any] = {}
        if kp_connectivity is not None:
            if isinstance(kp_connectivity, str) and kp_connectivity in SKELETONS.keys():
                kp_params["connectivity"] = SKELETONS[kp_connectivity]
            elif isinstance(kp_connectivity, list) and all(isinstance(kpc, tuple) for kpc in kp_connectivity):
                kp_params["connectivity"] = kp_connectivity
            else:
                raise ValueError(f"Did not recognize connectivity, got: {kp_connectivity}")
        # if kp_visibility is not None:
        #     kp_params["visibility"] = kp_visibility
        if "kp_colors" in kwargs:
            kp_params["colors"] = kwargs["kp_colors"]
        if "kp_radius" in kwargs:
            kp_params["radius"] = kwargs["kp_radius"]
        if "kp_width" in kwargs:
            kp_params["width"] = kwargs["kp_width"]

        # draw the key points - draw them one-by-one, if colors is a list
        if "colors" in kp_params and isinstance(kp_params["colors"], list):
            colors = kp_params.pop("colors")
            if len(colors) != len(key_points):
                raise ValueError(f"There are {len(colors)} colors given but only {len(key_points)} key point tensors")

            for i, color in enumerate(colors):
                int_img = draw_keypoints(image=int_img, keypoints=key_points[i].unsqueeze(0), colors=color, **kp_params)
        else:
            int_img = draw_keypoints(image=int_img, keypoints=key_points, **kp_params)

    if show:
        # print and or show the image
        torch_show_image(int_img, show=show, **kwargs)

    return int_img


@t.no_grad()
def show_images_with_additional(
    imgs: list[Union[Image, t.Tensor]],
    key_points: list[t.Tensor] = None,
    bboxes: list[tvte.BoundingBoxes] = None,
    show: bool = True,
    kp_connectivity: Union[str, list[str], list[tuple[int, int]], list[list[tuple[int, int]]]] = None,
    kp_visibilities: list[t.Tensor] = None,
    **kwargs,
) -> None:
    """Draw one or multiple torch tensor images, potentially adding key points and bounding boxes on top.

    Notes:
        Additional kwargs are passed down.
        See the kwargs of :func:`show_image_with_additional` for more information.

    Args:
        imgs: A list containing some kind of torch images to be shown.
            The shape of every image has to be three-dimensional: ``[C x H x W]``.
        key_points: If present, key points to be drawn on top of every image.
            There can be a different number of detected key points per image.
        bboxes: If present, a list of bounding boxes.
            Bounding boxes as :class:`~torchvision.tv_tv_tensors.BoundingBoxes` objects.
        show: Whether to show the grid of images after drawing the image(s).
        kp_connectivity: If present can contain a list of instructions or one single connectivity.
            The keypoint connectivity is a list of tuples (ID start -> ID end) or
            a string (key) value of :data:``~dgs.utils.constants.SKELETONS``.
        kp_visibilities: (Bool) Tensor containing the visibility of every key point.
            Default None means that all the keypoints are visible.

    Keyword Args:
        grid_nrow: see value `nrow` in :func:`torchvision.utils.make_grid`
        grid_padding: see value `padding` in :func:`torchvision.utils.make_grid`
        grid_normalize: see value `normalize` in :func:`torchvision.utils.make_grid`
        grid_value_range: see value `value_range` in :func:`torchvision.utils.make_grid`
        grid_scale_each: see value `scale_each` in :func:`torchvision.utils.make_grid`
        grid_pad_value: see value `pad_value` in :func:`torchvision.utils.make_grid`

    Raises:
        ValueError: If one of the inputs is not None but has the wrong number of entries.

    """
    amount = len(imgs)

    # key points
    if key_points is None:
        key_points = [None for _ in range(amount)]
    elif len(key_points) != amount:
        raise ValueError(f"Number of key points {len(key_points)} has to be equal to number of images {amount}.")
    # boxes
    if bboxes is None:
        bboxes = [None for _ in range(amount)]
    elif len(bboxes) != amount:
        raise ValueError(f"Number of bboxes {len(bboxes)} has to be equal to number of images {amount}.")
    #
    if kp_connectivity is None:
        kp_connectivity = [None for _ in range(amount)]
    elif isinstance(kp_connectivity, str):
        if kp_connectivity not in SKELETONS.keys():
            raise ValueError(f"Unknown connectivity name {kp_connectivity}. Has to be in {SKELETONS.keys()}")
        kp_connectivity = [kp_connectivity for _ in range(amount)]
    elif isinstance(kp_connectivity, list) and all(isinstance(kpc, tuple) for kpc in kp_connectivity):
        # a single skeleton connectivity
        kp_connectivity = [kp_connectivity for _ in range(amount)]
    elif isinstance(kp_connectivity, list) and len(kp_connectivity) != amount:
        raise ValueError(
            f"Number of values in connectivity {len(kp_connectivity)} has to be equal to number of images {amount}."
        )
    # vis
    if kp_visibilities is None:
        kp_visibilities = [None for _ in range(amount)]
    elif len(kp_visibilities) != amount:
        raise ValueError(f"Number of visibilities {len(kp_visibilities)} has to be equal to number of images {amount}.")

    grid_params: dict[str, any] = {
        key.replace("grid_", ""): kwargs.pop(key) for key in kwargs if key.startswith("grid_")
    }

    new_imgs: list[t.Tensor] = []
    for img, kps, bbox, kpc, kpv in zip(imgs, key_points, bboxes, kp_connectivity, kp_visibilities):
        new_imgs.append(
            show_image_with_additional(
                img=img,
                key_points=kps,
                bbox=bbox,
                show=False,
                kp_connectivity=kpc,
                kp_visibility=kpv,
                **kwargs,
            )
        )

    make_grid(new_imgs, **grid_params)

    if show:
        plt.show()
