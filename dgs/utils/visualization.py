"""
Helpers for visualizing data.

For pytorch, torchvision and cv2 image descriptions, see the :ref:`image file description <image_util_page>`.

Matplotlib uses a different order for the images: `[B x H x W x C]`.
At least, the channel for matplotlib is RGB too.
"""
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import ToPILImage
from torchvision.utils import make_grid

from dgs.utils.types import Image, TVImage
from dgs.utils.utils import torch_to_numpy


def torch_show_image(
    imgs: Union[Image, list[Image], torch.Tensor, list[torch.Tensor]], show: bool = True, **kwargs
) -> None:  # pragma: no cover
    """Show a single torch image using matplotlib.

    Args:
        imgs: some kind of torch image to be shown.
            The image can be a modified image, batch of images or list of images.
            Additionally, modified images like with bboxes, skeleton, or grid of images.
            The shape is only three-dimensional: ``[B x C x H x W]``
        show: Whether to show the image.
            Set show as false to print multiple images at once, or to modify the plt object.
            Default: True
    """
    if isinstance(imgs, torch.Tensor) and imgs.ndim == 4:
        imgs = [make_grid(imgs, nrow=kwargs.get("nrow", 8))]
    elif not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if show:
        plt.show()


def torch_to_matplotlib(img: Union[TVImage, torch.Tensor]) -> np.ndarray:
    """Convert a given single or batched torch image Tensor to a numpy.ndarray on the cpu.
    The dimensions are switched from ``[B x C x H x W]`` -> ``[B x H x W x C]``

    Fixme: Can this be removed?

    Args:
        img: torch tensor image with dimensions of ``[B x C x H x W]``

    Returns:
        output shape ``[B x H x W x C]``

        Numpy array of the image
    """
    img = img.squeeze()
    # transform to numpy then switch dimensions
    if img.ndim == 3:
        return torch_to_numpy(img).transpose([1, 2, 0])
    return torch_to_numpy(img).transpose([0, 2, 3, 1])


def save_or_show(
    file_path: str, plot_format: str = "svg", plot_block: bool = False, plot_close: bool = True, **kwargs
) -> None:  # pragma: no cover
    """
    Helper for saving or showing the current plot

    Args:
        file_path: acts as boolean, if "" file will be shown saved otherwise at file_path location
        plot_format: image format, string to be passed to savefig
        plot_block: when showing an image, whether to block further code execution
        plot_close: whether to close the plot after showing/saving
        kwargs: all kwargs that savefig or show accept
    """
    if file_path == "":
        # just show the plot
        plt.show(block=plot_block, **kwargs)
    else:
        # add the file extension if not present yet
        if not file_path.endswith(plot_format):
            file_path += "." + plot_format
        # save plot
        plt.rcParams["savefig.bbox"] = "tight"  # fixme why is this necessary?
        plt.savefig(fname=file_path, format=plot_format, **kwargs)
    if plot_close:
        # plot / fig finalized, so close it
        plt.close()
