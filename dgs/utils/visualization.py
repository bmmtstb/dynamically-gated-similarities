"""
Helpers for visualizing data.

Within pytorch an image is a FloatTensor with a shape of ``[C x h x w]``.
A Batch of torch images therefore has a shape of ``[B x C x h x w]``.
Withing torch the images have channels in order of RGB.

Matplotlib uses a different order for the images: `[B x H x W x C]`.
The channel for matplotlib is RGB too.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from dgs.utils.constants import JOINT_CONNECTIONS_ITOP
from dgs.utils.types import TVImage
from dgs.utils.utils import torch_to_numpy


def torch_to_matplotlib(img: TVImage) -> np.ndarray:
    """
    Convert a given single or batched torch image Tensor to a numpy.ndarray on the cpu.
    The dimensions are switched from ``[B x C x H x W]`` -> ``[B x H x W x C]``

    Args:
        img: torch tensor image with dimensions of ``[B x C x H x W]``

    Returns:
        output shape ``[B x H x W x C]``

        Numpy array of the image
    """
    img = img.squeeze()
    # transform to numpy then switch dimensions
    if len(img.shape) == 3:
        return torch_to_numpy(img).transpose([1, 2, 0])
    return torch_to_numpy(img).transpose([0, 2, 3, 1])


def save_or_show(
    file_path: str, plot_format: str = "svg", plot_block: bool = False, plot_close: bool = True, **kwargs
) -> None:
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
        # just show plot
        plt.show(block=plot_block, **kwargs)
    else:
        # add file extension if not present yet
        if not file_path.endswith(plot_format):
            file_path += "." + plot_format
        # save plot
        plt.savefig(fname=file_path, format=plot_format, **kwargs)
    if plot_close:
        # plot / fig finalised, so close it
        plt.close()


def show_tensor_image(img: TVImage, file_path: str = "", **kwargs) -> None:
    """Display a single tensor image using matplotlib.

    Args:
        img: ``[C x H x W]`` - image to be shown
        file_path: acts as boolean,
            the file is shown if file_path is empty and will be saved otherwise at file_path location.

    Keyword Args:
        see save_or_show()
    """
    np_img = torch_to_matplotlib(img)
    plt.imshow(np_img)
    plt.axis("off")

    # set bbox_inches to "tight" if it does not have a value yet
    kwargs.setdefault("bbox_inches", "tight")
    # handle displaying
    save_or_show(file_path=file_path, **kwargs)


def show_2d_joints(
    img: TVImage,
    joints: torch.Tensor | np.ndarray,
    file_path: str = "",
    **kwargs,
) -> None:
    """Creates a plot to show 2d joint coordinates on image.

    Args:
        img: C x H x W - image as ByteTensor
        joints: K x 2|3 - 2d or 3d joint coordinates
        file_path: acts as boolean, the file will be shown, if string is empty, otherwise the file is saved at file_path

    Keyword Args:
        show_points: whether to print the joint coordinates as points. Default True
        show_skeleton: whether to print the skeleton. Default True
        switch_xy: switch x and y coordinate of joint coords, because images may have switched directions. Default True

        Additionally, see save_or_show()
    """
    # K, d = joints.shape
    if isinstance(joints, torch.Tensor):
        np_joints = torch_to_numpy(joints)
    else:
        np_joints = joints

    np_img = torch_to_matplotlib(img)
    plt.imshow(np_img)

    if kwargs.get("switch_xy", True):
        np_joints[:, [1, 0]] = np_joints[:, [0, 1]]

    if kwargs.get("show_points", True):
        # print 2d points
        for c in np_joints:
            plt.scatter(c[0], c[1], color="b")

    if kwargs.get("show_skeleton", True):
        # print joint connections (skeleton)
        for start, end, color in JOINT_CONNECTIONS_ITOP:  # fixme define or load joint configurations
            plt.plot(
                [np_joints[start][0], np_joints[end][0]],
                [np_joints[start][1], np_joints[end][1]],
                color=color,
            )

    save_or_show(file_path=file_path, bbox_inches="tight", pad_inches=0.1, **kwargs)
