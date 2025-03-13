"""
General utility functions.
"""

import os.path
import re
import socket
import sys
import threading
import time
import traceback
import tracemalloc
from functools import wraps
from typing import Union

import numpy as np
import requests
import torch as t
import torch.nn.functional as F
from torchvision import tv_tensors as tvte
from torchvision.io import write_jpeg
from torchvision.transforms import v2 as tvt
from torchvision.transforms.functional import convert_image_dtype

from dgs.utils.config import DEF_VAL
from dgs.utils.files import mkdir_if_missing
from dgs.utils.image import CustomCropResize, load_image_list
from dgs.utils.types import Device, FilePath, FilePaths, Image, Images


def torch_to_numpy(tensor: t.Tensor) -> np.ndarray:
    """Clone and convert torch tensor to numpy.

    Args:
        tensor: Torch tensor on arbitrary hardware.

    Returns:
        A single numpy array with the same shape and type as the original tensor.
    """
    # Detach creates a new tensor with the same data, so it is important to clone.
    # May not be necessary if moving from GPU to CPU, but better safe than sorry
    return tensor.detach().cpu().clone().numpy()


def replace_file_type(fp: FilePath, new_type: str, old_types: Union[None, list[str]] = None) -> FilePath:
    """Replace the file extension of a file path with a new type.

    Args:
        fp: The original file path.
        new_type: The new file type to replace the old one.
        old_types: A list of old file types that are allowed to be replaced.

    Returns:
        The file path with the new file type.

    Raises:
        ValueError: If the old file type is not in the list of allowed types.
    """
    assert isinstance(new_type, str)
    if "." not in new_type:
        new_type = "." + new_type
    base_path, old_type = os.path.splitext(fp)
    if old_types is not None:
        old_types = [str(ot) if str(ot).startswith(".") else "." + str(ot) for ot in old_types]
        if old_type not in old_types:
            raise ValueError(f"Expected file type '{old_type}' to be in {old_types} with or without leading dot.")
    return base_path + new_type


def extract_crops_from_images(
    imgs: Images, bboxes: tvte.BoundingBoxes, kps: t.Tensor = None, **kwargs
) -> tuple[Image, Union[t.Tensor, None]]:
    """Given a list of images, use the bounding boxes to extract the respective image crops.
    Additionally, compute the local key-point coordinates if global key points are given.

    Args:
        imgs: A list containing one or multiple :class:`tv_tensors.Image` tensors.
        bboxes: The bounding boxes as :class:`tv_tensors.BoundingBoxes` of arbitrary format.
        kps: The key points of the respective images.
            The key points will be transformed with the images to obtain the local key point coordinates.
            Default None just means that a placeholder is passed and returned.

    Keyword Args:
        crop_size (ImgShape): The target shape of the image crops.
            Defaults to ``DEF_VAL.images.crop_size``.
        transform (tvt.Compose): A torchvision transform given as Compose to get the crops from the original image.
            Defaults to a version of CustomCropResize.
        crop_mode (str): Defines the resize mode in the transform function.
            Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`.
            Default ``DEF_VAL.images.mode``.

    Returns:
        The 4D image crop(s) with the same format as the image, as tv_tensors of shape ``[N x C x H x W]``.
        The local key points are returned iff ``kps`` was not ``None``.
    """
    if len(imgs) == 0:
        return tvte.Image(t.empty(0, 3, 1, 1)), None

    if len(imgs) != len(bboxes):
        raise ValueError(f"Expected length of imgs {len(imgs)} and number of bounding boxes {len(bboxes)} to match.")

    transform = kwargs.get(
        "transform",
        tvt.Compose([tvt.ConvertBoundingBoxFormat(format="XYWH"), tvt.ClampBoundingBoxes(), CustomCropResize()]),
    )
    res = transform(
        {
            "images": imgs,
            "box": bboxes,
            "keypoints": kps if kps is not None else t.zeros((len(imgs), 1, 2), device=imgs[0].device),
            "mode": kwargs.get("crop_mode", DEF_VAL["images"]["crop_mode"]),
            "output_size": kwargs.get("crop_size", DEF_VAL["images"]["crop_size"]),
        }
    )
    crop = res["image"]
    assert crop.ndim == 4, "dummy check the shape of the crop tensor"

    return crop, None if kps is None else res["keypoints"]


def extract_crops_and_save(
    img_fps: Union[list[FilePath], FilePaths],
    boxes: tvte.BoundingBoxes,
    new_fps: Union[list[FilePath], FilePaths],
    key_points: t.Tensor = None,
    **kwargs,
) -> tuple[Image, t.Tensor]:
    """Given a list of original image paths and a list of target image-crops paths,
    use the given bounding boxes to extract the image content as image crops and save them as new images.

    Does only work if the images have the same size, because otherwise the bounding-boxes would not match anymore.

    Notes:
        It is expected that ``img_fps``, ``new_fps``, and ``boxes`` have the same length.

    Args:
        img_fps: An iterable of absolute paths pointing to the original images.
        boxes: The bounding boxes as tv_tensors.BoundingBoxes of arbitrary format.
        new_fps: An iterable of absolute paths pointing to the image crops.
        key_points: Key points of the respective images.
            The key points will be transformed with the images. Default None just means that a placeholder is passed.

    Keyword Args:
        crop_size (ImgShape): The target shape of the image crops.
            Defaults to `DEF_VAL.images.crop_size`.
        transform (tvt.Transform): A torchvision transform given as Compose to get the crops from the original image.
            Defaults to a cleaner version of :class:`.CustomCropResize`.
        crop_mode (str): Defines the resize mode in the transform function.
            Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`.
            Default ``DEF_VAL.images.mode``.
        quality (int): The quality to save the jpegs as.
            The default of torchvision is 75.
            Default ``DEF_VAL.images.quality``.

    Returns:
        crops, key_points: The computed image crops and their respective key points on the device specified in kwargs.
        The image-crops are saved already, which means in most cases the return values can be ignored.

    Raises:
        ValueError: If input lengths don't match.
    """
    if not len(img_fps) == len(new_fps) == boxes.shape[-2] or (
        key_points is not None and not len(key_points) == len(img_fps)
    ):
        raise ValueError("There has to be an equal amount of image-, crop-paths, boxes, and key points if present.")

    # extract kwargs
    device: Device = kwargs.get("device", "cuda" if t.cuda.is_available() else "cpu")

    imgs: Images = load_image_list(filepath=tuple(img_fps), device=device)

    crops, loc_kps = extract_crops_from_images(imgs=imgs, bboxes=boxes, kps=key_points, **kwargs)

    for i, (fp, crop) in enumerate(zip(new_fps, crops.cpu())):
        mkdir_if_missing(os.path.dirname(fp))
        write_jpeg(
            input=convert_image_dtype(crop, t.uint8),
            filename=fp,
            quality=kwargs.get("quality", DEF_VAL["images"]["jpeg_quality"]),
        )
        if key_points is not None:
            t.save(loc_kps[i].unsqueeze(0), replace_file_type(str(fp), new_type=".pt"))

    return crops.to(device=device), None if key_points is None else loc_kps


class HidePrint:
    """Safely disable printing for a block of code.

    Source: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print/45669280#45669280

    Examples:
        >>> with HidePrint():
        ...     print("Hello")
        ... print("Bye")
        Bye
    """

    _original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def ids_to_one_hot(ids: Union[t.Tensor, t.Tensor], nof_classes: int) -> t.Tensor:
    """Given a tensor containing the class ids as LongTensor, return the one hot representation as LongTensor."""
    return F.one_hot(ids.long(), nof_classes)  # pylint: disable=not-callable


class MemoryTracker:  # pragma: no cover
    """A Wrapper for tracking RAM usage.

    Args:
        interval: How long to sleep after every iteration in seconds.
            Default 1.0 seconds.
        top_n: How many elements to print.
            Default 10.

    Examples:
        >>> @MemoryTracker(interval=0.1)
            def memory_intensive_function():
                # Simulate some memory allocations
                data = []
                for i in range(100000):
                    data.append(i)
                    if i % 10000 == 0:
                        time.sleep(0.01)
    """

    def __init__(self, interval: float = 1.0, top_n: int = 10) -> None:
        self.interval: float = interval
        self.top_n: int = top_n

        self.running: bool = False
        self.thread = None

    def _track_memory(self):
        """Track the memory usage and print it to the console."""
        while self.running:
            snapshot = tracemalloc.take_snapshot()
            line_nos = snapshot.statistics("lineno")

            print(f"\nTop {self.top_n} memory usage:")
            for i, stat in enumerate(line_nos[: self.top_n]):
                print(f"{i}: {stat}")
                for line in stat.traceback.format():
                    print(f"\t{line}")
            print("\n")
            time.sleep(self.interval)

    def start(self):
        """Start the memory tracker."""
        tracemalloc.start()
        self.running = True
        self.thread = threading.Thread(target=self._track_memory)
        self.thread.start()

    def stop(self):
        """Stop the memory tracker."""
        self.running = False
        if self.thread:
            self.thread.join()
        tracemalloc.stop()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start()
            try:
                result = func(*args, **kwargs)
            finally:
                self.stop()
            return result

        return wrapper


def send_discord_notification(message: str) -> None:  # pragma: no cover
    """Sends a notification message to a Discord channel via a webhook.

    Args:
        message: The message content to send to the Discord channel.

    Raises:
        ValueError: If the Discord webhook URL is not set.
    """
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

    if not DISCORD_WEBHOOK_URL:
        raise ValueError("Discord webhook URL is not set. Please set the 'DISCORD_WEBHOOK_URL' environment variable.")
    sender = socket.gethostname()
    message += f"\nSent by: {sender}"
    if len(message) > 2000:
        message = "(truncated) ... " + message[-1980:]
    # escape discord markdown -  with kind regards to:
    # https://github.com/Rapptz/discord.py/blob/59f877fcf013c4ddeeb2b39fc21f03e76f995461/discord/utils.py#L909
    message = re.sub(r"/([_\\~|*`])", r"\\$1", string=message)
    data = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Discord notification: {e}\n\nGot Message:\n{message}")


def notify_on_completion_or_error(info: str = "", min_time: float = 0.0):  # pragma: no cover
    """A decorator that sends a Discord notification when the decorated function
    completes successfully or fails.

    Args:
        info: Additional information to send.
        min_time: Minimum time in seconds the function has to run before sending a notification.

    Returns:
        function: The decorated function with notification functionality.

    Raises:
        Exception: Any exception raised by the decorated function is re-raised after sending a notification.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                # return early if the function completed too quickly (e.g. results are already computed)
                if elapsed_time < min_time:
                    return result
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                message = f":white_check_mark: Function `{func.__name__}` completed successfully in {formatted_time}"
                if len(info) > 0:
                    message += f". {info}"
                if result is not None:
                    message += f"\nResult: {result}"
                if len(args) > 0:
                    message += f"\nargs: `{', '.join(a for a in args)}`"
                if len(kwargs) > 0:
                    message += (
                        f"\nkwargs: "
                        f"{', '.join(f'{k}: {v}' for k, v in kwargs.items() if isinstance(v, (int, float, str)))}"
                    )
                send_discord_notification(message)
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                message = (
                    f":x: Function `{func.__name__}` failed after "
                    f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
                )
                if len(info) > 0:
                    message += f". {info}"
                if len(args) > 0:
                    message += f"\nargs: `{', '.join(a for a in args)}`"
                if len(kwargs) > 0:
                    message += (
                        f"\nkwargs: "
                        f"{', '.join(f'{k}: {v}' for k, v in kwargs.items() if isinstance(v, (int, float, str)))}"
                    )
                err_msg = traceback.format_exc()
                if len(err_msg) > 1000:
                    message += f"\n:warning: Error: ... {err_msg[-1000:]}"
                else:
                    message += f"\n:warning: Error: {err_msg}"
                send_discord_notification(message)
                raise e

        return wrapper

    return decorator
