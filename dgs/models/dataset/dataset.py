"""
Module for handling data loading and preprocessing using torch Datasets.
"""

import math
import os
from abc import ABC, abstractmethod
from typing import Union

import torch as t
import torchvision
import torchvision.transforms.v2 as tvt
from torch.utils.data import Dataset as TDataset
from torchvision import tv_tensors as tvte
from torchvision.io import VideoReader
from torchvision.transforms.v2.functional import to_dtype

from dgs.models.module import BaseModule
from dgs.utils.config import DEF_VAL
from dgs.utils.constants import VIDEO_FORMATS
from dgs.utils.files import is_project_dir, is_project_file, to_abspath
from dgs.utils.image import CustomCropResize, CustomResize, CustomToAspect, load_image
from dgs.utils.state import collate_states, State
from dgs.utils.types import Config, FilePath, Image, NodePath, Validations  # pylint: disable=unused-import
from dgs.utils.utils import replace_file_type

base_dataset_validations: Validations = {
    "dataset_path": [str, ("any", [("folder exists", False), ("folder exists in project", True)])],
    "data_path": [("any", [str, ("all", [list, ("forall", str)])])],
    # optional
    "crop_mode": ["optional", str, ("in", CustomToAspect.modes)],
    "crop_size": ["optional", tuple, ("len", 2), ("forall", [int, ("gt", 0)])],
    "batch_size": ["optional", int, ("gt", 0)],
    "paths": ["optional", ("any", [("all", [list, ("forall", str)]), str])],
}

video_dataset_validations: Validations = {
    "data_path": [str],
    # optional
    "stream": ["optional", str],
    "num_threads": ["optional", int],
    "video_backend": ["optional", str, ("in", ["pyav", "cuda", "video_reader"])],
    "paths": ["optional", ("any", [("all", [list, ("forall", str)]), str])],
}

dataloader_validations: Validations = {
    "batch_size": ["optional", int],
    "drop_last": ["optional", bool],
    "collate_fn": ["optional", str, ("in", ["lists", "states", "history"])],
    "workers": ["optional", int, ("gte", 0)],
}

image_hist_validations: Validations = {
    "L": [int, ("gt", 0)],
}


class BaseDataset(BaseModule, TDataset):
    """Base class for custom datasets.

    Every dataset is based around the :class:`.State` object,
    which is just a fancy dict containing all the data of the current step.
    But there are two different approaches when thinking about "one sample".
    They have different use-cases and therefore different advantages and disadvantages.

    Using the Bounding Box as Index
    -------------------------------

    One sample of the dataset (one :meth:`.__getitem__` call)
    contains the data of one single bounding-box.
    Therefore, a batch of this dataset contains ``B`` bounding-boxes,
    with the same amount of filepaths, images, key-points, ... .
    The bounding-boxes can be sampled randomly from the dataset, because there is no time-based information.
    This method can be used for generating and training the visual Re-ID embeddings, because the model
    does not care when or in which order the bounding-boxes representing the Person-ID (class labels) are perceived.

    A dataset with this kind of structure will always return a single :class:`.State` containing the data.

    Using the Image ID as Index
    ---------------------------

    One sample of the dataset contains the data of all people / bounding-boxes detected on one single image.
    Because now we can be sure that all detections of this image are in the current sample,
    it is possible to move through the dataset in a frame-by-frame manner, keeping time-based information.
    This is especially useful when working with tracks because the tracks of the current frame
    depend on the tracks of the previous frame(s), at least in most scenarios.

    Due to the time-dependencies, it is not (really) possible to use batches without precomputing part of the tracks,
    which might result in a worse performance during training.
    During evaluation, it is possible to use the ground-truth track-information,
    even though this might change the results of the model,
    and it has to be shown how grave this influences the results.

    When batches are used, the batch-size will vary because every image can have a different number of detections.
    Every image has a specific number of detections ``N``, ranging from zero to an arbitrary number.
    This means that with a batch size of ``B``, we obtain ``B`` original images,
    but the length of the resulting :class:`.State` will most likely be larger.
    Constant batches are possible when trimming overflowing detections, but this is not recommended.
    The other option is to keep the batches with differing sizes.
    The :class:`~torch.utils.data.DataLoader` loads a fixed batch of images,
    the :meth:`~BaseDataset.__getitem__` call of the Dataset then computes the resulting detections
    and returns those as a list of :class:`.State` 's.

    Params
    ------

    dataset_path (FilePath):
        Path to the directory of the dataset.
        The value has to either be a local project path, or a valid absolute path.
    data_path (FilePath):
        The path to the file containing the data for this dataset.
        Either from within the ``dataset_path`` directory, or as absolute path.
        If you want to combine multiple files to a single (concatenated) dataset,
        check out the function :func:`.get_concatenated_dataset` or the ``paths`` parameter.

    Optional Params
    ---------------

    force_img_reshape (bool, optional):
        Whether to accept that images in one folder might have different shapes.
        Default ``DEF_VAL.dataset.force_img_reshape``.
    image_mode (str, optional):
        Only applicable if ``force_img_reshape`` is True.
        The cropping mode used for loading the full images when calling :meth:`.get_image_crops`.
        Value has to be in :attr:`.CustomToAspect.modes``.
        Default ``DEF_VAL.images.image_mode``.
    image_size (tuple[int, int], optional):
        Only applicable if ``force_img_reshape`` is True.
        The size that the original images should have.
        Default ``DEF_VAL.images.image_size``.
    crops_folder (FilePath, optional):
        A path (global, project local, or dataset local), containing the previously cropped images.
        The structure is dataset-dependent, and might not be necessary for some datasets.
        Default is not set, and the crops are generated live.
        Default ``DEF_VAL.dataset.crops_folder``.
    crop_mode (str, optional):
        The mode for image cropping used when calling :meth:`.get_image_crops`.
        Value has to be in :attr:`.CustomToAspect.modes`.
        Default ``DEF_VAL.images.crop_mode``.
    crop_size (tuple[int, int], optional):
        The size, the resized image should have.
        Default ``DEF_VAL.images.crop_size``.
    paths (list[FilePath], optional):
        A list of file paths to concatenate into a single dataset using :func:`.get_concatenated_dataset`.
        Will be ignored by the single dataset.
        Can contain '*' and similar wildcards used in :func:`glob.glob` to search for multiple files matching a pattern.

    Additional Params for the DataLoader
    ------------------------------------

    batch_size (int, optional):
        The batch size to use while creating the DataLoader for this Dataset.
        Default ``DEF_VAL.dataloader.batch_size``.
    drop_last (bool, optional):
        Whether to drop the last batch if its size is unequal to the target batch size.
        Default ``DEF_VAL.dataloader.drop_last``.
    shuffle (bool, optional):
        Whether to shuffle the dataset.
        Default ``DEF_VAL.dataloader.shuffle``.
    workers (int, optional):
        The number of workers for multi-device data-loading.
        Not fully supported!
        Therefore, default 0, no multi-device.
        Default ``DEF_VAL.dataloader.workers``.
    collate_fn (bool, optional):
        Which collate function to use, when collating the States for the DataLoader.
        Can be ``None`` or one of ``"lists"``, ``"states"``, or ``"history"``.
        Default ``DEF_VAL.dataloader.collate_fn``.

    Default Values
    --------------

    .. datatemplate:yaml::
        :source: ../../dgs/default_values.yaml

        {% for param in data['dataloader'] %}
        - {{param}}: {{data['dataloader'][param]}}
        {% endfor %}

        {% for param in data['dataset'] %}
        - {{param}}: {{data['dataset'][param]}}
        {% endfor %}

    """

    data: list
    """Arbitrary data, which will be converted using :func:`self.arbitrary_to_ds`"""

    dataset_path: FilePath
    """The base path to the dataset. Can be used as a starting point for relative paths."""

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(base_dataset_validations)

        self.dataset_path = self.params["dataset_path"]

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        """Has to override call from BaseModule"""
        raise NotImplementedError("Dataset can't be called.")

    def __len__(self) -> int:
        """Override len() functionality for torch.

        The Length of the dataset is technically just len(data), but might be obtained otherwise.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[State, list[State]]:
        """Retrieve data at index from given dataset.
        Should load / precompute the images from given filepaths if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            idx: An index of the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.

        Returns:
            A single :class:`State` containing all the data of this index or a list of multiple states.
        """
        # don't call .to(self.device), the DS should be created on the correct device!
        s: Union[State, list[State]] = self.arbitrary_to_ds(a=self.data[idx], idx=idx)
        return s

    @abstractmethod
    def arbitrary_to_ds(self, a: any, idx: int) -> Union[State, list[State]]:
        """Given an index, convert arbitrary data into a :class:`State` or a list of States."""
        raise NotImplementedError

    def get_image_crops(self, ds: State) -> None:
        """Add the image crops and local key-points to a given state.
        Works for single or batched :class:`State` objects.
        This function modifies the given State in place.

        Will load precomputed image crops by setting ``self.params["crops_folder"]``.
        """
        # image crop is already present
        if "image_crop" in ds.data and "keypoints_local" in ds.data:
            return

        # State has length zero and image and local key points are just placeholders
        if len(ds) == 0:
            ds.data["image_crop"] = tvte.Image(t.empty((0, 3, 1, 1)), device=ds.device)
            ds.data["keypoints_local"] = t.empty(
                (0, ds.J if "keypoints" in ds else 1, ds.joint_dim if "keypoints" in ds else 2), device=ds.device
            )
            return

        # load local keypoints if bbox and keypoints exist
        # fixme this is wrong, because the kps will be transformed if the aspect of the local kps changes
        # if "bbox" in ds and "keypoints" in ds and "keypoints_local" not in ds and ds["keypoints"] is not None:
        #     bbox = convert_bounding_box_format(ds.bbox.detach().clone(), new_format=tvte.BoundingBoxFormat.XYWH)
        #     ds.keypoints_local = ds.keypoints - bbox[:, :2].unsqueeze(-2)

        # check whether precomputed image crops exist
        if "crops_folder" in self.params:
            if "crop_path" not in ds:
                crops_dir: FilePath = self.get_path_in_dataset(self.params.get("crops_folder"))
                ds.crop_path = tuple(
                    os.path.join(
                        crops_dir,
                        ds["img_path"][i].split("/")[-2],
                        f"{ds['image_id'][i]}_{str(ds['person_id'][i])}.jpg",
                    )
                    for i in range(len(ds))
                )
            ds.load_image_crop(store=True)

            ds.data["keypoints_local"] = ds.keypoints_and_weights_from_paths(
                tuple(replace_file_type(fp, new_type=".pt") for fp in ds.crop_path), save_weights=True
            )
            return

        # no crop folder path given, compute the crops
        ds.to(self.device)

        # load original image, reshape if requested
        if self.params.get("force_img_reshape", DEF_VAL["dataset"]["force_img_reshape"]):
            ds.data["image"] = load_image(
                ds.filepath,
                force_reshape=True,
                mode=self.params.get("image_mode", DEF_VAL["images"]["image_mode"]),
                output_size=self.params.get("image_size", DEF_VAL["images"]["image_size"]),
                dtype=t.uint8,
                device=ds.device,
            )
        else:
            ds.data["image"] = load_image(ds.filepath, device=ds.device, dtype=t.uint8)

        structured_input = {
            "images": ds.image,
            "box": ds.bbox,
            "keypoints": (
                ds.keypoints
                if "keypoints" in ds
                else t.zeros((ds.bbox.size(0), 1, 2), device=self.device, dtype=t.float32)
            ),
            "output_size": self.params.get("crop_size", DEF_VAL["images"]["crop_size"]),
            "mode": self.params.get("crop_mode", DEF_VAL["images"]["crop_mode"]),
        }
        new_state = self.transform_crop_resize()(structured_input)

        ds.data["image_crop"] = tvte.Image(
            to_dtype(new_state["image"].to(device=self.device), dtype=t.uint8, scale=True)
        )
        if "keypoints" in ds:
            ds.data["keypoints_local"] = new_state["keypoints"].to(dtype=t.float32, device=self.device)
            assert "joint_weight" in ds.data, "visibility should be given"

    @staticmethod
    def transform_resize_image() -> tvt.Compose:
        """Given an image, bboxes, and key-points, resize them with custom modes.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {
            "image": tv_tensors.Image,
            "box": tv_tensors.BoundingBoxes,
            "keypoints": torch.Tensor,
            "output_size": ImgShape,
            "mode": str,
        }

        Returns:
            A composed torchvision function that accepts a dict as input.
        """
        return tvt.Compose(
            [
                CustomToAspect(),  # make sure the image has the correct aspect ratio for the backbone model
                CustomResize(),
                tvt.ClampBoundingBoxes(),  # make sure to keep bboxes in their canvas_size
                # tvt.SanitizeBoundingBoxes(),  # clean up bboxes if available
                tvt.ToDtype({tvte.Image: t.float32, "others": None}, scale=True),
            ]
        )

    @staticmethod
    def transform_crop_resize() -> tvt.Compose:
        """Given one single image, with its corresponding bounding boxes and key-points,
        obtain a cropped image for every bounding box with localized key-points.

        This transform expects a custom structured input as a dict.

        >>> structured_input: dict[str, any] = {
            "image": tv_tensors.Image,
            "box": tv_tensors.BoundingBoxes,
            "keypoints": torch.Tensor,
            "output_size": ImgShape,
            "mode": str,
        }

        Returns:
            A composed torchvision function that accepts a dict as input.

            After calling this transform function, some values will have different shapes:

            image
                Now contains the image crops as tensor of shape ``[N x C x H x W]``.
            bboxes
                Zero, one, or multiple bounding boxes for this image as tensor of shape ``[N x 4]``.
                And the bounding boxes got transformed into the `XYWH` format.
            coordinates
                Now contains the joint coordinates of every detection in local coordinates in shape ``[N x J x 2|3]``.

        """
        return tvt.Compose(
            [
                tvt.ConvertBoundingBoxFormat(format=tvte.BoundingBoxFormat.XYWH),
                tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
                # tvt.SanitizeBoundingBoxes(),  # clean up bboxes
                CustomCropResize(),  # crop the image at the four corners specified in bboxes
                tvt.ClampBoundingBoxes(),  # duplicate ?
                # tvt.SanitizeBoundingBoxes(),  # duplicate ?
            ]
        )

    def get_path_in_dataset(self, path: FilePath) -> FilePath:
        """Given an arbitrary file- or directory-path, return its absolute path.

        1. check whether the path is a valid absolute path
        2. check whether the path is a valid project path
        3. check whether the path is an existing path within self.params["dataset_path"]

        Returns:
            The absolute found path to the file or directory.

        Raises:
            FileNotFoundError: If the path is not found.
        """
        if os.path.exists(path):
            return os.path.normpath(os.path.abspath(path))
        if is_project_file(path) or is_project_dir(path):
            return to_abspath(path)
        dataset_path = os.path.join(self.dataset_path, str(path))
        if is_project_file(dataset_path) or is_project_dir(dataset_path):
            return to_abspath(dataset_path)
        raise FileNotFoundError(f"Could not find a path to file or directory at {path}")


class BBoxDataset(BaseDataset, ABC):
    """A dataset using the bounding boxes as indices. See :class:`.BaseDataset` for more information."""

    def __getitem__(self, idx: int) -> State:
        """Retrieve data at index from given dataset.
        Should load / precompute the images from given filepaths if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            idx: An index of the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.

        Returns:
            A single :class:`State` containing all the data of this index.
        """
        s: State = self.arbitrary_to_ds(a=self.data[idx], idx=idx)
        return s

    @abstractmethod
    def arbitrary_to_ds(self, a: any, idx: int) -> State:
        """Given a single bounding box (ID) and other arbitrary data, convert everything to a :class:`State` object.
        The index ``idx`` is given additionally, though it might not be used.
        """
        raise NotImplementedError


class ImageDataset(BaseDataset, ABC):
    """A dataset using the image IDs as indices. See :class:`.BaseDataset` for more information."""

    def __getitem__(self, idx: int) -> list[State]:
        """Retrieve the image at index from a given dataset.

        This function should load or precompute the image from the given filepath if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            idx: An index of the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.

        Returns:
            A list of :class:`State`s containing all the data of this index.
        """
        s: list[State] = self.arbitrary_to_ds(a=self.data[idx], idx=idx)
        return s

    @abstractmethod
    def arbitrary_to_ds(self, a: any, idx: int) -> list[State]:
        """Given a single image ID or filepath, obtain the image, bbox, and possibly more information,
        then convert everything to a :class:`State` object.

        The index ``idx`` is given additionally, though it might not be used.
        """
        raise NotImplementedError


class VideoDataset(BaseDataset, ABC):
    """A dataset containing a single video.

    Should support many file formats, but .mp4 works best.

    Notes:
        The torchvision Video-API is in beta status and will most likely change.
        So make sure everything works before upgrading the version of torchvision.

    Params
    ------

    data_path (FilePath):
        The path to the file containing the data for this dataset.
        Either from within the ``dataset_path`` directory, or as absolute path.
        If you want to combine multiple files to a single (concatenated) dataset,
        check out the function :func:`.get_concatenated_dataset` or the ``paths`` parameter.

    Optional Params
    ---------------

    stream (str):
        Default ``DEF_VAL.video_dataset.stream``.
    num_threads (int):
        The number of threads used when loading the video.
        The default is 0 and lets ffmpeg decide the best configuration.
        Default ``DEF_VAL.video_dataset.num_threads``.
    video_backend (str):
        The backend to use when loading the video.
        Default ``DEF_VAL.video_dataset.video_backend``.
    paths (list[FilePath], optional):
        A list of file paths to concatenate into a single dataset.
        Will be ignored by the single dataset.

    """

    data: VideoReader

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(video_dataset_validations)

        torchvision.set_video_backend(self.params.get("video_backend", DEF_VAL["video_dataset"]["video_backend"]))

        if not self.params["data_path"].endswith(VIDEO_FORMATS):
            raise ValueError(f"File with unknown file format. Got {self.params['data_path']}")
        video_path = self.get_path_in_dataset(self.params["data_path"])

        stream = self.params.get("stream", DEF_VAL["video_dataset"]["stream"])

        self.data = VideoReader(
            src=video_path,
            stream=stream,
            num_threads=self.params.get("num_threads", DEF_VAL["video_dataset"]["num_threads"]),
        )
        m = self.data.get_metadata()
        self.fps = m[stream]["fps"][-1]
        self.duration = m[stream]["duration"][-1]

    def __len__(self) -> int:
        """Override len() functionality for torch.

        The Length of the video is obtainable using :meth:`.VideoReader.get_metadata`.
        """
        return math.ceil(self.fps * self.duration)

    def __getitem__(self, idx: int) -> Union[State, list[State]]:
        """Retrieve data at index from given dataset.
        Should load / precompute the images from given filepaths if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            idx: An index of the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.

        Returns:
            A :class:`State` containing all the data of this index or a list of those states.
        """
        # don't call .to(self.device), the DS should be created on the correct device!
        self.data.seek(time_s=float(idx) / self.fps)
        frame = tvte.Image(next(self.data)["data"], device=self.device)
        s: State = self.arbitrary_to_ds(a=frame, idx=idx)
        return s

    @abstractmethod
    def arbitrary_to_ds(self, a: Image, idx: int) -> Union[State, list[State]]:
        raise NotImplementedError


class ImageHistoryDataset(BaseDataset, ABC):
    """A dataset with one index per image ID, the main difference is that in addition to the current frame,
    the last ``L`` frames are given as well.

    See :class:`.BaseDataset` for more information.

    Params
    ------

    L (int):
        The number of frames to include in the history.
    """

    L: int

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(image_hist_validations)

        self.L: int = self.params["L"]

        if self.params.get("collate_fn", False) != "history":
            raise ValueError("The ImageHistoryDataset should always return a list of States.")

    def __len__(self) -> int:
        """Override len() functionality for torch, to make sure, that the first ``L`` indices can't be picked."""
        return len(self.data) - self.L

    def __getitem__(self, idx: int) -> list[State]:
        """Retrieve the image at index from a given dataset plus all the ``L`` images beforehand.

        This function should load or precompute the image and its crops from the given filepath if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            idx: An index of the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.

        Returns:
            A list of :class:`State`s containing the next ``L`` :class:`State`s and the current
            :class:`State`.
            The indices are from ``idx`` to ``idx + L``, where ``idx + L`` is the current frame.

        """
        s: list[State] = self.arbitrary_to_ds(a=self.data[idx : (idx + self.L + 1)], idx=idx)
        return s

    def __getitems__(self, indices: list[int]) -> list[State]:
        """For every index, retrieve the image at that index from a given dataset plus all the ``L`` images beforehand.

        This function should load or precompute the image-crops from the given filepath if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            indices: A list of indices within the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.
                Every index is from ``idx`` to ``idx + L``, where ``idx + L`` is the current frame.

        Returns:
            A list of :class:`State`s containing the next ``L`` (combined) :class:`State`s and the current
            (combined) :class:`State`s.


        """
        states: list[list[State]] = []
        for idx in indices:
            states.append(self.arbitrary_to_ds(a=self.data[idx : (idx + self.L + 1)], idx=idx))
        # combine all the indices, all idx+1, ..., idx+L
        return [collate_states([states[i][l] for i in range(len(states))]) for l in range(self.L)]

    @abstractmethod
    def arbitrary_to_ds(self, a: list[any], idx: int) -> list[State]:
        """Given a single image ID or filepath, obtain the image, bbox, and possibly more information,
        then convert everything to a :class:`State` object.

        The index ``idx`` is given additionally, though it might not be used.
        """
        raise NotImplementedError
