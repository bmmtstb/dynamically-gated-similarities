"""
Module for handling data loading and preprocessing using torch Datasets.
"""

import os
from abc import abstractmethod

import torch
import torchvision.transforms.v2 as tvt
from torch.utils.data import Dataset as TorchDataset
from torchvision import tv_tensors

from dgs.models.module import BaseModule
from dgs.utils.files import is_project_dir, is_project_file, to_abspath
from dgs.utils.image import CustomCropResize, CustomResize, CustomToAspect, load_image
from dgs.utils.state import State
from dgs.utils.types import Config, FilePath, NodePath, Validations  # pylint: disable=unused-import

base_dataset_validations: Validations = {
    "dataset_path": [str, ("any", [("folder exists", False), ("folder exists in project", True)])],
    # optional
    "crop_mode": ["optional", str, ("in", CustomToAspect.modes)],
    "crop_size": ["optional", tuple, ("len", 2), ("forall", (int, ("gt", 0)))],
    "batch_size": ["optional", int, ("gt", 0)],
}


class BaseDataset(BaseModule, TorchDataset):
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
    and returns those as a :class:`.State`.

    Params
    ------

    dataset_path (FilePath):
        Path to the directory of the dataset.
        The value has to either be a local project path, or a valid absolute path.
    force_img_reshape (bool, optional):
        Whether to accept that images in one folder might have different shapes.
        Default False.
    image_mode (str, optional):
        Only applicable if ``force_img_reshape`` is True.
        The cropping mode used for loading the full images when calling :meth:`.get_image_crops`.
        Value has to be in :attr:`.CustomToAspect.modes`.
        Default "zero-pad".
    image_size (tuple[int, int], optional):
        Only applicable if ``force_img_reshape`` is True.
        The size that the original images should have.
        Default (1024, 1024).
    crops_folder (FilePath, optional):
        A path (global, project local, or dataset local), containing the previously cropped images.
        The structure is dataset-dependent, and might not be necessary for some datasets.
        Default is not set, and the crops are generated live.
    crop_mode (str, optional):
        The mode for image cropping used when calling :meth:`.get_image_crops`.
        Value has to be in :attr:`.CustomToAspect.modes`.
        Default "zero-pad".
    crop_size (tuple[int, int], optional):
        The size, the resized image should have.
        Default (256, 256).

    Additional Params for the DataLoader
    ------------------------------------

    batch_size (int, optional):
        The batch size to use while creating the DataLoader for this Dataset.
    drop_last (bool, optional):
        Whether to drop the last batch if its size is unequal to the target batch size.
    shuffle (bool, optional):
        Whether to shuffle the dataset.
    workers (int, optional):
        The number of workers for multi-device data-loading.
        Not fully supported!
        Therefore, default 0, no multi-device.

    Default Values
    --------------

    .. datatemplate:yaml::
        :source: ../../dgs/default_values.yaml

        {% for param in data['dataloader'] %}
        - {{param}}: {{data['dataloader'][param]}}
        {% endfor %}

    """

    data: list
    """Arbitrary data, which will be converted using :func:`self.arbitrary_to_ds`"""

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        """Has to override call from BaseModule"""
        raise NotImplementedError("Dataset can't be called.")

    def __len__(self) -> int:
        """Override len() functionality for torch.

        The Length of the dataset is technically just len(data), but might be obtained otherwise.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> State:
        """Retrieve data at index from given dataset.
        Should load / precompute the images from given filepaths if not done already.

        This method uses the function :func:`self.arbitrary_to_ds` to obtain the data.

        Args:
            idx: An index of the dataset object.
                Is a reference to :attr:`data`, the same object referenced by :func:`__len__`.

        Returns:
            A :class:`State` containing all the data of this index.
        """
        # don't call .to(self.device), the DS should be created on the correct device!
        s: State = self.arbitrary_to_ds(a=self.data[idx], idx=idx)
        if "image_crop" not in s:
            self.get_image_crops(s)
        return s

    @abstractmethod
    def __getitems__(self, indices: list[int]) -> State:
        """Given a list of indices, return a single :class:`State` object containing them all."""
        raise NotImplementedError

    @abstractmethod
    def arbitrary_to_ds(self, a: any, idx: int) -> State:
        """Given a single sample of arbitrary data, convert it to a :class:`State` object.
        The index ``idx`` is given additionally, though it might not be used by other datasets.
        """
        raise NotImplementedError

    def get_image_crops(self, ds: State) -> None:
        """Add the image crops and local key-points to a given state.
        Works for single or batched :class:`State` objects.
        This function modifies the given State in place.

        Will load precomputed image crops by setting ``self.params["crops_folder"]``.
        """
        if "crop_path" in ds:
            ds.image_crop = load_image(ds.crop_path, device=self.device)
            ds.keypoints_local = torch.stack([torch.load(fp.replace(".jpg", ".pt")) for fp in ds.crop_path]).to(
                device=self.device
            )
            return

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
            ds.image_crop = load_image(ds.crop_path, device=self.device)
            ds.keypoints_local = torch.stack([torch.load(fp.replace(".jpg", ".pt")) for fp in ds.crop_path]).to(
                device=self.device
            )
            return

        # no crop folder path given, compute the crops
        self.logger.debug("computing image crops")
        ds.to(self.device)

        if self.params.get("force_img_reshape", False):
            ds.image = load_image(
                ds.filepath,
                force_reshape=True,
                mode=self.params.get("image_mode", "zero-pad"),
                output_size=self.params.get("image_size", (1024, 1024)),
                device=ds.device,
            )
        else:
            ds.image = load_image(ds.filepath, device=ds.device)

        structured_input = {
            "image": ds.image,
            "box": ds.bbox,
            "keypoints": ds.keypoints,
            "output_size": self.params.get("crop_size", (256, 256)),
            "mode": self.params.get("crop_mode", "zero-pad"),
        }
        new_state = self.transform_crop_resize()(structured_input)

        ds.image_crop = new_state["image"].to(dtype=torch.float32, device=self.device)
        ds.keypoints_local = new_state["keypoints"].to(dtype=torch.float32, device=self.device)

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
                tvt.ClampBoundingBoxes(),  # keep bboxes in their canvas_size
                # tvt.SanitizeBoundingBoxes(),  # clean up bboxes if available
                tvt.ToDtype({tv_tensors.Image: torch.float32}, scale=True),
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
                tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),
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
            return os.path.normpath(path)
        if is_project_file(path) or is_project_dir(path):
            return to_abspath(path)
        dataset_path = os.path.join(self.params["dataset_path"], str(path))
        if is_project_file(dataset_path) or is_project_dir(dataset_path):
            return to_abspath(dataset_path)
        raise FileNotFoundError(f"Could not find a path to file or directory at {path}")
