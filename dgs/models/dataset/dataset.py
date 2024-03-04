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
from dgs.utils.states import DataSample
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

    Using the Bounding Box as Index
    -------------------------------

    The BaseDataset assumes that one sample of dataset (one :meth:`self.__getitem__` call)
    contains one single bounding box.
    It should be possible to work with a plain dict,
    but to have a few quality-of-life features, the DataSample class was implemented.

    Why not use the Image ID as Index?
    ----------------------------------

    This is **not** chosen since the batch-size might vary when using the image index to create batches,
    because every image can have a different number of detections.
    With a batch size of B, we obtain B original images.
    Every one of those has a specific number of detections N, ranging from zero to an arbitrary number.

    This means that for obtaining batches of a constant size, it is necessary to 'flatten' the inputs.
    Thus creating a mapping from image name and bounding box to the respective dataset.
    With that in place, the DataLoader can retrieve batches with the same batch size.
    The detections of one image might be split into different batches.

    The other option is to have batches with slightly different sizes.
    The DataLoader loads a fixed batch of images, the Dataset computes the resulting detections and returns those.

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
        The cropping mode used for loading the full images when calling :func:``self.get_image_crop``.
        Value has to be in CustomToAspect.modes.
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
        The mode for image cropping used when calling :func:``self.get_image_crop``.
        Value has to be in CustomToAspect.modes.
        Default "zero-pad".
    crop_size (tuple[int, int], optional):
        The size, the resized image should have.
        Default (256, 256).
    batch_size (int, optional):
        The batch size to use while creating the DataLoader for this Dataset.
        Default 16.
    """

    data: list
    """Arbitrary data, which will be converted using :func:`self.arbitrary_to_ds()`"""

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

    def __getitem__(self, idx: int) -> DataSample:
        """Retrieve data at index from given dataset.
        Should load / precompute the images from given filepaths if not done already.

        This method uses the function :func:`self.arbitrary_to_ds()` to obtain the data.

        Args:
            idx: index of the dataset object. Is a reference to the same object as len().

        Returns:
            The DataSample containing all the data of this index.
        """
        # don't call .to(self.device), the DS should be created on the correct device!
        sample: DataSample = self.arbitrary_to_ds(a=self.data[idx], idx=idx)
        if "image_crop" not in sample:
            self.get_image_crops(sample)
        return sample

    @abstractmethod
    def __getitems__(self, indices: list[int]) -> DataSample:
        """Given a list of indices, return a single DataSample object containing them all."""
        raise NotImplementedError

    @abstractmethod
    def arbitrary_to_ds(self, a: any, idx: int) -> DataSample:
        """Given a single arbitrary data sample, convert it to a DataSample object.
        The index ``i`` is given additionally, though might not be used by other datasets.
        """
        raise NotImplementedError

    def get_image_crops(self, ds: DataSample) -> None:
        """Add the image crops and local key points to a given sample.
        Works for single or batched DataSample objects.
        Modifies the given DataSample in place.

        Will load precomputed image crops by setting ``self.params["crops_folder"]``.
        """
        if "crop_path" in ds:
            ds.image_crop = load_image(ds.crop_path, dtype=torch.float32, device=self.device)
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
            ds.image_crop = load_image(ds.crop_path, dtype=torch.float32, device=self.device)
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
        new_sample = self.transform_crop_resize()(structured_input)

        ds.image_crop = new_sample["image"].to(dtype=torch.float32, device=self.device)
        ds.keypoints_local = new_sample["keypoints"].to(dtype=torch.float32, device=self.device)

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
