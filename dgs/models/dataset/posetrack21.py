"""
Load bboxes and poses from an existing .json file of the |PT21|_ dataset.

See https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md#reid-pose-tracking for type definitions.


PoseTrack21 format:

* Bounding boxes have format XYWH
* The 17 key points and their respective visibilities are stored in one list of len 51 [x_i, y_i, vis_i, ...]
"""
import glob
import os
import re
import warnings
from typing import Union

import imagesize
import torch
from torch.utils.data import ConcatDataset, Dataset as TorchDataset
from torchvision import tv_tensors
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset
from dgs.models.dataset.pose_dataset import PoseDataset
from dgs.models.states import DataSample
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import mkdir_if_missing, read_json, to_abspath
from dgs.utils.types import Config, Device, FilePath, ImgShape, NodePath, Validations
from dgs.utils.utils import extract_crops_from_images
from torchreid.data import ImageDataset

# Do not allow import of 'PoseTrack21' base dataset
__all__ = ["validate_pt21_json", "get_pose_track_21", "PoseTrack21JSON", "PoseTrack21Torchreid"]

pt21_json_validations: Validations = {"path": [None]}


def validate_pt21_json(json: dict) -> None:
    """Check whether the given json is valid for the PoseTrack21 dataset.

    Args:
        json (dict): The content of the loaded json file as dictionary.

    Raises:
        ValueError: If json is not a dict.
        KeyError: If json does not contain keys for 'images' or 'annotations'.
    """
    if not isinstance(json, dict):
        raise ValueError(f"The PoseTrack21 json file is expected to be a dict, but was {type(json)}")
    if "images" not in json:
        raise KeyError(f"It is expected that a PoseTrack21 .json file has an images key. But keys are {json.keys()}")
    if "annotations" not in json:
        raise KeyError(
            f"It is expected that a PoseTrack21 .json file has an annotations key. But keys are {json.keys()}"
        )


def extract_crops_from_json_annotation(
    base_dataset_path: FilePath, json_file: FilePath, crops_dir: FilePath, individually: bool = False, **kwargs
) -> None:
    """
    Given the path to

    Args:
        base_dataset_path (FilePath): The absolute path to the base of the dataset.
        json_file (FilePath): The absolute path to the json file containing the annotations.
        crops_dir (FilePath): The absolute path to the directory containing all the crops of the current dataset-split.
        individually (bool): Whether to load and crop the images all at once or individually.
            If the sizes of the images in one json file don't match, individually has to be set to True.
            Default False.

    Keyword Args:
        check_img_sizes (bool): Whether to check if all images in a given folder have the same size before stacking them
            for cropping. Default False.
        device (Device): Device to run the cropping on. Defaults to "cuda" if available "cpu" otherwise.

    Notes:
        For more kwargs see :func:`~dgs.utils.utils.extract_crops_from_images`.

    """
    device: Device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    base_dataset_path = to_abspath(base_dataset_path)

    # load and validate json
    json_file = to_abspath(json_file, root=base_dataset_path)
    json = read_json(json_file)
    validate_pt21_json(json)

    # get the folder name which is the name of the sub-dataset and create the folder within crops
    crops_subset_path = os.path.join(base_dataset_path, crops_dir, json_file.removesuffix(".json").split("/")[-1])
    mkdir_if_missing(crops_subset_path)

    # skip if the folder has the correct number of files (images + key points -> 2*len)
    if len(os.listdir(crops_subset_path)) == 2 * len(json["annotations"]):
        return

    # Because the images in every folder have the same shape,
    # it is possible to stack them and use the batches on the GPU.
    map_id_to_path: dict[int, FilePath] = {i["id"]: i["file_name"] for i in json["images"]}
    d: dict[str, list] = {
        "img_fps": [],
        "new_img_fps": [],
        "boxes": [],
        "key_points": [],
        "sizes": [],
    }

    for anno in json["annotations"]:
        d["boxes"].append(torch.tensor(anno["bbox"], dtype=torch.float32, device=device))
        img_fp: FilePath = os.path.normpath(os.path.join(base_dataset_path, map_id_to_path[anno["image_id"]]))
        d["img_fps"].append(img_fp)
        # imagesize.get() output = (w,h) and our own format = (h, w)
        d["sizes"].append(imagesize.get(img_fp)[::-1])
        kp: torch.Tensor = torch.tensor(anno["keypoints"])
        if kp.shape[0] == 51:
            d["key_points"].append(kp.reshape((17, 3)).to(device=device, dtype=torch.float32))
        else:  # empty key points
            d["key_points"].append(torch.zeros((17, 3)).to(device=device, dtype=torch.float32))

        # There will be multiple detections per image.
        # Therefore, the new image crop name has to include the image id and person id.
        d["new_img_fps"].append(
            os.path.normpath(os.path.join(crops_subset_path, f"{anno['image_id']}_{str(anno['person_id'])}.jpg"))
        )

    # check that the image sizes in the images folder are all the same
    if kwargs.get("check_img_sizes", False) and len(set(d["sizes"])) != 1:
        warnings.warn(f"Not all the images within {json_file} have the same size. Sizes are: {set(d['sizes'])}")

    if individually:
        for img_fp, size, new_fp, bbox, kp in tqdm(
            zip(d["img_fps"], d["sizes"], d["new_img_fps"], d["boxes"], d["key_points"]),
            desc="imgs",
            position=2,
            total=len(d["sizes"]),
            leave=False,
        ):
            extract_crops_from_images(
                img_fps=[img_fp],
                new_fps=[new_fp],
                boxes=tv_tensors.BoundingBoxes(bbox, format="XYWH", canvas_size=size, device=device),
                key_points=kp.unsqueeze(0),
                **kwargs,
            )
    else:
        extract_crops_from_images(
            img_fps=d["img_fps"],
            new_fps=d["new_img_fps"],
            boxes=tv_tensors.BoundingBoxes(
                torch.stack(d["boxes"]), format="XYWH", canvas_size=max(set(d["sizes"])), device=device
            ),
            key_points=torch.stack(d["key_points"]).to(device=device),
            **kwargs,
        )


def extract_all_bboxes(
    base_dataset_path: FilePath = "./data/PoseTrack21/",
    anno_dir: FilePath = "./posetrack_data/",
    **kwargs,
) -> None:
    """Given the path to the |PT21| dataset,
    create a new ``crops`` folder containing the image crops and its respective key point coordinates of every
    bounding box separated by test, train, and validation sets like the images.
    Within every set is one folder per video ID, in which then lie the image crops.
    The name of the crops is: ``{person_id}_{image_id}.jpg``.

    Args:
        base_dataset_path (FilePath): The path to the |PT21| dataset directory.
        anno_dir (FilePath): The name of the directory containing the folders for the training and test annotations.

    Notes:
        For more kwargs see :func:`~dgs.models.dataset.posetrack.extract_crops_from_json_annotation`
        and :func:`~dgs.utils.utils.extract_crops_from_images`.
    """
    # create a few paths and necessary directories
    base_dataset_path = to_abspath(base_dataset_path)
    crops_path = os.path.join(base_dataset_path, "crops")
    mkdir_if_missing(crops_path)

    for abs_anno_path, _, files in tqdm(
        os.walk(os.path.join(base_dataset_path, anno_dir)),
        desc="annotation-folders",
        position=0,
        total=len([f for f in os.listdir(os.path.join(base_dataset_path, anno_dir)) if os.path.isdir(f)]),
    ):
        # skip directories that don't contain files, e.g., the folder containing the datasets
        if len(files) == 0:
            continue
        # inside crops folder, create same structure as within images directory
        # target      => .../PoseTrack21/crops/{train}/{dataset_name}/{img_id}_{person_id}.jpg
        # abs_anno_path => .../PoseTrack21/images/{train}/{dataset_name}.json

        # create folder {train} inside crops
        train_sub_folder = abs_anno_path.split("/")[-1]
        crops_train_dir = os.path.join(crops_path, train_sub_folder)
        mkdir_if_missing(crops_train_dir)

        for anno_file in tqdm(files, desc="annotation-files", position=1, leave=False):
            extract_crops_from_json_annotation(
                base_dataset_path=base_dataset_path,
                json_file=os.path.join(abs_anno_path, anno_file),
                abs_anno_path=abs_anno_path,
                crops_dir=crops_train_dir,
                validate_json=validate_pt21_json,
                **kwargs,
            )


def get_pose_track_21(config: Config, path: NodePath) -> TorchDataset:
    """Load PoseTrack JSON files.

    The path parameter can be one of the following:

    - a path to a directory
    - a single json filepath
    - a list of json filepaths

    In all cases, the path can be
        a global path,
        a path relative to the package,
        or a local path under the dataset_path directory.

    Args:
        config (Config): The overall configuration for the tracker.
        path (NodePath): The path to the dataset-specific parameters.

    Returns:
        An instance of TorchDataset, containing the requested dataset(s) as concatenated torch dataset.
    """
    ds = PoseTrack21(config, path)

    ds_path = ds.params["path"]

    paths: list[FilePath]
    if isinstance(ds_path, (list, tuple)):
        paths = ds_path
    else:
        # path is either directory or single json file
        abs_path: FilePath = ds.get_path_in_dataset(ds_path)
        if os.path.isfile(abs_path):
            paths = [abs_path]
        else:
            paths = [
                os.path.normpath(os.path.join(abs_path, child_path))
                for child_path in os.listdir(abs_path)
                if child_path.endswith(".json")
            ]

    return ConcatDataset(
        [PoseTrack21JSON(config=config, path=path, json_path=p) for p in tqdm(paths, desc="loading datasets")]
    )


class PoseTrack21(BaseDataset):
    """Non-Abstract class for PoseTrack21 dataset to be able to initialize it in :meth:`get_get_pose_track_21`.

    Should not be instantiated.
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

    def arbitrary_to_ds(self, a) -> DataSample:
        raise NotImplementedError


class PoseTrack21JSON(BaseDataset):
    """Load a single precomputed json file."""

    def __init__(self, config: Config, path: NodePath, json_path: FilePath = None) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(pt21_json_validations)

        # validate and get the path to the json
        if json_path is None:
            json_path: FilePath = self.get_path_in_dataset(self.params["path"])
        else:
            if self.print("debug"):
                print(f"Used given json_path '{json_path}' instead of self.params['path'] '{self.params['path']}'")

        # validate and get json data
        json: dict[str, list[dict[str, any]]] = read_json(json_path)
        validate_pt21_json(json)

        # create a mapping from image id to full filepath
        self.map_img_id_path: dict[int, FilePath] = {
            img["image_id"]: to_abspath(os.path.join(self.params["dataset_path"], str(img["file_name"])))
            for img in json["images"]
        }

        # imagesize.get() output = (w,h) and our own format = (h, w)

        self.img_shape: ImgShape = imagesize.get(list(self.map_img_id_path.values())[0])[::-1]

        if any(imagesize.get(path)[::-1] != self.img_shape for img_id, path in self.map_img_id_path.items()):
            raise ValueError(f"The images within a single folder should have equal shapes. json_path: {json_path}")

        self.len = len(json["annotations"])
        self.data: list[dict[str, any]] = json["annotations"]

    def __len__(self) -> int:
        return self.len

    def __getitems__(self, indices: list[int]) -> DataSample:
        """Given list of indices, return DataSample object.

        Batching might be faster if we do not have to create DataSample twice.
        Once for every single object, once for the batch.

        Args:
            indices: List of indices.

        Returns:
            A single DataSample object containing a batch of data.
        """

        def stack_key(key: str) -> torch.Tensor:
            return torch.stack([torch.tensor(self.data[i][key], device=self.device) for i in indices])

        keypoints, visibility = (
            torch.tensor(
                torch.stack([torch.tensor(self.data[i]["keypoints"]).reshape((17, 3)) for i in indices]),
            )
            .to(device=self.device, dtype=torch.float32)
            .split([2, 1], dim=-1)
        )
        ds = DataSample(
            validate=False,
            filepath=tuple(self.map_img_id_path[self.data[i]["image_id"]] for i in indices),
            bbox=tv_tensors.BoundingBoxes(
                stack_key("bbox"), format="XYWH", canvas_size=self.img_shape, device=self.device
            ),
            keypoints=keypoints,
            person_id=stack_key("person_id").int(),
            # additional values which are not required
            joint_weight=visibility,
            track_id=stack_key("track_id").int(),
            id=stack_key("id").int(),
            image_id=stack_key("image_id").int(),
            category_id=stack_key("category_id").int(),
            bbox_head=tv_tensors.BoundingBoxes(
                stack_key("bbox_head"), format="XYWH", canvas_size=self.img_shape, device=self.device
            ),
        )
        # make sure to get image crop for batch
        self.get_image_crop(ds)
        return ds

    def arbitrary_to_ds(self, a: dict) -> DataSample:
        """Convert raw PoseTrack21 annotations to DataSample object."""
        keypoints, visibility = (
            torch.tensor(a["keypoints"], device=self.device, dtype=torch.float32)
            .reshape((1, 17, 3))
            .split([2, 1], dim=-1)
        )
        return DataSample(
            validate=False,  # This is given PT21 data, no need to validate...
            filepath=tuple([self.map_img_id_path[a["image_id"]]]),
            bbox=tv_tensors.BoundingBoxes(a["bbox"], format="XYWH", canvas_size=self.img_shape, device=self.device),
            keypoints=keypoints,
            person_id=a["person_id"] if "person_id" in a else -1,
            # additional values which are not required
            joint_weight=visibility,
            track_id=a["track_id"],
            id=a["id"],
            image_id=a["image_id"],
            category_id=a["category_id"],
            bbox_head=tv_tensors.BoundingBoxes(
                a["bbox_head"], format="XYWH", canvas_size=self.img_shape, device=self.device
            ),
        )


class PoseTrack21Torchreid(ImageDataset, PoseDataset):
    r"""Load PoseTrack21 as torchreid dataset.
    Depending on the argument ``instance`` this Dataset either contains image crops or key point crops.

    Reference
    ---------

    Doering et al. Posetrack21: A dataset for person search, multi-object tracking and multi-person pose tracking.
    IEEE / CVF 2022.

    URL
    ----

    `<https://github.com/andoer/PoseTrack21>`_

    Dataset statistics
    ------------------

        - identities: The training set contains 5474 unique person ids.
        - images: 163411 images, divided into: 96215 train, 46751 test (gallery), and 20444 val (query)

    Args:
        root (str): Root directory of all the datasets. Default "./data/".
        instance (str): Whether this module works as an ImageDataset or a PoseDataset.
            Has to be one of: ["images", "key_points"]. Default "all".

    Notes:
        The bbox crops are generated using either the modified :func:`self.download_dataset` or
        if you don't want to use default configuration something similar using :func:`extract_all_bboxes`.

    Notes:
        Train is for training the model.
        The query and gallery are used for testing,
        where for each image in the query you find similar persons in the gallery set.
    """
    _junk_pids: list[int] = [-1]

    dataset_dir: FilePath = "PoseTrack21"
    """Name of the directory containing the dataset within ``root``."""

    def __init__(self, root: str = "", instance: str = "images", **kwargs):
        self.root: FilePath = (
            os.path.abspath(os.path.expanduser(root)) if root else os.path.join(PROJECT_ROOT, "./data/")
        )
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.instance: str = instance

        # annotation directory
        self.annotation_dir: FilePath = os.path.join(self.dataset_dir, "posetrack_data")

        # image directory
        train_dir: FilePath = os.path.join(self.dataset_dir, "crops/train")
        query_dir: FilePath = os.path.join(self.dataset_dir, "crops/query")
        gallery_dir: FilePath = os.path.join(self.dataset_dir, "crops/val")

        if self.instance == "images":
            train: list[tuple] = self.process_dir(train_dir, path_glob="*/*.jpg", relabel=True)
            query: list[tuple] = self.process_dir(query_dir, path_glob="*.jpg", cam_id=1)
            gallery: list[tuple] = self.process_dir(gallery_dir, path_glob="*/*.jpg")
        elif self.instance == "key_points":
            train: list[tuple] = self.process_dir(train_dir, path_glob="*/*.pt", relabel=True)
            query: list[tuple] = self.process_dir(query_dir, path_glob="*.pt", cam_id=1)
            gallery: list[tuple] = self.process_dir(gallery_dir, path_glob="*/*.pt")
        else:
            raise NotImplementedError(f"instance {self.instance} is not valid.")

        self.check_before_run([self.dataset_dir, train_dir, query_dir, gallery_dir])

        super().__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index: int) -> dict[str, any]:
        if self.instance == "images":
            return ImageDataset.__getitem__(self, index)
        if self.instance == "key_points":
            return PoseDataset.__getitem__(self, index)
        raise NotImplementedError(f"instance {self.instance} is not valid.")

    def process_dir(
        self, dir_path: FilePath, path_glob: str, relabel: bool = False, cam_id: int = 0
    ) -> list[tuple[str, int, int, int]]:  # pragma: no cover
        """
        Process all the data of one directory.

        Args:
            dir_path (FilePath): The absolute path to the directory containing images.
                In this case will be something like '.../data/PoseTrack21/crops/train'.
            path_glob (str): The glob pattern to find the files within the given ``dir_path``.
            relabel (bool, optional): Whether to create labels from to pids,
                to reduce the number of parameters in the model. Default False.
            cam_id (int, optional): The id of the camera to use.
                The cam_id of the query dataset has to be different from the cam_id of the gallery,
                see `this issue <https://github.com/KaiyangZhou/deep-person-reid/issues/442#issuecomment-868757430>`_
                for more details.
                Default 0.

        Returns:
            data (list[tuple[str, int, int, int]]): A list of tuples containing the absolute image path,
                person id (label), camera id, and dataset id.
                The dataset id is the video_id with a leading 1 for mpii and 2 for bonn, to remove duplicates.

        Raises:
            OSError: If there is no data.
        """
        img_paths: list[str] = glob.glob(os.path.join(dir_path, path_glob))
        pattern = re.compile(r"(\d+)_(\d+)")

        if len(img_paths) == 0:
            raise OSError(f"Could not find any instances using glob: {path_glob}.")

        pid_container: set = set(int(pattern.search(fp).groups()[1]) for fp in img_paths)
        pid_container -= set(self._junk_pids)  # junk images are just ignored
        pid2label: dict[int, int] = {pid: label for label, pid in enumerate(pid_container)}

        data: list[tuple[str, int, int, int]] = []
        # (path, pid, camid, dsetid)
        # path: is the absolute path to the file of the cropped image
        # pid: person id
        # camid: id of the camera = 0 for all train and gallery images; 1 for all in query
        # dsetid: dataset id = video_id with a leading 1 for mpii and 2 for bonn

        for img_path in img_paths:
            _, pid = map(int, pattern.search(img_path).groups())
            if pid in self._junk_pids:
                continue  # junk images are just ignored
            if relabel:
                pid = pid2label[pid]

            # create dsetid as int({"1" if ds_type == "mpii" else "2"}{video_id})
            if "_" not in (ds_dir := img_path.split("/")[-2]):
                dsetid: int = 0
            else:
                ds_id, ds_type, *_ = ds_dir.split("_")
                dsetid: int = int(f"{'1' if ds_type == 'mpii' else '2'}{str(ds_id)}")

            data.append((img_path, pid, cam_id, dsetid))
        return data

    # I want download_dataset() to be callable using ``PoseTrack21Torchreid.download_dataset()``
    # pylint: disable = unused-argument, arguments-differ
    @staticmethod
    def download_dataset(
        dataset_dir: FilePath = "./data/PoseTrack21", dataset_url: Union[FilePath, None] = None, **kwargs
    ) -> None:  # pragma: no cover
        """Originally intended to download the dataset, but authentication is required.
        Therefore, this function will only extract the image crops and image-crop-local key-point coordinates,
        given the full dataset.

        Args:
            dataset_dir (FilePath): Path to the directory containing the dataset. Default "./data/PoseTrack21".
            dataset_url (Union[FilePath, None]): Irrelevant here.

        Keyword Args:
            crop_size (ImgShape): The target shape of the image crops. Defaults to ``(256, 256)``.
            device (Device): Device to run the cropping on. Defaults to "cuda" if available "cpu" otherwise.
            transform (tvt.Compose): A torchvision transform given as Compose to get the crops from the original image.
                Defaults to a version of CustomCropResize.
            transform_mode (str): Defines the resize mode in the transform function.
                Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`. Default "zero-pad".
            quality (int): The quality to save the jpegs as. Default 90. The default of torchvision is 75.
            individually (bool): Whether to extract the image crops for train and val individually.
                The value has to be true for the 'query', due to different image shapes. Default False.

        Warnings:
            Warning: Information that this function only extracts the crops and does not download the dataset.

        Notes:
            There is no batched variant of the extract_crops... functions.
            Either all images in the folder are stacked and processed, or only one image is computed at a time.
            Depending on your processing power and (v)RAM,
            computing the stack of up to 1000 images is quite bad for performance.
            Therefore, it is possible to use the `individually=True` flag,
            to compute the crop of every image individually.
            Adding a little overhead, but it might still be faster in the end.
            Takes roughly 30 minutes in total.

        Notes:
            For further information about the kwargs see :func:`~dgs.dataset.posetrack.extract_all_bboxes()`.
        """
        warnings.warn(
            "Download not implemented, will only extract crops. "
            "For more information for the download see https://github.com/andoer/PoseTrack21 for more details.",
            Warning,
        )
        print("Extract crops from annotations.")
        extract_all_bboxes(
            base_dataset_path=dataset_dir,
            # anno_dir="./posetrack_data/",
            individually=kwargs.pop("individually", False),
            **kwargs,
        )
        print("Extract crops from query")
        extract_crops_from_json_annotation(
            base_dataset_path=dataset_dir,
            json_file="./posetrack_person_search/query.json",
            crops_dir="./crops/",  # query subdir will be created as "sub-dataset"
            individually=True,  # we know that the query annotations have different sizes
            **kwargs,
        )