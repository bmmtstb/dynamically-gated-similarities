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
from dgs.models.dataset.pose_dataset import TorchreidPoseDataset
from dgs.models.states import DataSample
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import mkdir_if_missing, read_json, to_abspath
from dgs.utils.types import Config, Device, FilePath, ImgShape, NodePath, Validations
from dgs.utils.utils import extract_crops_from_images
from torchreid.data import ImageDataset as TorchreidImageDataset

# Do not allow import of 'PoseTrack21' base dataset
__all__ = ["validate_pt21_json", "get_pose_track_21", "PoseTrack21JSON", "PoseTrack21Torchreid"]

pt21_json_validations: Validations = {"json_path": [("instance", FilePath)]}


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

    Notes:
        The image crops are saved in subfolders of the ``crops`` folder
        equal to the original structure of the images' directory.
        The name of the image crops is: ``{image_id}_{person_id}.jpg``.
        The name of the file containing the local keypoints is: ``{image_id}_{person_id}.pt``.
    """
    device: Device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    base_dataset_path = to_abspath(base_dataset_path)

    # load and validate json
    json_file = to_abspath(json_file, root=base_dataset_path)
    json = read_json(json_file)
    validate_pt21_json(json)

    # get the folder name which is the name of the sub-dataset and create the folder within crops
    json_file_name = json_file.removesuffix(".json").split("/")[-1]
    crops_subset_path = os.path.join(base_dataset_path, crops_dir, json_file_name)
    mkdir_if_missing(crops_subset_path)

    # skip if the folder has the correct number of files (images -> len, key points -> len)
    if len(glob.glob(os.path.join(crops_subset_path, "**/*.jpg"), recursive=True)) == len(json["annotations"]) and len(
        glob.glob(os.path.join(crops_subset_path, "**/*.pt"), recursive=True)
    ) == len(json["annotations"]):
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
        ds_path = img_fp.split("/")[-2]
        if ds_path not in crops_subset_path:
            subset_path = os.path.join(crops_subset_path, ds_path)
        else:
            subset_path = crops_subset_path

        d["new_img_fps"].append(
            os.path.normpath(os.path.join(subset_path, f"{anno['image_id']}_{str(anno['person_id'])}.jpg"))
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

        for anno_file in tqdm(files, desc="annotation-files", position=1, leave=False):
            # create folder {train} inside crops
            train_sub_folder = abs_anno_path.split("/")[-1]

            crops_train_dir = os.path.join(crops_path, train_sub_folder)
            mkdir_if_missing(crops_train_dir)

            extract_crops_from_json_annotation(
                base_dataset_path=base_dataset_path,
                json_file=os.path.join(abs_anno_path, anno_file),
                abs_anno_path=abs_anno_path,
                crops_dir=crops_train_dir,
                validate_json=validate_pt21_json,
                **kwargs,
            )


def extract_pt21_image_crops(dataset_dir: FilePath = "./data/PoseTrack21", individually: bool = True, **kwargs) -> None:
    """This function will extract the image crops and image-crop-local key-point coordinates,
    given the full |PT21| dataset.
    The whole function takes about 30 minutes to complete on the whole dataset.

    Args:
        dataset_dir (FilePath): Path to the directory containing the dataset. Default "./data/PoseTrack21".
        individually (bool): Whether to compute the image crops within the test folders at once or one by one.
            True may be used if the Computer has decent hardware and GPU support.
            Still, running the function individually is not that much slower.
            Default True.

    Keyword Args:
        anno_dir (FilePath): A dataset-local path pointing to the directory containing the annotations.
            Default "./posetrack_data/".
        crop_size (ImgShape): The target shape of the image crops. Defaults to ``(256, 256)``.
        device (Device): Device to run the cropping on. Defaults to "cuda" if available "cpu" otherwise.
        transform (tvt.Compose): A torchvision transform given as Compose to get the crops from the original image.
            Defaults to a version of CustomCropResize.
        transform_mode (str): Defines the resize mode in the transform function.
            Has to be in the modes of :class:`~dgs.utils.image.CustomToAspect`. Default "zero-pad".
        quality (int): The quality to save the jpegs as. Default 90. The default of torchvision is 75.

    Notes:
        There is no batched variant of the extract_crops... functions.
        Either all images in the folder are stacked and processed, or only one image is computed at a time.
        Depending on your processing power and (v)RAM,
        computing the stack of up to 1000 images is quite bad for performance.
        Therefore, it is possible to use the `individually=True` flag,
        to compute the crop of every image individually.
        Adding a little overhead by running the transforms more often, but it might still be faster in the end.

    Notes:
        For further information about the kwargs see the functions:
        :func:`~dgs.dataset.posetrack.extract_all_bboxes()` and
        :func:`~dgs.dataset.posetrack.extract_crops_from_json_annotation()`.
    """
    print("Extract crops from annotations.")
    extract_all_bboxes(
        base_dataset_path=dataset_dir,
        anno_dir=kwargs.get("anno_dir", "./posetrack_data/"),
        individually=individually,
        **kwargs,
    )
    print("Extract crops from query")
    extract_crops_from_json_annotation(
        base_dataset_path=dataset_dir,
        json_file=kwargs.get("query_json_file", "./posetrack_person_search/query.json"),
        crops_dir="./crops/",  # query subdir will be created as "sub-dataset"
        individually=True,  # the query annotations have different sizes and have to be looked at individually
        **kwargs,
    )


def generate_pt21_submission(outfile: FilePath) -> None:
    """Given data, generate a |PT21| submission file.

    References:
        https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md

    Args:
        outfile (FilePath):

    Notes:
        The structure of the PT21 submission file is similar to the structure of the inputs::

            {
                "images": [
                    {
                        "file_name": "images/train/000001_bonn_train/000000.jpg",
                        "id": 10000010000,
                        "frame_id": 10000010000
                    },
                ],
                "annotations": [
                    {
                        "bbox": [x1,  y1, w, h],
                        "image_id": 10000010000,
                        "keypoints": [x1, y1, vis1, ..., x17, y17, vis17],
                        "scores": [s1, ..., s17],
                        "person_id": 1024,
                        "track_id": 0
                    },
                ]
            }

        Additionally, note that the visibilities are ignored during evaluation.

    """
    raise NotImplementedError(f"Not implemented {outfile}")


def get_pose_track_21(config: Config, path: NodePath) -> Union[BaseDataset, TorchDataset]:
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

    ds_path = ds.params["dataset_path"]
    ds.validate_params(pt21_json_validations)

    paths: list[FilePath]
    if isinstance(ds_path, (list, tuple)):
        paths = ds_path
    else:
        # path is either directory or single json file
        abs_path: FilePath = ds.get_path_in_dataset(ds.params["json_path"])
        if os.path.isfile(abs_path):
            paths = [abs_path]
        else:
            paths = [
                os.path.normpath(os.path.join(abs_path, child_path))
                for child_path in os.listdir(abs_path)
                if child_path.endswith(".json")
            ]

    return ConcatDataset(
        [
            PoseTrack21JSON(config=config, path=path, json_path=p)
            for p in tqdm(
                paths, desc=f"loading datasets: {ds_path}{ds.params['json_path'] if 'json_path' in ds.params else ''}"
            )
        ]
    )


class PoseTrack21(BaseDataset):
    """Non-Abstract class for PoseTrack21 dataset to be able to initialize it in :func:`get_pose_track_21`.

    Should not be instantiated.
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

    def __getitems__(self, indices: list[int]) -> DataSample:
        raise NotImplementedError

    def arbitrary_to_ds(self, a) -> DataSample:
        raise NotImplementedError


class PoseTrack21JSON(BaseDataset):
    """Load a single precomputed json file from the |PT21| dataset.

    Params
    ------

    json_path (FilePath):
        The path to the json file, either from within the ``dataset_path`` directory, or as absolute path.

    Important Inherited Params
    --------------------------

    dataset_path (FilePath):
        Path to the directory of the dataset.
        The value has to either be a local project path, or a valid absolute path.
    force_img_reshape (bool, optional):
        Whether to accept that images in one folder might have different shapes.
        Default False.

    """

    def __init__(self, config: Config, path: NodePath, json_path: FilePath = None) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(pt21_json_validations)

        # validate and get the path to the json
        if json_path is None:
            json_path: FilePath = self.get_path_in_dataset(self.params["json_path"])

        # validate and get json data
        json: dict[str, list[dict[str, any]]] = read_json(json_path)
        validate_pt21_json(json)

        # create a mapping from image id to full filepath
        self.map_img_id_path: dict[int, FilePath] = {
            img["id"]: to_abspath(os.path.join(self.params["dataset_path"], str(img["file_name"])))
            for img in json["images"]
        }

        # imagesize.get() output = (w,h) and our own format = (h, w)
        img_sizes: set[ImgShape] = {imagesize.get(fp)[::-1] for fp in self.map_img_id_path.values()}
        if self.params.get("force_img_reshape", False):
            # take the biggest value of every dimension
            self.img_shape: ImgShape = (max(size[0] for size in img_sizes), max(size[1] for size in img_sizes))
        else:
            if len(img_sizes) > 1:
                raise ValueError(f"The images within a single folder should have equal shapes. json_path: {json_path}")
            self.img_shape: ImgShape = img_sizes.pop()

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

        def stack_key(key: str, requires_grad: bool = self.rg) -> torch.Tensor:
            return torch.stack(
                [torch.tensor(self.data[i][key], device=self.device, requires_grad=requires_grad) for i in indices]
            )

        keypoints, visibility = (
            torch.stack(
                [
                    (
                        torch.tensor(self.data[i]["keypoints"], requires_grad=self.rg).reshape((17, 3))
                        if len(self.data[i]["keypoints"])
                        else torch.zeros((17, 3))
                    )  # if there are no values present, use zeros
                    for i in indices
                ]
            )
            .to(device=self.device, dtype=torch.float32)
            .split([2, 1], dim=-1)
        )
        ds = DataSample(
            validate=False,  # This is given PT21 data, no need to validate...
            filepath=tuple(self.map_img_id_path[self.data[i]["image_id"]] for i in indices),
            bbox=tv_tensors.BoundingBoxes(
                stack_key("bbox").float(),
                format="XYWH",
                canvas_size=self.img_shape,
                dtype=torch.float32,
                device=self.device,
                requires_grad=self.rg,
            ),
            keypoints=keypoints,
            person_id=stack_key("person_id", requires_grad=False).int(),
            # additional values which are not required
            joint_weight=visibility,
            image_id=stack_key("image_id", requires_grad=False).int(),
        )
        # add the paths to the image crops if the directory containing the crops is given
        if "crops_folder" in self.params:
            dir_path = self.get_path_in_dataset(self.params.get("crops_folder"))

            ds.crop_path = tuple(
                os.path.join(
                    dir_path,
                    self.map_img_id_path[self.data[i]["image_id"]].split("/")[-2],  # dataset name
                    f"{self.data[i]['image_id']}_{str(self.data[i]['person_id'])}.jpg",
                )
                for i in indices
            )

        # make sure to get the image crops for this batch
        self.get_image_crops(ds)
        return ds

    def arbitrary_to_ds(self, a: dict) -> DataSample:
        """Convert raw PoseTrack21 annotations to DataSample object."""
        keypoints, visibility = (
            (
                torch.tensor(a["keypoints"], device=self.device, dtype=torch.float32, requires_grad=self.rg)
                if len(a["keypoints"])
                else torch.zeros((17, 3), device=self.device, dtype=torch.float32, requires_grad=self.rg)
            )
            .reshape((1, 17, 3))
            .split([2, 1], dim=-1)
        )
        ds = DataSample(
            validate=False,  # This is given PT21 data, no need to validate...
            filepath=tuple([self.map_img_id_path[a["image_id"]]]),
            bbox=tv_tensors.BoundingBoxes(
                torch.tensor(a["bbox"]).float(),
                format="XYWH",
                dtype=torch.float32,
                canvas_size=self.img_shape,
                device=self.device,
                requires_grad=self.rg,
            ),
            keypoints=keypoints,
            person_id=torch.tensor(a["person_id"] if "person_id" in a else -1, device=self.device, dtype=torch.long),
            # additional values which are not required
            joint_weight=visibility,
            image_id=torch.tensor(a["image_id"], device=self.device, dtype=torch.long),
        )

        # add the paths to the image crops if the directory containing the crops is given
        if "crops_folder" in self.params:
            dir_path = self.get_path_in_dataset(self.params.get("crops_folder"))

            ds.crop_path = (
                os.path.join(
                    dir_path,
                    self.map_img_id_path[a["image_id"]].split("/")[-2],  # dataset name
                    f"{a['image_id']}_{str(a['person_id'])}.jpg",
                ),  # make sure this is a tuple
            )
        return ds


class PoseTrack21Torchreid(TorchreidImageDataset, TorchreidPoseDataset):
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

        - identities: The training set contains 5474 unique person ids. The biggest person id is 6878
        - images: 163411 images, divided into: 96215 train, 46751 test (gallery), and 20444 val (query)

    Args:
        root (str): Root directory of all the datasets. Default "./data/".
        instance (str): Whether this module works as a TorchreidImageDataset or a custom TorchreidPoseDataset.
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
            query: list[tuple] = self.process_dir(query_dir, path_glob="*/*.jpg", cam_id=1)
            gallery: list[tuple] = self.process_dir(gallery_dir, path_glob="*/*.jpg")
        elif self.instance == "key_points":
            train: list[tuple] = self.process_dir(train_dir, path_glob="*/*.pt", relabel=True)
            query: list[tuple] = self.process_dir(query_dir, path_glob="*/*.pt", cam_id=1)
            gallery: list[tuple] = self.process_dir(gallery_dir, path_glob="*/*.pt")
        else:
            raise NotImplementedError(f"instance {self.instance} is not valid.")

        self.check_before_run([self.dataset_dir, train_dir, query_dir, gallery_dir])

        super().__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index: int) -> dict[str, any]:
        if self.instance == "images":
            return TorchreidImageDataset.__getitem__(self, index)
        if self.instance == "key_points":
            return TorchreidPoseDataset.__getitem__(self, index)
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
        """Originally intended to download the dataset, but authentication is required."""
        warnings.warn(
            "Download not implemented, will only extract crops. "
            "For more information for the download see https://github.com/andoer/PoseTrack21 for more details.",
            Warning,
        )
