r"""
Load bboxes and poses from an existing .json file of the |PT21|_ dataset.

See https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md#reid-pose-tracking for type definitions.


PoseTrack21 format:

* Bounding boxes have format XYWH
* The 17 key points and their respective visibilities are stored in one list of len 51.
  The list contains the x- and y-coordinate and the visibility: \n
  [``x``\ :sub:`i`, ``y``\ :sub:`i`, ``vis``\ :sub:`i`, ...]

Notes:
    The original P21-paper said,
    that during evaluation they ignore all person detections that overlap with the ignore regions.
"""

import glob
import os
import re
import shutil
import warnings
from abc import ABC
from typing import Union

import imagesize
import numpy as np
import torch as t
from torchvision import tv_tensors as tvte
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset, BBoxDataset, ImageDataset, ImageHistoryDataset
from dgs.models.dataset.torchreid_pose_dataset import TorchreidPoseDataset
from dgs.utils.config import DEF_VAL
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import mkdir_if_missing, read_json, to_abspath
from dgs.utils.state import collate_bboxes, collate_tensors, State
from dgs.utils.types import Config, Device, FilePath, FilePaths, ImgShape, NodePath, Validations
from dgs.utils.utils import extract_crops_and_save

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Cython evaluation.*is unavailable", category=UserWarning)
    try:
        # If torchreid is installed using `./dependencies/torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.data import ImageDataset as TorchreidImageDataset
    except ModuleNotFoundError:
        # if torchreid is installed using `pip install torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.reid.data import ImageDataset as TorchreidImageDataset


pt21_json_validations: Validations = {
    "crops_folder": [str, ("folder exists", False)],
    # optional
    "id_map": ["optional", str],
    "load_img_crops": ["optional", bool],
}


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
    Given the path to a json file containing images and annotations in the PT21 style, extract all the bbox image crops.

    Args:
        base_dataset_path (FilePath): The absolute path to the base of the dataset.
        json_file (FilePath): The absolute path to the json file containing the annotations.
        crops_dir (FilePath): The absolute path to the directory containing all the crops of the current dataset-split.
        individually (bool): Whether to load and crop the images all at once or individually.
            If the sizes of the images in one json file don't match, individually has to be set to True.
            Default False.

    Keyword Args:
        check_img_sizes (bool): Whether to check if all images in a given folder have the same size before stacking them
            for cropping.
            Default ``DEF_VAL.dataset.pt21.check_img_sizes``.
        device (Device): Device to run the cropping on. Defaults to "cuda" if available "cpu" otherwise.

    Notes:
        For more kwargs see :func:`~dgs.utils.utils.extract_crops_from_images`.

    Notes:
        The image crops are saved in subfolders of the ``crops`` folder
        equal to the original structure of the images' directory.
        The name of the image crops is: ``{image_id}_{person_id}.jpg``.
        The name of the file containing the local keypoints is: ``{image_id}_{person_id}.pt``.
    """
    device: Device = kwargs.get("device", "cuda" if t.cuda.is_available() else "cpu")

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
        d["boxes"].append(t.tensor(anno["bbox"], dtype=t.float32, device=device))
        img_fp: FilePath = os.path.normpath(os.path.join(base_dataset_path, map_id_to_path[anno["image_id"]]))
        d["img_fps"].append(img_fp)
        # imagesize.get() output = (w,h) and our own format = (h, w)
        d["sizes"].append(imagesize.get(img_fp)[::-1])
        kp: t.Tensor = t.tensor(anno["keypoints"])
        if kp.shape[0] == 51:
            kp, _ = kp.reshape((17, 3)).to(device=device, dtype=t.float32).split([2, 1], dim=-1)
            d["key_points"].append(kp)
        else:  # empty key points
            d["key_points"].append(t.zeros((17, 2)).to(device=device, dtype=t.float32))

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
    if kwargs.get("check_img_sizes", DEF_VAL["dataset"]["pt21"]["check_img_sizes"]) and len(set(d["sizes"])) != 1:
        warnings.warn(f"Not all the images within {json_file} have the same size. Sizes are: {set(d['sizes'])}")

    if individually:
        for img_fp, size, new_fp, bbox, kp in tqdm(
            zip(d["img_fps"], d["sizes"], d["new_img_fps"], d["boxes"], d["key_points"]),
            desc="imgs",
            position=2,
            total=len(d["sizes"]),
            leave=False,
        ):
            extract_crops_and_save(
                img_fps=[img_fp],
                new_fps=[new_fp],
                boxes=tvte.BoundingBoxes(bbox, format="XYWH", canvas_size=size, device=device),
                key_points=kp.unsqueeze(0),
                **kwargs,
            )
    else:
        extract_crops_and_save(
            img_fps=d["img_fps"],
            new_fps=d["new_img_fps"],
            boxes=tvte.BoundingBoxes(
                t.stack(d["boxes"]), format="XYWH", canvas_size=max(set(d["sizes"])), device=device
            ),
            key_points=t.stack(d["key_points"]).to(device=device),
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
        crop_mode (str): Defines the resize mode in the transform function.
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
        :func:`~.extract_all_bboxes` and
        :func:`~.extract_crops_from_json_annotation`.
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
    print("Extract crops from val")
    val_dst = os.path.abspath(
        os.path.join(dataset_dir, kwargs.get("val_dst_json_file", "./posetrack_person_search/gallery.json"))
    )
    val_src = to_abspath(os.path.join(dataset_dir, kwargs.get("val_json_file", "./posetrack_person_search/val.json")))
    shutil.copyfile(val_src, val_dst)
    extract_crops_from_json_annotation(
        base_dataset_path=dataset_dir,
        json_file=val_dst,
        crops_dir="./crops/",  # val / gallery subdir will be created as "sub-dataset"
        individually=True,  # the val annotations have different sizes and have to be looked at individually
        **kwargs,
    )
    print("Extract crops from train")
    extract_crops_from_json_annotation(
        base_dataset_path=dataset_dir,
        json_file=kwargs.get("train_json_file", "./posetrack_person_search/train.json"),
        crops_dir="./crops/",  # train subdir will be created as "sub-dataset"
        individually=True,  # the train annotations have different sizes and have to be looked at individually
        **kwargs,
    )


class PoseTrack21BaseDataset(BaseDataset, ABC):
    """Abstract base class for the |PT21|_ based datasets."""

    img_shape: ImgShape
    """The size of the images in the dataset."""

    skeleton_name = "coco"
    """The format of the skeleton."""

    nof_kps: int = 17
    """The number of key points."""

    bbox_format: str = tvte.BoundingBoxFormat.XYWH
    """The format of the bounding boxes."""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config=config, path=path)

        # validate params
        self.validate_params(pt21_json_validations)

        # validate and get json data
        self.json: dict[str, list[dict[str, any]]] = read_json(self.get_path_in_dataset(self.params["data_path"]))
        validate_pt21_json(self.json)

        # create a mapping from image id to full filepath
        self.map_img_id_to_img_path: dict[int, FilePath] = {
            img["id"]: to_abspath(os.path.join(self.params["dataset_path"], str(img["file_name"])))
            for img in self.json["images"]
        }

        self._obtain_image_size(fps=list(self.map_img_id_to_img_path.values()))

        # the precomputed image crops lie in a specific folder
        self.crops_dir: FilePath = self.get_path_in_dataset(self.params.get("crops_folder"))

    @staticmethod
    def _get_dataset_name_from_img_path(img_path: FilePath) -> str:
        """Get the dataset name from the image path."""
        return os.path.basename(os.path.dirname(img_path))

    @staticmethod
    def _get_dataset_name_from_json_path(json_path: FilePath) -> str:
        """Get the dataset name from the json path."""
        return os.path.splitext(os.path.basename(json_path))[0]

    def _obtain_image_size(self, fps: Union[FilePaths, list[FilePath]]) -> None:
        """Get the size of the images in the dataset."""
        if self.params.get("force_img_reshape", DEF_VAL["dataset"]["force_img_reshape"]):
            # force reshape, therefore use the given image size
            self.img_shape: ImgShape = self.params.get("image_size", DEF_VAL["images"]["image_size"])
            return

        # imagesize.get() output = (w,h) and our own format = (h, w)
        img_sizes: set[ImgShape] = {imagesize.get(fp)[::-1] for fp in fps}
        if len(img_sizes) > 1:
            raise ValueError(
                f"The images within a single dataset should have equal shapes. "
                f"paths: {fps[:5]} ..., shapes: {img_sizes}"
            )
        self.img_shape: ImgShape = img_sizes.pop()

    def _get_tensor_from_annos(self, annotations: list[dict[str, any]], key: str, dtype: t.dtype) -> t.Tensor:
        """Get all values of key from the annotations as tensor with dtype on the correct device."""
        return t.tensor([d[key] for d in annotations], device=self.device, dtype=dtype).flatten()

    def _get_kps_and_vis(self, d: dict[str, any]) -> tuple[t.Tensor, t.Tensor]:
        """Get the key-points and visibilities from the data."""
        kps, vis = (
            (
                t.tensor(d["keypoints"], device=self.device, dtype=t.float32)
                if len(d["keypoints"])
                else t.zeros((1, self.nof_kps, 3), device=self.device, dtype=t.float32)
            )
            .reshape((1, self.nof_kps, 3))
            .split(split_size=[2, 1], dim=-1)
        )
        return kps.reshape((1, self.nof_kps, 2)), vis.reshape((1, self.nof_kps, 1))

    def _get_bbox(self, d: dict[str, any]) -> tvte.BoundingBoxes:
        """Get the bounding box from the data."""
        if not hasattr(self, "img_shape") or self.img_shape is None:
            raise ValueError("Expected the image shape to be set before calling this function.")

        return tvte.BoundingBoxes(
            t.tensor(d["bbox"], device=self.device, dtype=t.float32),
            format=self.bbox_format,
            canvas_size=self.img_shape,
            dtype=t.float32,
            device=self.device,
        )

    def _get_anno_data(
        self, annos: list[any], anno_ids: list[int]
    ) -> tuple[t.Tensor, t.Tensor, tvte.BoundingBoxes, tuple[FilePath, ...]]:
        """Helper for getting the key-points, visibilities, bboxes, and crop paths from a list of annotation IDs."""
        keypoints: list[t.Tensor] = []
        visibilities: list[t.Tensor] = []
        bboxes: list[tvte.BoundingBoxes] = []
        crop_paths: list[FilePath] = []

        if not hasattr(self, "img_shape") or self.img_shape is None:
            raise ValueError("Expected the image shape to be set before calling this function.")

        for anno_id in anno_ids:
            anno = annos[anno_id]

            kps, visibility = self._get_kps_and_vis(d=anno)
            box = self._get_bbox(d=anno)

            keypoints.append(kps)
            visibilities.append(visibility)
            bboxes.append(box)
            crop_paths.append(anno["crop_path"])

        if len(bboxes) == 0:
            # return empty objects
            return (
                t.empty((0, self.nof_kps, 2)),
                t.empty((0, self.nof_kps, 1)),
                tvte.BoundingBoxes(t.empty((0, 4)), canvas_size=self.img_shape, format=self.bbox_format),
                (),
            )
        return collate_tensors(keypoints), collate_tensors(visibilities), collate_bboxes(bboxes), tuple(crop_paths)


class PoseTrack21_BBox(BBoxDataset, PoseTrack21BaseDataset):
    """Load a single precomputed json file from the |PT21|_ dataset.

    Params
    ------

    id_map (FilePath, optional):
        The (local or absolute) path to a json file containing a mapping from person ID to classifier ID.
        Both IDs are python integers, the IDs of the classifier should be continuous and zero-indexed.
        If the number of classes is required for other parameters, e.g., the size of a classifier,
        the length of this ID map should have the correct value.
        By default, this value is not set or None.
        In case this value is not present, the mapping will be created from scratch as the enumerated sorted person IDs.


    Important Inherited Params
    --------------------------

    dataset_path (FilePath):
        Path to the directory of the dataset.
        The value has to either be a local project path, or a valid absolute path.
    force_img_reshape (bool, optional):
        Whether to accept that images in one folder might have different shapes.
        Default ``DEF_VAL.dataset.force_img_reshape``.
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.len = len(self.json["annotations"])
        map_img_id_frame_id: dict[int, FilePath] = {img["id"]: str(img["frame_id"]) for img in self.json["images"]}

        # precomputed image crops in a specific folder
        crops_dir: FilePath = self.get_path_in_dataset(self.params.get("crops_folder"))

        # create a mapping from person id to (custom) zero-indexed class id or load an existing mapping
        map_pid_to_cid: dict[int, int] = (
            {int(i): int(j) for i, j in read_json(self.params["id_map"]).items()}
            if "id_map" in self.params and self.params["id_map"] is not None
            else {
                int(pid): int(i) for i, pid in enumerate(sorted(set(a["person_id"] for a in self.json["annotations"])))
            }
        )
        # save the image-, person-, and class-ids for later use as torch tensors
        frame_id_list: list[int] = []
        cid_list: list[int] = []

        for anno in self.json["annotations"]:
            frame_id_list.append(int(map_img_id_frame_id[anno["image_id"]]))
            cid_list.append(int(map_pid_to_cid[int(anno["person_id"])]))
            # add image and crop filepaths
            anno["img_path"] = self.map_img_id_to_img_path[anno["image_id"]]
            anno["crop_path"] = os.path.join(
                crops_dir,
                self._get_dataset_name_from_img_path(anno["img_path"]),  # dataset name
                f"{anno['image_id']}_{str(anno['person_id'])}.jpg",
            )

        self.img_ids: t.Tensor = self._get_tensor_from_annos(self.json["annotations"], key="image_id", dtype=t.long)
        self.pids: t.Tensor = self._get_tensor_from_annos(self.json["annotations"], key="person_id", dtype=t.long)
        self.frame_ids: t.Tensor = t.tensor(frame_id_list, dtype=t.long, device=self.device)
        self.cids: t.Tensor = t.tensor(cid_list, dtype=t.long, device=self.device)

        # as np.ndarray to not store large python objects
        self.data: np.ndarray[dict[str, any]] = np.asarray(self.json["annotations"])
        del self.json

    def __len__(self) -> int:
        return self.len

    def arbitrary_to_ds(self, a: dict, idx: int) -> State:
        """Convert raw PoseTrack21 annotations to a :class:`State` object."""
        keypoints, visibility = self._get_kps_and_vis(d=a)

        ds = State(
            validate=False,  # This is given PT21 data, no need to validate...
            device=self.device,
            filepath=(a["img_path"],),
            bbox=self._get_bbox(d=a),
            keypoints=keypoints,
            person_id=self.pids[idx],
            # custom values
            class_id=self.cids[idx],
            crop_path=(a["crop_path"],),
            # additional values which might not be required
            joint_weight=visibility,
            image_id=self.img_ids[idx],
            skeleton_name=(self.skeleton_name,),
            frame_id=self.frame_ids[idx],
        )
        # make sure to get the image crop for this State if requested
        if self.params.get("load_img_crops", DEF_VAL["dataset"]["pt21"]["load_img_crops"]):
            self.get_image_crops(ds)
        return ds

    # def __getitems__(self, indices: list[int]) -> State:
    #     """Get a batch of items at once from the dataset. Does only work for non concatenated datasets."""


class PoseTrack21_Image(ImageDataset, PoseTrack21BaseDataset):
    """Load a single precomputed json file from the |PT21| dataset where every index represents one image.
    Every getitem call therefore returns a :class:`.State` object,
    containing zero or more bounding-boxes of people detected on this image.

    Params
    ------

    id_map (FilePath, optional):
        The (local or absolute) path to a json file containing a mapping from person ID to classifier ID.
        Both IDs are python integers, the IDs of the classifier should be continuous and zero-indexed.
        If the number of classes is required for other parameters, e.g., the size of a classifier,
        the length of this ID map should have the correct value.
        By default, this value is not set or None.
        In case this value is not present, the mapping will be created from scratch as the enumerated sorted person IDs.
    load_img_crops (bool, optional):
        Whether to load the image crops during the __getitem__ call.
        Default ``DEF_VAL["dataset"]["pt21"]["load_img_crops"]``.

    Important Inherited Params
    --------------------------

    dataset_path (FilePath):
        Path to the directory of the dataset.
        The value has to either be a local project path, or a valid absolute path.
    force_img_reshape (bool, optional):
        Whether to accept that images in one folder might have different shapes.
        Default ``DEF_VAL.dataset.force_img_reshape``.
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.len = len(self.json["images"])

        # create a mapping from person id to (custom) zero-indexed class id or load an existing mapping
        map_pid_to_cid: dict[int, int] = (
            {int(i): int(j) for i, j in read_json(self.params["id_map"]).items()}
            if "id_map" in self.params and self.params["id_map"] is not None
            else {
                int(pid): int(i) for i, pid in enumerate(sorted(set(a["person_id"] for a in self.json["annotations"])))
            }
        )

        # create a mapping from image id to a list of all annotations
        self.map_img_id_to_anno_ids: dict[int, list[int]] = {int(img["id"]): [] for img in self.json["images"]}

        cid_list: list[int] = []

        for anno_id, anno in enumerate(self.json["annotations"]):
            img_id = int(anno["image_id"])
            # append the ID of the current annotation to the annotation-list of the respective image
            self.map_img_id_to_anno_ids[img_id].append(anno_id)
            # save the image-, person-, and class-ids for later use as torch tensors
            pid = int(anno["person_id"])
            cid_list.append(map_pid_to_cid[pid])
            # add the crop path to annotation
            anno["crop_path"] = os.path.join(
                self.crops_dir,
                self._get_dataset_name_from_img_path(self.map_img_id_to_img_path[img_id]),
                # do not use int(anno["image_id"]), because it might remove leading zeros
                f"{str(anno['image_id'])}_{str(anno['person_id'])}.jpg",  # int() might remove leading zeros
            )

        self.img_ids: t.Tensor = self._get_tensor_from_annos(self.json["annotations"], key="image_id", dtype=t.long)
        self.pids: t.Tensor = self._get_tensor_from_annos(self.json["annotations"], key="person_id", dtype=t.long)
        self.cids: t.Tensor = t.tensor(cid_list, dtype=t.long, device=self.device)

        # store as np.ndarray to not store large python objects
        self.data: np.ndarray[dict[str, any]] = np.asarray(self.json["images"])
        self.annos: np.ndarray[dict[str, any]] = np.asarray(self.json["annotations"])
        del self.json

    def __len__(self) -> int:
        return self.len

    def arbitrary_to_ds(self, a: dict, idx: int) -> State:
        """Convert raw PoseTrack21 annotations to a :class:`State` object."""
        img_id: int = int(a["id"])
        anno_ids: list[int] = self.map_img_id_to_anno_ids[img_id]

        keypoints, visibilities, bboxes, crop_paths = self._get_anno_data(annos=self.annos, anno_ids=anno_ids)

        ds = State(
            validate=False,  # This is given PT21 data, no need to validate...
            device=self.device,
            # add filepath to tuple even though there is no data to be able to draw the image later
            filepath=tuple(self.map_img_id_to_img_path[img_id] for _ in range(max(len(anno_ids), 1))),
            bbox=bboxes,
            keypoints=keypoints,
            person_id=self.pids[anno_ids].flatten(),
            # custom values
            class_id=self.cids[anno_ids].flatten(),
            crop_path=crop_paths,
            joint_weight=visibilities,
            skeleton_name=tuple(self.skeleton_name for _ in range(len(anno_ids))),
            # optional values
            # Add at least one value for image and frame ID, to be able to generate the results later!
            image_id=t.ones(max(len(anno_ids), 1), device=self.device, dtype=t.long) * img_id,
            frame_id=t.ones(max(len(anno_ids), 1), device=self.device, dtype=t.long) * img_id,
        )
        # make sure to get the image crop for this State if requested
        if self.params.get("load_img_crops", DEF_VAL["dataset"]["pt21"]["load_img_crops"]):
            self.get_image_crops(ds)
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
        self.anno_dir: FilePath = os.path.join(self.dataset_dir, "./posetrack_person_search/")

        # image (crop) directory
        train_dir: FilePath = os.path.join(self.dataset_dir, "crops/train")
        query_dir: FilePath = os.path.join(self.dataset_dir, "crops/query")
        gallery_dir: FilePath = os.path.join(self.dataset_dir, "crops/gallery")

        train: list[tuple]
        query: list[tuple]
        gallery: list[tuple]

        if self.instance == "images":
            train = self.process_file("train.json", train_dir, relabel=True)
            query = self.process_file("query.json", query_dir, cam_id=1)
            gallery = self.process_file("val.json", gallery_dir)
        elif self.instance == "key_points":
            train = self.process_file("train.json", train_dir, relabel=True, is_kp=True)
            query = self.process_file("query.json", query_dir, cam_id=1, is_kp=True)
            gallery = self.process_file("val.json", gallery_dir, is_kp=True)
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

    # pylint: disable=too-many-arguments
    def process_file(
        self, filepath: FilePath, crops_dir: FilePath, relabel: bool = False, cam_id: int = 0, is_kp: bool = False
    ) -> list[tuple[str, int, int, int]]:  # pragma: no cover
        """Process all the data in a single directory.

        Args:
            filepath (FilePath): The absolute path to the json file containing the annotations and image paths.
                In this case will be something like '.../data/PoseTrack21/posetrack_person_search/train.json'.
            crops_dir (FilePath): The absolute path to the directory containing the image crops.
                In this case will be something like '.../data/PoseTrack21/crops/train/'.
            relabel (bool, optional): Whether to create labels from to pids,
                to reduce the number of parameters in the model. Default False.
            cam_id (int, optional): The id of the camera to use.
                The cam_id of the query dataset has to be different from the cam_id of the gallery,
                see `this issue <https://github.com/KaiyangZhou/deep-person-reid/issues/442#issuecomment-868757430>`_
                for more details.
                Default 0.
            is_kp (bool, optional): Whether the files that should be loaded are key point or image files.
                Default False, means image files ('.jpg').

        Returns:
            data (list[tuple[str, int, int, int]]): A list of tuples containing the absolute image path,
                person id (label), camera id, and dataset id.
                The dataset id is the video_id with a leading 1 for mpii and 2 for bonn, to remove duplicates.
        """
        json: dict[str, list[dict[str, any]]] = read_json(os.path.join(self.anno_dir, filepath))

        map_img_id_path: dict[int, FilePath] = {
            img["id"]: to_abspath(os.path.join(self.dataset_dir, str(img["file_name"]))) for img in json["images"]
        }

        pid_container: set = set(int(anno["person_id"]) for anno in json["annotations"])
        pid_container -= set(self._junk_pids)  # junk images are just ignored
        pid2label: dict[int, int] = {pid: label for label, pid in enumerate(pid_container)}

        data: list[tuple[str, int, int, int]] = []
        # (path, pid, camid, dsetid)
        # path: is the absolute path to the file of the cropped image
        # pid: person id
        # camid: id of the camera = 0 for all train and gallery images; 1 for all in query
        # dsetid: dataset id = video_id with a leading 1 for mpii and 2 for bonn

        for anno in json["annotations"]:
            pid = anno["person_id"]
            if pid in self._junk_pids:
                continue  # junk images are just ignored

            ds_name = re.split(r"[\\/]", map_img_id_path[anno["image_id"]])[-2]
            crop_path = os.path.join(crops_dir, ds_name, f"{anno['image_id']}_{str(pid)}.{'pt' if is_kp else 'jpg'}")
            if relabel:
                pid = pid2label[pid]
            # create dsetid as int({"1" if ds_type == "mpii" else "2"}{video_id})
            if "_" not in ds_name:
                dsetid: int = 0
            else:
                ds_id, ds_type, *_ = ds_name.split("_")
                dsetid: int = int(f"{'1' if ds_type == 'mpii' else '2'}{str(ds_id)}")
            data.append((crop_path, pid, cam_id, dsetid))
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


class PoseTrack21_ImageHistory(ImageHistoryDataset, PoseTrack21BaseDataset):
    """A |PT21|_ dataset that creates combined states from a current state and its history."""

    data: np.ndarray[dict[str, any]]
    """A dict mapping the """

    annos: np.ndarray[dict[str, any]]

    def __init__(self, config: Config, path: NodePath):
        PoseTrack21BaseDataset.__init__(self=self, config=config, path=path)
        ImageHistoryDataset.__init__(self=self, config=config, path=path)

        # create a mapping from person id to (custom) zero-indexed class id or load an existing mapping
        map_pid_to_cid: dict[int, int] = (
            {int(i): int(j) for i, j in read_json(self.params["id_map"]).items()}
            if "id_map" in self.params and self.params["id_map"] is not None
            else {
                int(pid): int(i) for i, pid in enumerate(sorted(set(a["person_id"] for a in self.json["annotations"])))
            }
        )

        # create a mapping from image id to a list of all annotations
        self.map_img_id_to_anno_ids: dict[int, list[int]] = {int(img["id"]): [] for img in self.json["images"]}

        cid_list: list[int] = []
        for anno_id, anno in enumerate(self.json["annotations"]):
            img_id = int(anno["image_id"])
            # append the ID of the current annotation to the annotation-list of the respective image
            self.map_img_id_to_anno_ids[img_id].append(anno_id)
            # save the image-, person-, and class-ids for later use as torch tensors
            cid_list.append(map_pid_to_cid[int(anno["person_id"])])
            # add the crop path to annotation
            anno["crop_path"] = os.path.join(
                self.crops_dir,
                f"./{self._get_dataset_name_from_img_path(self.map_img_id_to_img_path[img_id])}/",
                # do not use int(anno["image_id"]), because it might remove leading zeros
                f"{str(anno['image_id'])}_{str(anno['person_id'])}.jpg",
            )

        self.img_ids: t.Tensor = self._get_tensor_from_annos(self.json["annotations"], key="image_id", dtype=t.long)
        self.pids: t.Tensor = self._get_tensor_from_annos(self.json["annotations"], key="person_id", dtype=t.long)
        self.cids: t.Tensor = t.tensor(cid_list, dtype=t.long, device=self.device)

        # store as np.ndarray to not store large python objects
        # don't add images without labels
        self.data: np.ndarray[dict[str, any]] = np.asarray(
            [img for img in self.json["images"] if img["has_labeled_person"]]
        )
        self.annos: np.ndarray[dict[str, any]] = np.asarray(self.json["annotations"])
        del self.json

    def __len__(self) -> int:
        """Force usage of the ImageHistoryDataset.__len__ method."""
        return ImageHistoryDataset.__len__(self=self)

    def __getitem__(self, idx: int) -> list[State]:
        """Force usage of the ImageHistoryDataset.__getitem__ method."""
        return ImageHistoryDataset.__getitem__(self=self, idx=idx)

    def __getitems__(self, indices: list[int]) -> list[State]:
        """Force usage of the ImageHistoryDataset.__getitems__ method."""
        return ImageHistoryDataset.__getitems__(self=self, indices=indices)

    def arbitrary_to_ds(self, a: list[any], idx: int) -> list[State]:
        """Convert raw PoseTrack21 annotations to a list of :class:`State` objects."""
        img_ids: list[int] = [int(a_i["id"]) for a_i in a]
        if len(img_ids) == 0:
            raise NotImplementedError(
                f"No image ids given for {idx} in dataset: {self._get_dataset_name_from_img_path(a[0]['crop_path'])}\n"
                f"states: {a}"
            )
        states = []
        for img_id in img_ids:
            anno_ids: list[int] = self.map_img_id_to_anno_ids[img_id]
            if len(anno_ids) == 0:
                raise NotImplementedError(f"There are no annotations for image id: {img_id}")

            keypoints, visibilities, bboxes, crop_paths = self._get_anno_data(annos=self.annos, anno_ids=anno_ids)

            states.append(
                State(
                    validate=False,  # This is given PT21 data, no need to validate...
                    device=self.device,
                    # add filepath to tuple even though there is no data to be able to draw the image later
                    filepath=tuple(self.map_img_id_to_img_path[img_id] for _ in range(max(len(anno_ids), 1))),
                    bbox=bboxes,
                    keypoints=keypoints,
                    person_id=(self.pids[anno_ids].flatten() if len(anno_ids) > 0 else t.empty(0, device=self.device)),
                    # custom values
                    class_id=(self.cids[anno_ids].flatten() if len(anno_ids) > 0 else t.empty(0, device=self.device)),
                    crop_path=crop_paths,
                    joint_weight=visibilities,
                    # optional values
                )
            )
        return states
