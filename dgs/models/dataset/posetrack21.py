r"""
Load bboxes and poses from an existing .json file of the |PT21|_ dataset.

See https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md#reid-pose-tracking for type definitions.


PoseTrack21 format:

* Bounding boxes have format XYWH
* The 17 key points and their respective visibilities are stored in one list of len 51.
  The list contains the x- and y-coordinate and the visibility: \n
  [``x``\ :sub:`i`, ``y``\ :sub:`i`, ``vis``\ :sub:`i`, ...]
"""

import glob
import os
import shutil
import warnings
from typing import Type, Union

import imagesize
import numpy as np
import torch as t
from torch.utils.data import ConcatDataset, Dataset as TorchDataset
from torchvision import tv_tensors as tvte
from torchvision.transforms.v2.functional import convert_bounding_box_format
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset, BBoxDataset, dataloader_validations, ImageDataset
from dgs.models.dataset.torchreid_pose_dataset import TorchreidPoseDataset
from dgs.utils.config import DEF_VAL
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import mkdir_if_missing, read_json, to_abspath, write_json
from dgs.utils.state import collate_bboxes, collate_tensors, State
from dgs.utils.types import Config, Device, FilePath, ImgShape, NodePath, Validations
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

# Do not allow import of 'PoseTrack21' base dataset
__all__ = [
    "validate_pt21_json",
    "get_pose_track_21",
    "PoseTrack21_BBox",
    "PoseTrack21_Image",
    "PoseTrack21Torchreid",
    "generate_pt21_submission_file",
    "submission_data_from_state",
]

pt21_json_validations: Validations = {
    "data_path": [("any", [str, ("all", [list, ("forall", str)])])],
    "crops_folder": [str, ("folder exists", False)],
    # optional
    "id_map": ["optional", str],
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


def submission_data_from_state(s: State) -> tuple[dict[str, any], list[dict[str, any]]]:
    """Given a :class:`.State`, extract data for the 'images' and 'annotations' list used in the pt21 submission.

    See :func:`.generate_pt21_submission_file` for more details on the submission format.

    Returns:
        The image and annotation data as dictionaries.
        The annotation data is a list of dicts, because every image can have multiple detections / annotations.
    """
    # pylint: disable=too-many-branches
    # validate the image data
    for key in ["filepath", "image_id", "frame_id"]:
        if key not in s:
            raise KeyError(f"Expected key '{key}' to be in State.")
        if isinstance(s[key], str):
            # str -> tuple of str, this will always be correct, add at least one value for later usage
            s[key] = (s[key] for _ in range(max(1, s.B)))
        elif s.B > 1:
            if (l := len(s[key])) != s.B:
                raise ValueError(f"Expected '{key}' ({l}) to have the same length as the State ({s.B}).")
            if any(s[key][i] != s[key][0] for i in range(1, s.B)):
                raise ValueError(f"State has different {key}s, expected all {key}s to match. got: '{s[key]}'.")
        elif (l := len(s[key])) != 1:
            raise ValueError(f"Expected '{key}' ({l}) to have a length of exactly 1.")
    # add frame id if missing as duplicate of image id
    if "frame_id" not in s:
        s["frame_id"] = s["image_id"]

    # get the image data
    image_data = {
        "file_name": s.filepath[0],
        "id": int(s["image_id"][0].item() if isinstance(s["image_id"], t.Tensor) else s["image_id"][0]),
        "frame_id": int(s["frame_id"][0].item() if isinstance(s["frame_id"], t.Tensor) else s["frame_id"][0]),
    }
    if s.B == 0:
        return image_data, []

    # validate the annotation data
    for key in ["person_id", "pred_tid", "bbox", "keypoints", "joint_weight"]:
        if key not in s:
            raise KeyError(f"Expected key '{key}' to be in State.")
        if (l := len(s[key])) != s.B:
            raise ValueError(f"Expected '{key}' ({l}) to have the same length as the State ({s.B}).")

    # get the annotation data
    anno_data = []
    bboxes = convert_bounding_box_format(s.bbox, new_format=tvte.BoundingBoxFormat.XYWH)
    for i in range(s.B):
        kps = t.cat([s.keypoints[i], s.joint_weight[i]], dim=-1)
        anno_data.append(
            {
                "bboxes": bboxes[i].flatten().tolist(),
                "kps": kps.flatten().tolist(),
                "scores": s["scores"][i].flatten().tolist() if "scores" in s else [0.0 for _ in range(17)],
                "image_id": int(s["image_id"][0].item() if isinstance(s["image_id"], t.Tensor) else s["image_id"][0]),
                "person_id": int(s.person_id[i].item()),
                "track_id": int(s["pred_tid"][i].item()),
            }
        )

    return image_data, anno_data


def generate_pt21_submission_file(
    outfile: FilePath, images: list[dict[str, any]], annotations: list[dict[str, any]]
) -> None:  # pragma: no cover
    """Given data, generate a |PT21| submission file.

    References:
        https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md

        https://github.com/leonid-pishchulin/poseval

    Args:
        outfile: The path to the target file
        images: A list containing the IDs and file names of the images
        annotations: A list containing the per-bbox predicted annotations.

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
    data = {"images": images, "annotations": annotations}
    try:
        write_json(obj=data, filepath=outfile)
    except TypeError as e:
        print(f"images: {images}")
        print(f"annotations: {annotations}")
        raise TypeError from e


def get_pose_track_21(config: Config, path: NodePath, ds_name: str = "bbox") -> Union[BaseDataset, TorchDataset]:
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
        ds_name (str): Name of the dataset type to use.
            Either "image" for :class:`.PoseTrack21_Image` or "bbox" for :class:`.PoseTrack21_BBox` .

    Returns:
        An instance of TorchDataset, containing the requested dataset(s) as concatenated torch dataset.
    """
    ds = PoseTrack21(config, path)
    ds.validate_params(pt21_json_validations)
    ds.validate_params(dataloader_validations)

    ds_type: Union[Type[PoseTrack21_Image], Type[PoseTrack21_BBox]] = (
        PoseTrack21_Image if ds_name == "image" else PoseTrack21_BBox
    )

    if isinstance(data_path := ds.params["data_path"], (list, tuple)):
        print(f"Loading list of datasets from {os.path.normpath(ds.params['dataset_path'])}, paths: {data_path}")
        return ConcatDataset(
            [ds_type(config=config, path=path, data_path=ds.get_path_in_dataset(path=p)) for p in tqdm(data_path)]
        )

    # path is either directory or single json file
    paths: list[FilePath]
    abs_path: FilePath = ds.get_path_in_dataset(ds.params["data_path"])
    if os.path.isfile(abs_path):
        paths = [abs_path]
    else:
        paths = [
            os.path.normpath(os.path.join(abs_path, child_path))
            for child_path in os.listdir(abs_path)
            if child_path.endswith(".json")
        ]
        paths.sort()  # make sure systems behave similarly

    if len(paths) == 1:
        print(f"Loading dataset: {paths[0]}")
        return ds_type(config=config, path=path, data_path=paths[0])

    return ConcatDataset(
        [
            ds_type(config=config, path=path, data_path=p)
            for p in tqdm(paths, desc=f"Loading datasets: {os.path.normpath(ds.params['dataset_path'])}")
        ]
    )


class PoseTrack21(BaseDataset):
    """Non-Abstract class for PoseTrack21 dataset to be able to initialize it in :func:`get_pose_track_21`.

    Should not be instantiated.
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

    def arbitrary_to_ds(self, a, idx: int) -> State:
        raise NotImplementedError


class PoseTrack21_BBox(BBoxDataset):
    """Load a single precomputed json file from the |PT21| dataset.

    Params
    ------

    data_path (FilePath):
        The path to the json file, either from within the ``dataset_path`` directory, or as absolute path.
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

    def __init__(self, config: Config, path: NodePath, data_path: FilePath = None) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(pt21_json_validations)

        # validate and get the path to the json
        if data_path is None:
            data_path: FilePath = self.get_path_in_dataset(self.params["data_path"])

        # validate and get json data
        json: dict[str, list[dict[str, any]]] = read_json(data_path)
        validate_pt21_json(json)
        self.len = len(json["annotations"])

        # create a mapping from image id to full filepath
        map_img_id_path: dict[int, FilePath] = {
            img["id"]: to_abspath(os.path.join(self.params["dataset_path"], str(img["file_name"])))
            for img in json["images"]
        }
        map_img_id_frame_id: dict[int, FilePath] = {img["id"]: str(img["frame_id"]) for img in json["images"]}

        # imagesize.get() output = (w,h) and our own format = (h, w)
        img_sizes: set[ImgShape] = {imagesize.get(fp)[::-1] for fp in map_img_id_path.values()}
        if self.params.get("force_img_reshape", DEF_VAL["dataset"]["force_img_reshape"]):
            # take the biggest value of every dimension
            self.img_shape: ImgShape = (max(size[0] for size in img_sizes), max(size[1] for size in img_sizes))
        else:
            if len(img_sizes) > 1:
                raise ValueError(
                    f"The images within a single dataset should have equal shapes. "
                    f"data_path: {data_path}, shapes: {img_sizes}"
                )
            self.img_shape: ImgShape = img_sizes.pop()

        # precomputed image crops in a specific folder
        crops_dir: FilePath = self.get_path_in_dataset(self.params.get("crops_folder"))

        # create a mapping from person id to (custom) zero-indexed class id or load an existing mapping
        map_pid_to_cid: dict[int, int] = (
            {int(i): int(j) for i, j in read_json(self.params["id_map"]).items()}
            if "id_map" in self.params and self.params["id_map"] is not None
            else {int(pid): int(i) for i, pid in enumerate(sorted(set(a["person_id"] for a in json["annotations"])))}
        )
        # save the image-, person-, and class-ids for later use as torch tensors
        img_id_list: list[int] = []
        frame_id_list: list[int] = []
        pid_list: list[int] = []
        cid_list: list[int] = []

        for anno in json["annotations"]:
            img_id_list.append(int(anno["image_id"]))
            frame_id_list.append(int(map_img_id_frame_id[anno["image_id"]]))
            pid_list.append(int(anno["person_id"]))
            cid_list.append(int(map_pid_to_cid[int(anno["person_id"])]))
            # add image and crop filepaths
            anno["img_path"] = map_img_id_path[anno["image_id"]]
            anno["crop_path"] = os.path.join(
                crops_dir,
                anno["img_path"].split("/")[-2],  # dataset name
                f"{anno['image_id']}_{str(anno['person_id'])}.jpg",
            )

        self.img_ids: t.Tensor = t.tensor(img_id_list, dtype=t.long, device=self.device)
        self.frame_ids: t.Tensor = t.tensor(frame_id_list, dtype=t.long, device=self.device)
        self.pids: t.Tensor = t.tensor(pid_list, dtype=t.long, device=self.device)
        self.cids: t.Tensor = t.tensor(cid_list, dtype=t.long, device=self.device)

        # as np.ndarray to not store large python objects
        self.data: np.ndarray[dict[str, any]] = np.asarray(json["annotations"])
        self.skeleton_name = "coco"

    def __len__(self) -> int:
        return self.len

    def arbitrary_to_ds(self, a: dict, idx: int) -> State:
        """Convert raw PoseTrack21 annotations to a :class:`State` object."""
        keypoints, visibility = (
            (
                t.tensor(a["keypoints"], device=self.device, dtype=t.float32)
                if len(a["keypoints"])
                else t.zeros((1, 17, 3), device=self.device, dtype=t.float32)
            )
            .reshape((1, 17, 3))
            .split([2, 1], dim=-1)
        )
        ds = State(
            validate=False,  # This is given PT21 data, no need to validate...
            device=self.device,
            filepath=(a["img_path"],),
            bbox=tvte.BoundingBoxes(
                t.tensor(a["bbox"]).float(),
                format="XYWH",
                dtype=t.float32,
                canvas_size=self.img_shape,
                device=self.device,
            ),
            keypoints=keypoints,
            person_id=self.pids[idx],
            # custom values
            class_id=self.cids[idx],
            crop_path=(a["crop_path"],),
            # additional values which are not required
            joint_weight=visibility,
            image_id=self.img_ids[idx],
            skeleton_name=(self.skeleton_name,),
            frame_id=self.frame_ids[idx],
        )
        # make sure to get the image crop for this State
        self.get_image_crops(ds)
        return ds


class PoseTrack21_Image(ImageDataset):
    """Load a single precomputed json file from the |PT21| dataset where every index represents one image.
    Every getitem call therefore returns a :class:`.State` object,
    containing zero or more bounding-boxes of people detected on this image.

    Params
    ------

    data_path (FilePath):
        The path to the json file, either from within the ``dataset_path`` directory, or as absolute path.
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

    def __init__(self, config: Config, path: NodePath, data_path: FilePath = None) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(pt21_json_validations)

        # validate and get the path to the json
        if data_path is None:
            data_path: FilePath = self.get_path_in_dataset(self.params["data_path"])

        # validate and get json data
        json: dict[str, list[dict[str, any]]] = read_json(data_path)
        validate_pt21_json(json)
        self.len = len(json["images"])

        # create a mapping from image id to full filepath
        self.map_img_id_to_path: dict[int, FilePath] = {
            img["id"]: to_abspath(os.path.join(self.params["dataset_path"], str(img["file_name"])))
            for img in json["images"]
        }

        # imagesize.get() output = (w,h) and our own format = (h, w)
        img_sizes: set[ImgShape] = {imagesize.get(fp)[::-1] for fp in self.map_img_id_to_path.values()}
        if self.params.get("force_img_reshape", DEF_VAL["dataset"]["force_img_reshape"]):
            # take the biggest value of every dimension
            self.img_shape: ImgShape = (max(size[0] for size in img_sizes), max(size[1] for size in img_sizes))
        else:
            if len(img_sizes) > 1:
                raise ValueError(
                    f"The images within a single dataset should have equal shapes. "
                    f"data_path: {data_path}, shapes: {img_sizes}"
                )
            self.img_shape: ImgShape = img_sizes.pop()

        # precomputed image crops in a specific folder
        crops_dir: FilePath = self.get_path_in_dataset(self.params.get("crops_folder"))

        # create a mapping from person id to (custom) zero-indexed class id or load an existing mapping
        map_pid_to_cid: dict[int, int] = (
            {int(i): int(j) for i, j in read_json(self.params["id_map"]).items()}
            if "id_map" in self.params and self.params["id_map"] is not None
            else {int(pid): int(i) for i, pid in enumerate(sorted(set(a["person_id"] for a in json["annotations"])))}
        )

        # create a mapping from image id to a list of all annotations
        self.map_img_id_to_anno_ids: dict[int, list[int]] = {int(img["id"]): [] for img in json["images"]}

        img_id_list: list[int] = []
        pid_list: list[int] = []
        cid_list: list[int] = []

        for anno_id, anno in enumerate(json["annotations"]):
            img_id = int(anno["image_id"])
            pid = int(anno["person_id"])
            # append the ID of the current annotation to the annotation-list of the respective image
            self.map_img_id_to_anno_ids[img_id].append(anno_id)
            # save the image-, person-, and class-ids for later use as torch tensors
            img_id_list.append(img_id)
            pid_list.append(pid)
            cid_list.append(map_pid_to_cid[pid])
            # add the crop path to annotation
            anno["crop_path"] = os.path.join(
                crops_dir,
                self.map_img_id_to_path[img_id].split("/")[-2],  # dataset name
                f"{str(anno['image_id'])}_{str(anno['person_id'])}.jpg",  # int() might remove leading zeros
            )

        self.img_ids: t.Tensor = t.tensor(img_id_list, dtype=t.long, device=self.device)
        self.pids: t.Tensor = t.tensor(pid_list, dtype=t.long, device=self.device)
        self.cids: t.Tensor = t.tensor(cid_list, dtype=t.long, device=self.device)

        # store as np.ndarray to not store large python objects
        self.data: np.ndarray[dict[str, any]] = np.asarray(json["images"])
        self.annos: np.ndarray[dict[str, any]] = np.asarray(json["annotations"])
        self.skeleton_name = "coco"

    def __len__(self) -> int:
        return self.len

    def arbitrary_to_ds(self, a: dict, idx: int) -> State:
        """Convert raw PoseTrack21 annotations to a :class:`State` object."""
        img_id: int = int(a["id"])
        anno_ids: list[int] = self.map_img_id_to_anno_ids[img_id]

        keypoints, visibilities, bboxes, crop_paths = self._get_anno_data(anno_ids)

        ds = State(
            validate=False,  # This is given PT21 data, no need to validate...
            device=self.device,
            # add filepath to tuple even though there is no data to be able to draw the image later
            filepath=tuple(self.map_img_id_to_path[img_id] for _ in range(max(len(anno_ids), 1))),
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
        # make sure to get the image crop for this State
        self.get_image_crops(ds)
        return ds

    def _get_anno_data(
        self, anno_ids: list[int]
    ) -> tuple[t.Tensor, t.Tensor, tvte.BoundingBoxes, tuple[FilePath, ...]]:
        """Helper for getting the key-points, visibilities, bboxes, and crop paths from a list of annotation IDs."""
        keypoints: list[t.Tensor] = []
        visibilities: list[t.Tensor] = []
        bboxes: list[tvte.BoundingBoxes] = []
        crop_paths: list[FilePath] = []

        for anno_id in anno_ids:
            anno = self.annos[anno_id]

            kps, visibility = (
                t.tensor(anno["keypoints"], device=self.device, dtype=t.float32).reshape((1, 17, 3))
                if len(anno["keypoints"])
                else t.zeros((1, 17, 3), device=self.device, dtype=t.float32)
            ).split([2, 1], dim=-1)
            box = tvte.BoundingBoxes(
                t.tensor(anno["bbox"]), format="XYWH", dtype=t.float32, canvas_size=self.img_shape, device=self.device
            )

            keypoints.append(kps.reshape((1, 17, 2)))
            visibilities.append(visibility.reshape((1, 17, 1)))
            bboxes.append(box)
            crop_paths.append(anno["crop_path"])

        if len(bboxes) == 0:
            # return empty objects
            return (
                t.empty((0, 17, 2)),
                t.empty((0, 17, 1)),
                tvte.BoundingBoxes(t.empty((0, 4)), canvas_size=(0, 0), format="XYXY"),
                (),
            )
        return collate_tensors(keypoints), collate_tensors(visibilities), collate_bboxes(bboxes), tuple(crop_paths)


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
            ds_name = map_img_id_path[anno["image_id"]].split("/")[-2]
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
