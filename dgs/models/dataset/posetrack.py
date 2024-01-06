"""
Load bboxes and poses from an existing .json file of the |PT21|_ dataset.

See https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md#reid-pose-tracking for type definitions.


PoseTrack21 format:

* Bounding boxes have format XYWH
* The 17 key points and their respective visibilities are stored in one list of len 51 [x_i, y_i, vis_i, ...]
"""
import os
import warnings

import imagesize
import torch
import torchvision.transforms.v2 as tvt
from torch.utils.data import ConcatDataset, Dataset as TorchDataset
from torchvision import tv_tensors
from torchvision.io import write_jpeg
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset
from dgs.models.states import DataSample
from dgs.utils.files import mkdir_if_missing, read_json, to_abspath
from dgs.utils.image import CustomCropResize, load_image
from dgs.utils.types import Config, Device, FilePath, ImgShape, NodePath, TVImage, Validations
from torchreid.data import Dataset

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


def extract_all_bboxes(
    base_dataset_path: FilePath = "./data/PoseTrack21/",
    anno_dir: FilePath = "./posetrack_data/",
    crop_size: ImgShape = (256, 256),
    transform_mode: str = "zero-pad",
    **kwargs,
) -> None:
    """Given the path to the |PT21| dataset, create a new ``crops`` folder containing the image crops of every
    bounding box separated by test, train, and validation sets like the images.
    Within every set is one folder per video ID, in which then lie the image crops.
    The name of the crops is: ``{person_id}_{image_id}.jpg``.

    Args:
        base_dataset_path (FilePath): The path to the |PT21| dataset directory.
        anno_dir (FilePath): The name of the directory containing the folders for the training and test annotations.
        crop_size (ImgShape): The target shape of the image crops.
        transform_mode (str): Defines the resize mode, has to be in the modes of
            :class:`~dgs.utils.image.CustomToAspect`. Default "zero-pad".

    Keyword Args:
        device (Device): Device to run the cropping on. Defaults to "cuda" if available "cpu" otherwise.
        quality (int): The quality to save the jpegs as. Default 90. Default of torchvision is 75.
        check_img_sizes (bool): Whether to check if all images in a given folder have the same size before stacking them
            for cropping. Default False.
        load_image (dict[str, any]): additional kwargs passed to load_image() function. Default {}.
    """
    # pylint: disable=too-many-locals

    base_dataset_path = to_abspath(base_dataset_path)
    crops_path = os.path.join(base_dataset_path, "crops")

    # extract kwargs
    device: Device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")

    mkdir_if_missing(crops_path)

    transform = tvt.Compose(
        [
            tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),
            tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
            CustomCropResize(),
        ]
    )

    for abs_anno_path, _, files in tqdm(os.walk(os.path.join(base_dataset_path, anno_dir)), desc="annos", position=0):
        # skip directories that don't contain files, e.g., the folder containing the datasets
        if len(files) == 0:
            continue
        # inside crops folder, create same structure as within images directory
        # target      => .../PoseTrack21/crops/{train}/{dataset_name}/{img_id}_{person_id}.jpg
        # abs_anno_path => .../PoseTrack21/images/{train}/{dataset_name}.json

        # create folder {train} inside crops
        crops_train_dir = os.path.join(crops_path, abs_anno_path.split("/")[-1])
        mkdir_if_missing(crops_train_dir)

        for anno_file in tqdm(files, desc="annotation-files", position=1):
            if not anno_file.endswith(".json"):
                continue

            # load and validate json
            json = read_json(os.path.join(abs_anno_path, anno_file))
            validate_pt21_json(json)

            # get the folder name which is the name of the sub-dataset and create the folder within crops
            crops_subset_path = os.path.join(crops_train_dir, anno_file.split(".")[0])
            mkdir_if_missing(crops_subset_path)

            # skip if the folder has the correct number of files
            if len(os.listdir(crops_subset_path)) == len(json["annotations"]):
                continue

            # check that the image sizes in every folder match
            if (
                kwargs.get("check_img_sizes", False)
                and len(set(imagesize.get(os.path.join(crops_subset_path, f)) for f in files)) != 1
            ):
                warnings.warn(f"In folder {crops_subset_path} the images do not have the same size.")

            # Because the images in every folder have the same shape,
            # it is possible to stack them and use the batches on the GPU.
            map_id_to_path: dict[int, FilePath] = {i["id"]: i["file_name"] for i in json["images"]}
            img_fps: list[FilePath] = []
            boxes: list = []
            new_fps: list[FilePath] = []

            for anno in json["annotations"]:
                boxes.append(torch.tensor(anno["bbox"], dtype=torch.float32, device=device))

                img_fps.append(os.path.join(base_dataset_path, map_id_to_path[anno["image_id"]]))

                # There will be multiple detections per image.
                # Therefore, the new image crop name has to include the image id and person id.
                new_img_name = f"{anno['image_id']}_{str(anno['person_id'])}.jpg"
                new_fps.append(os.path.join(crops_subset_path, new_img_name))

            del json

            imgs: TVImage = load_image(
                filepath=tuple(img_fps),
                device=device,
                requires_grad=False,
                **kwargs.get("load_image", {}),
            )

            # pass original images through CustomResizeCrop transform and get the resulting image crops
            crops = transform(
                {
                    "image": imgs,
                    "box": tv_tensors.BoundingBoxes(
                        torch.stack(boxes), format="XYWH", canvas_size=imgs.shape[-2:], device=device
                    ),
                    "keypoints": torch.zeros((imgs.shape[-4], 1, 2), device=device),
                    "mode": transform_mode,
                    "output_size": crop_size,
                }
            )["image"]

            for fp, crop in zip(new_fps, crops):
                write_jpeg(input=crop, filename=fp, quality=kwargs.get("quality", 90))


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
        super(BaseDataset, self).__init__(config=config, path=path)

    def arbitrary_to_ds(self, a) -> DataSample:
        raise NotImplementedError


class PoseTrack21JSON(BaseDataset):
    """Load a single precomputed json file."""

    def __init__(self, config: Config, path: NodePath, json_path: FilePath = None) -> None:
        super(BaseDataset, self).__init__(config=config, path=path)

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


class PoseTrack21Torchreid(Dataset):
    r"""Load PoseTrack21 as torchreid dataset.

    Reference:
        Doering et al. Posetrack21: A dataset for person search, multi-object tracking and multi-person pose tracking.
        IEEE / CVF 2022.

    URL: `<https://github.com/andoer/PoseTrack21>`_

    Dataset statistics:
        - identities: The training set contains 5474 unique person ids.
        - images: 163411 images, divided into: 96215 train, 46751 test (gallery), and 20444 val (query)

    Keyword Args:
        transform_mode: Defines the resize mode, has to be in the modes of
            :class:`~dgs.utils.image.CustomToAspect`. Default "zero-pad".
    """

    dataset_dir = "PoseTrack21"

    def __init__(self, root="", **kwargs):
        # get values from init call
        self.transform_mode = kwargs.get("transform_mode", "zero-pad")
        self.height = kwargs.get("height", 256)
        self.width = kwargs.get("width", 256)

        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        # annotation directory
        self.annotation_dir = os.path.join(self.dataset_dir, "posetrack_data")

        # image directory
        train_dir = os.path.join(self.dataset_dir, "images/train")
        query_dir = os.path.join(self.dataset_dir, "images/val")
        gallery_dir = os.path.join(self.dataset_dir, "images/val")  # fixme there are no annotations for test obviously

        train = self.process_dir(train_dir)
        query = self.process_dir(query_dir)
        gallery = self.process_dir(gallery_dir)

        self.check_before_run([self.dataset_dir, train_dir, query_dir, gallery_dir])

        super().__init__(train, query, gallery, **kwargs)

        self.transform = tvt.Compose(
            [
                tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),
                tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
                # normalize the images using imagenet mean and std
                tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
                CustomCropResize(),
            ]
        )

    def __getitem__(self, index: int):
        img_path, pid, camid, dsetid, box = self.data[index]

        # in torchreid img is expected to be a FloatTensor # fixme, or is it model dependent?
        try:
            img = load_image(
                filepath=os.path.join(self.dataset_dir, img_path),
                dtype=torch.float32,
                requires_grad=False,
            )
        except (ValueError, RuntimeError):
            # only reshape if necessary
            img = load_image(
                filepath=os.path.join(self.dataset_dir, img_path),
                dtype=torch.float32,
                force_reshape=True,
                mode=self.transform_mode,
                output_size=(1000, 1000),
                requires_grad=False,
            )

        data = {
            "image": img,
            "box": tv_tensors.BoundingBoxes(box, format="XYWH", canvas_size=img.shape[-2:]),
            "keypoints": torch.zeros((1, 1, 2)),
            "mode": self.transform_mode,
            "output_size": (self.height, self.width),
        }
        crop = self.transform(data)["image"].squeeze()  # expects 3D tensor [C, H, W]

        item = {"img": crop, "pid": pid, "camid": camid, "impath": img_path, "dsetid": dsetid}
        return item

    def process_dir(self, dir_path: FilePath) -> list[tuple]:
        """Process all the data of one directory"""
        anno_dir = os.path.join(self.annotation_dir, dir_path.split("/")[-1])
        data: list[tuple[str, int, int, int, torch.Tensor]] = []
        # (path, pid, camid, dsetid, box)
        # path: is the path to the image file
        # pid: person id
        # camid: id of the camera = 0 for all videos
        # dsetid: dataset id = video_id with a leading 1 for mpii and 2 for bonn
        # box: Bounding Box around the human as regular tensor with format XYWH

        for _, _, files in os.walk(anno_dir):
            for file in files:
                file_id, source, *_ = str(file).split("_")
                if not file.endswith(".json"):
                    continue
                json = read_json(os.path.join(anno_dir, file))
                map_id_to_path: dict[int, str] = {i["id"]: i["file_name"] for i in json["images"]}
                for anno in json["annotations"]:
                    fp: str = map_id_to_path[anno["image_id"]]
                    pid: int = anno["person_id"]
                    dsetid: int = int(("1" if source == "mpii" else "2") + str(file_id))
                    box: torch.Tensor = torch.tensor(anno["bbox"])
                    data.append((fp, pid, 0, dsetid, box))
        print(f"dir: {dir_path}, max person id: {max(d[1] for d in data)}")
        return data

    def download_dataset(self, dataset_dir, dataset_url) -> None:
        raise NotImplementedError("See https://github.com/andoer/PoseTrack21 for more details.")
