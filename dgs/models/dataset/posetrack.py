"""
Load bboxes and poses from an existing .json file of the PoseTrack21 dataset.

See https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md#reid-pose-tracking for type definitions.


PoseTrack21 format:

* Bounding boxes have format XYWH
* The 17 key points and their respective visibilities are stored in one list of len 51 [x_i, y_i, vis_i, ...]
"""
import os

import imagesize
import torch
from torch.utils.data import ConcatDataset, Dataset as TorchDataset
from torchvision import tv_tensors
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset
from dgs.models.states import DataSample
from dgs.utils.files import read_json, to_abspath
from dgs.utils.types import Config, FilePath, ImgShape, NodePath, Validations

pt21_json_validations: Validations = {"path": [None]}


def validate_pt21_json(json: dict) -> None:
    """Check whether the given json is valid for the PoseTrack21 dataset."""
    if not isinstance(json, dict):
        raise ValueError(f"The PoseTrack21 json file is expected to be a dict, but was {type(json)}")
    if "images" not in json:
        raise KeyError(f"It is expected that a PoseTrack21 .json file has an images key. But keys are {json.keys()}")
    if "annotations" not in json:
        raise KeyError(
            f"It is expected that a PoseTrack21 .json file has an annotations key. But keys are {json.keys()}"
        )


def get_pose_track_21(config: Config, path: NodePath) -> TorchDataset:
    """Load PoseTrack JSON files.

    The path parameter can be one of the following:

    * a path to a directory
    * a single json filepath
    * a list of json filepaths

    In all cases, the path can be
        a global path,
        a path relative to the package,
        or a local path under the dataset_path directory.

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
    """Non-Abstract class for PoseTrack21 dataset."""

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
