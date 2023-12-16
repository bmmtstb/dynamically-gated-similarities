"""
Load bboxes and poses from an existing .json file of the PoseTrack21 dataset.

See https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md#reid-pose-tracking for type definitions.
"""
import os

import imagesize
import torch
from torchvision import tv_tensors

from dgs.models.dataset.dataset import BaseDataset
from dgs.models.states import DataSample
from dgs.utils.files import project_to_abspath, read_json
from dgs.utils.types import Config, FilePath, ImgShape, NodePath, Validations

pt21_loader_validations: Validations = {"path": [None]}
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


class PoseTrack21Loader(BaseDataset):
    """Load given PoseTrack JSON files.

    The files can either be:

    * under a given directory
    * a single json filepath
    * a list of json filepaths
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super(BaseDataset, self).__init__(config=config, path=path)

        self.validate_params(pt21_loader_validations)


class PoseTrack21JSON(BaseDataset):
    """Load a single precomputed json file."""

    json: dict[str, dict]
    """The content of the given PoseTrack21 json file."""

    def __init__(self, config: Config, path: NodePath) -> None:
        super(BaseDataset, self).__init__(config=config, path=path)

        self.validate_params(pt21_json_validations)

        # validate and get the path to the json
        path: FilePath = self.get_filepath_in_dataset(self.params["path"])

        # validate and get json data
        self.json = read_json(path)
        validate_pt21_json(self.json)

        # create a mapping from image id to all the information about this image that was provided by PT21
        self.map_img_id: dict[int, dict[str, any]] = {img["image_id"]: img for img in self.json["images"]}
        for k, v in self.map_img_id.items():
            # Add the full file path to the image
            self.map_img_id[k]["file_path"] = project_to_abspath(
                os.path.join(self.params["dataset_path"], str(v["file_name"]))
            )
            # Add original image shape - imagesize output = (w,h) and our own format = (h, w)
            self.map_img_id[k]["img_shape"]: ImgShape = imagesize.get(v["file_path"])[::-1]

        # generate list of data samples
        self.data: list[DataSample] = self.pt21_to_data_sample()
        del self.json

    def pt21_to_data_sample(self) -> list[DataSample]:
        """Convert every detection in PoseTrack21 JSON file to DataSample.

        Returns:
            samples: A list of DataSamples.
        """
        samples: list[DataSample] = []
        for anno in self.json["annotations"]:
            # Within PT21, the 17 key points and their visibility are stored in one list of len 51,
            # with a sorting of: [x_i, y_i, vis_i, ...].
            # By making sure the key points have three dimensions, we reduce the overhead later on.
            keypoints, visibility = torch.split(
                tensor=torch.FloatTensor(anno["keypoints"]).reshape((1, 17, 3)),
                split_size_or_sections=[2, 1],
                dim=-1,
            )
            samples.append(
                DataSample(
                    filepath=self.map_img_id[anno["image_id"]]["file_path"],
                    bbox=tv_tensors.BoundingBoxes(
                        anno["bbox"],
                        format="XYWH",  # all PT21 bboxes are in box_format XYWH
                        canvas_size=self.map_img_id[anno["image_id"]]["img_shape"][::-1],  # canvas (h,w)
                    ),
                    keypoints=keypoints,
                    person_id=anno["person_id"] if "person_id" in anno else -1,
                    # additional values which are not required
                    joint_weight=visibility,
                    track_id=anno["track_id"],
                    id=anno["id"],
                    image_id=anno["image_id"],
                    category_id=anno["category_id"],
                    bbox_head=tv_tensors.BoundingBoxes(
                        anno["bbox_head"],
                        format="XYWH",  # all PT21 bboxes are in box_format XYWH
                        canvas_size=self.map_img_id[anno["image_id"]]["img_shape"][::-1],  # canvas (h,w)
                    ),
                )
            )
        return samples
