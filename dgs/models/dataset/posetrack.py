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
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import read_json
from dgs.utils.types import Config, ImgShape, NodePath, Validations

pt21_load_validations: Validations = {"path": ["str", "file exists in project", ("endswith", ".json")]}


class PoseTrack21Loader(BaseDataset):
    """Load precomputed json files"""

    def __init__(self, config: Config, path: NodePath) -> None:
        super(BaseDataset, self).__init__(config=config, path=path)

        self.validate_params(pt21_load_validations)

        self.json: dict[str, dict]
        self.json = read_json(self.params["path"])
        if not isinstance(self.json, dict):
            raise ValueError(f"The PoseTrack21 json file is expected to be a dict, but was {type(self.json)}")
        if "images" not in self.json:
            raise KeyError(
                f"It is expected that a PoseTrack21 .json file has an images key. But keys are {self.json.keys()}"
            )
        if "annotations" not in self.json:
            raise KeyError(
                f"It is expected that a PoseTrack21 .json file has an annotations key. But keys are {self.json.keys()}"
            )

        self.img_folder_path: str = self.params.get("img_folder_path", str(PROJECT_ROOT) + "/data/PoseTrack21/")
        # mapping from image id to all the information about this image that was provided by PT21
        self.map_img_id_to_img_obj: dict[int, dict[str, any]] = {img["image_id"]: img for img in self.json["images"]}
        for k, v in self.map_img_id_to_img_obj.items():
            # Add the full file path to the image
            self.map_img_id_to_img_obj[k]["file_path"] = os.path.join(self.img_folder_path, v["file_name"])
            # Add original image shape with imagesize output (w,h) and our own format (h, w)
            self.map_img_id_to_img_obj[k]["img_shape"]: ImgShape = imagesize.get(v["file_path"])[::-1]

        # generator for data
        self.data: list[DataSample] = self.pt21_to_data_sample()

    def pt21_to_data_sample(self) -> list[DataSample]:
        """Convert every detection in PoseTrack21 JSON file to DataSample.

        Returns:
            samples: A list of DataSamples.
        """
        samples: list[DataSample] = []
        for anno in self.json["annotations"]:
            # in PT21, the 17 key points and their visibility are stored in one list of len 51 [x_i, y_i, vis_i, ...]
            keypoints, visibility = torch.split(
                tensor=torch.FloatTensor(anno["keypoints"]).reshape((17, 3)),
                split_size_or_sections=[2, 1],
                dim=1,
            )
            samples.append(
                DataSample(
                    filepath=self.map_img_id_to_img_obj[anno["image_id"]]["file_path"],
                    bbox=tv_tensors.BoundingBoxes(
                        anno["bboxes"],
                        format="XYWH",  # all PT21 bboxes are in box_format XYWH
                        canvas_size=self.map_img_id_to_img_obj[anno["image_id"]]["img_shape"][::-1],  # canvas (h,w)
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
                        canvas_size=self.map_img_id_to_img_obj[anno["image_id"]]["img_shape"][::-1],  # canvas (h,w)
                    ),
                )
            )
        return samples
