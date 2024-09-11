"""
Module for creating submission files for |PT21|_ .

References:
    https://github.com/anDoer/PoseTrack21/blob/main/doc/dataset_structure.md

    https://github.com/leonid-pishchulin/poseval

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

import torch as t
from torchvision import tv_tensors as tvte
from torchvision.transforms.v2.functional import convert_bounding_box_format

from dgs.models.submission.submission import SubmissionFile
from dgs.utils.constants import PT21_CATEGORIES
from dgs.utils.files import write_json
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath


class PoseTrack21Submission(SubmissionFile):
    """Class for creating and appending to a |PT21|_ -style submission file."""

    data: dict[str, list[any]]

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        # add the categories to the json data and create the empty lists for the images and annotations
        self.clear()

    def append(self, s: State, *_args, **_kwargs) -> None:
        """Given data, append to the created |PT21| submission file."""
        self.data["images"].append(self.get_image_data(s))
        self.data["annotations"] += self.get_anno_data(s)

    def save(self) -> None:
        """Save the submission data in a file."""
        try:
            write_json(obj=self.data, filepath=self.fp)
        except TypeError as e:
            self.logger.exception(f"data: {self.data}")
            raise TypeError from e

    @staticmethod
    def get_image_data(s: State) -> dict[str, any]:
        """Given a :class:`.State`, extract data for the 'images' used in the submission file."""
        # validate the image data
        for key in ["filepath", "image_id", "frame_id"]:
            if key not in s:
                raise KeyError(f"Expected key '{key}' to be in State. Got {s}")
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

        # get the file_name in the PT21 directory
        file_name = f".{s.filepath[0].split('PoseTrack21')[-1]}"

        # get the image data
        image_data = {
            "file_name": file_name,
            "id": int(s["image_id"][0].item() if isinstance(s["image_id"], t.Tensor) else s["image_id"][0]),
            "image_id": int(s["image_id"][0].item() if isinstance(s["image_id"], t.Tensor) else s["image_id"][0]),
            "frame_id": int(s["frame_id"][0].item() if isinstance(s["frame_id"], t.Tensor) else s["frame_id"][0]),
        }
        return image_data

    @staticmethod
    def get_anno_data(s: State) -> list[dict[str, any]]:
        """Given a :class:`.State`, extract data for the 'annotations' list used in the submission file."""
        if s.B == 0:
            return []

        # validate the annotation data
        for key in ["person_id", "pred_tid", "bbox", "keypoints", "joint_weight"]:
            if key not in s:
                raise KeyError(f"Expected key '{key}' to be in State.")
            if (l := len(s[key])) != s.B:
                raise ValueError(f"Expected '{key}' ({l}) to have the same length as the State ({s.B}).")

        # get the annotation data
        anno_data = []
        if s.bbox.format != tvte.BoundingBoxFormat.XYWH:
            s.bbox = convert_bounding_box_format(s.bbox, new_format=tvte.BoundingBoxFormat.XYWH)
        assert s.bbox.format == tvte.BoundingBoxFormat.XYWH, f"got format: {s.bbox.format}"

        for i in range(s.B):
            kps = t.cat([s.keypoints[i], s.joint_weight[i]], dim=-1)
            scores: list[float]
            if "scores" in s:
                if isinstance(s["scores"], t.Tensor):
                    scores = s["scores"][i].to(dtype=t.float32).flatten().tolist()
                else:
                    scores = [float(score) for score in s["scores"]]
            else:
                scores = [0.0 for _ in range(17)]

            anno_data.append(
                {
                    "bboxes": s.bbox[i].flatten().tolist(),
                    "keypoints": kps.flatten().tolist(),
                    "scores": scores,
                    "score": (
                        float(sum(scores) / len(scores))
                        if "score" not in s
                        else s["score"][i].item() if isinstance(s["score"][i], t.Tensor) else s["score"][i]
                    ),
                    "image_id": int(
                        s["image_id"][i].item() if isinstance(s["image_id"], t.Tensor) else s["image_id"][i]
                    ),
                    "person_id": int(s.person_id[i].item()),
                    "track_id": int(s["pred_tid"][i].item()),
                }
            )

        return anno_data

    def clear(self) -> None:
        """Clear the submission data."""
        self.data = {
            "images": [],
            "annotations": [],
            "categories": PT21_CATEGORIES,
        }
