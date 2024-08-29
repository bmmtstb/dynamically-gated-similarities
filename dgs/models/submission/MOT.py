"""
Module for creating submission files for |MOT|_ based datasets.
For more information, visit the `submission instructions <https://motchallenge.net/instructions/>`_.

Notes:
    The structure of the submission files is similar to the structures of the inputs,
    where each line contains one annotated bounding-box::

        ``<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>``

    During loading, the ``<id>`` seems to be the person-ID,
    while for submissions it is the predicted track-ID or the predicted person-ID.
"""

import torchvision.tv_tensors as tvte
from torchvision.transforms.v2.functional import convert_bounding_box_format

from dgs.models.dataset.MOT import write_MOT_file
from dgs.models.submission.submission import SubmissionFile
from dgs.utils.config import DEF_VAL
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

mot_submission_validations: Validations = {
    # optional
    "bbox_decimals": ["optional", int, ("gte", 0)],
    "score_decimals": ["optional", int, ("gte", 0)],
}


class MOTSubmission(SubmissionFile):
    """Class for creating and appending to a |MOT|_ -style submission file.

    Optional Params
    ---------------

    bbox_decimals (int, optional):
        The number of decimals to save for the bbox coordinates.
        Default ``DEF_VAL["submission"]["MOT"]["bbox_decimals"]``.
    score_decimals (int, optional):
        The number of decimals to save for the score value.
        Is only used if the score is present and ``score >= 0``.
        Default ``DEF_VAL["submission"]["MOT"]["score_decimals"]``.
    """

    data: list[tuple[any, ...]]
    """A list containing the values as tuple, like:
    ``tuple(<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)``
    """

    frame_id: int

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(mot_submission_validations)

        # reset data
        self.clear()

        self.bbox_decimals: int = int(self.params.get("bbox_decimals", DEF_VAL["submission"]["MOT"]["bbox_decimals"]))
        self.score_decimals: int = int(
            self.params.get("score_decimals", DEF_VAL["submission"]["MOT"]["score_decimals"])
        )

    def append(self, s: State, *_args, **_kwargs) -> None:
        """Given a new state containing the detections of one image, append the data to the submission file."""

        def _get_bbox_value(_s: State, idx: int) -> str:
            """Given the state, get the value of the bbox at index ``[0, idx]`` as a string,
            with the requested number of decimals."""
            val = _s.bbox[0, idx].round(decimals=self.bbox_decimals).item()
            if self.bbox_decimals == 0:
                return str(int(val))
            return f"{val:.{self.bbox_decimals}f}"

        if "pred_tid" not in s:
            raise ValueError("The predicted track-ID should be set.")

        # convert bbox format to receive the height and width more easily later on
        if s.bbox.format != tvte.BoundingBoxFormat.XYWH:
            s.bbox = convert_bounding_box_format(s.bbox, new_format=tvte.BoundingBoxFormat.XYWH)
        assert s.bbox.format == tvte.BoundingBoxFormat.XYWH, f"got format: {s.bbox.format}"
        detections = s.split()
        for det in detections:
            tid = det["pred_tid"].item() + 1  # MOT is 1-indexed, but State is 0-indexed
            if "score" in det:
                score = round(float(det["score"].item()), self.score_decimals)
                conf = f"{score:.{self.score_decimals}f}"
            else:
                conf = str(1)
            x = det["x"] if "x" in det else -1
            y = det["y"] if "y" in det else -1
            z = det["z"] if "z" in det else -1
            self.data.append(
                (
                    self.frame_id,  # <frame>
                    tid,  # <track_id>
                    _get_bbox_value(det, 0),  # X = <bb_left>
                    _get_bbox_value(det, 1),  # Y = <bb_top>
                    _get_bbox_value(det, 2),  # W = <bb_width>
                    _get_bbox_value(det, 3),  # H = <bb_height>
                    conf,  # <conf>
                    x,  # <x>
                    y,  # <y>
                    z,  # <z>
                )
            )
            det.clean()

        self.frame_id += 1

    def save(self) -> None:
        """Save the current data to the given filepath."""
        try:
            # MOT / detection file
            assert len(self.data) > 0, "No data to save"
            write_MOT_file(fp=self.fp, data=self.data)
        except TypeError as te:
            self.logger.exception(f"data: {self.data}")
            raise TypeError from te
        except InvalidPathException as ipe:
            self.logger.exception(f"fp: {self.fp}")
            raise InvalidPathException from ipe

    def clear(self) -> None:
        """Clear the data."""
        self.data = []
        self.frame_id = 1
