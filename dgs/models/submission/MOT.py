"""
Module for creating submission files for |MOT|_ based datasets.
For more information, visit the `submission instructions <https://motchallenge.net/instructions/>`_.

Notes:
    The structure of the submission files is similar to the structures of the inputs,
    where each line contains one annotated bounding-box::

        ``<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>``

    During loading, the ``<id>`` seems to be the person-ID,
    while for submissions it is the track-ID or the predicted person-ID.

Notes:
    Iff the person_id is unknown, it can be set to ``-1``.
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
    "bbox_decimals": ["optional", int],
}


class MOTSubmission(SubmissionFile):
    """Class for creating and appending to a |MOT|_ -style submission file.

    Optional Params
    ---------------

    bbox_decimals (int, optional):
        The number of decimals to save for the bbox coordinates.
        Default ``DEF_VAL["submission"]["MOT"]["bbox_decimals"]``.
    """

    data: list[tuple[any, ...]]
    """A list containing the values as tuple, like:
    ``tuple(<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)``
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config=config, path=path)

        self.validate_params(mot_submission_validations)

        self.data = []
        self.frame_id: int = 1
        self.bbox_decimals: int = self.params.get("bbox_decimals", DEF_VAL["submission"]["MOT"]["bbox_decimals"])

    def append(self, s: State, *_args, **_kwargs) -> None:
        """Given a new state containing the detections of one image, append the data to the submission file."""

        def _get_bbox_value(_s: State, idx: int) -> str:
            """Given the state, get the value of the bbox at index ``[0, idx]`` as a string,
            with the requested number of decimals."""
            val = _s.bbox[0, idx].round(decimals=self.bbox_decimals).item()
            if self.bbox_decimals == 0:
                return str(int(val))
            return f"{val:.{self.bbox_decimals}}"

        # convert bbox format to receive the height and width more easily later on
        if s.bbox.format != tvte.BoundingBoxFormat.XYWH:
            convert_bounding_box_format(s.bbox, new_format=tvte.BoundingBoxFormat.XYWH)
        detections = s.split()
        for det in detections:
            tid = det["pred_tid"].item()
            conf = float(det["score"].item()) if "score" in det else 1
            x = det["x"] if "x" in det else -1
            y = det["y"] if "y" in det else -1
            z = det["z"] if "z" in det else -1
            self.data.append(
                (
                    self.frame_id,  # <frame>
                    tid,  # <track_id>
                    _get_bbox_value(s, 0),  # X = <bb_left>
                    _get_bbox_value(s, 1),  # Y = <bb_top>
                    _get_bbox_value(s, 2),  # W = <bb_width>
                    _get_bbox_value(s, 3),  # H = <bb_height>
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
            write_MOT_file(fp=self.fp, data=self.data)
        except TypeError as te:
            self.logger.exception(f"data: {self.data}")
            raise TypeError from te
        except InvalidPathException as ipe:
            self.logger.exception(f"fp: {self.fp}")
            raise InvalidPathException from ipe
