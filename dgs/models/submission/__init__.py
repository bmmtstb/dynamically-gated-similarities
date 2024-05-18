"""Modules for saving submission files in different formats."""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .posetrack21 import PoseTrack21Submission
from .submission import SubmissionFile

SUBMISSION_FORMATS: dict[str, Type[SubmissionFile]] = {
    "None": SubmissionFile,
    "PoseTrack21": PoseTrack21Submission,
}


def get_submission(name: str) -> Type[SubmissionFile]:
    """Given the name of one submission file format, return an instance of the respective class."""
    return get_instance(instance=name, instances=SUBMISSION_FORMATS, inst_class=SubmissionFile)


def register_submission(name: str, new_sub: Type[SubmissionFile]) -> None:
    """
    Register a new submission file format module in :data:``SUBMISSION_FORMATS``,
    to be able to use it from configuration files.
    """
    register_instance(name=name, instance=new_sub, instances=SUBMISSION_FORMATS, inst_class=SubmissionFile)
