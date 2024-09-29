"""Base module for generating submission files."""

from dgs.models.modules.named import NamedModule
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.types import Config, FilePath, NodePath, Validations

base_submission_validations: Validations = {
    # optional
    "submission_file": ["optional", str],
}


class SubmissionFile(NamedModule):
    """Base module for generating and handling submission files.

    The base module does not create a submission file and acts as the module with format "None".

    Params
    ------

    Optional Params
    ---------------

    file (str, optional):
        Path to the submission file, within the ``log_dir`` of the respective :class:`.BaseModule`.
        Default ``DEF_VAL.submission.file``.

    """

    fp: FilePath
    format: str
    data: any

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config=config, path=path)
        self.validate_params(base_submission_validations)

        self.fp = self.params.get("file", DEF_VAL["submission"]["file"])

    @property
    def module_type(self) -> str:
        return "submission"

    def __call__(self, *args, **kwargs) -> any:
        self.append(*args, **kwargs)

    def append(self, s: State, *_args, **_kwargs) -> None:
        """Append more data to the submission file."""

    def save(self) -> None:
        """Save the submission data to the submission file."""

    def terminate(self) -> None:
        """Terminate the submission file creation."""
        if self.data:
            del self.data

    def clear(self) -> None:
        """Clear the submission data."""
        if self.data:
            del self.data
        self.data = None
