"""
general utility functions
"""
import os

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.types import FilePath


def is_project_file(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing path in this project"""
    return os.path.isfile(os.path.join(PROJECT_ROOT, filepath))


def is_project_dir(filepath: FilePath) -> bool:
    """Returns whether this filepath is a path of an existing directory in this project"""
    return os.path.isdir(os.path.join(PROJECT_ROOT, filepath))


def is_file(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing file either everywhere or from this project"""
    return os.path.isfile(filepath) or is_project_file(filepath)


def is_dir(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing directory either everywhere or from this project"""
    return os.path.isdir(filepath) or is_project_dir(filepath)


def project_to_abspath(filepath: FilePath) -> FilePath:
    """Given a path return the absolute path of this file or directory.
    Will first check if the filepath is an absolute file or path and then if it is a local project file.

    """
    if os.path.isfile(filepath) or os.path.isdir(filepath) or os.path.isabs(filepath):
        return filepath
    if is_project_file(filepath) or is_project_dir(filepath):
        return os.path.join(PROJECT_ROOT, filepath)
    raise InvalidPathException(f"{filepath} is neither local nor global path.")
