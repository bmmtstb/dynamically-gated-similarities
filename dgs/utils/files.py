"""
Contains helper functions for loading and interacting with files and paths.
"""
import json
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


def is_abs_file(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing files absolute path"""
    return os.path.isfile(filepath) and os.path.isabs(filepath)


def is_abs_dir(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing absolute directory"""
    return os.path.isdir(filepath) and os.path.isabs(filepath)


def is_file(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing file either everywhere or from this project"""
    return is_abs_file(filepath) or is_project_file(filepath)


def is_dir(filepath: FilePath) -> bool:
    """Returns whether this filepath is an existing directory either everywhere or from this project"""
    return is_abs_dir(filepath) or is_project_dir(filepath)


def to_abspath(filepath: FilePath) -> FilePath:
    """Given a path return the absolute path of this file or directory.
    Will first check if the filepath already is an absolute file or path and then if it is a local project file.

    Args:
        filepath: str or path object as local or abspath.

    Returns:
        A valid path, global if it exists, otherwise local.

    Raises:
        InvalidPathException: If the path doesn't exist globally or locally.
    """
    if is_abs_file(filepath) or is_abs_dir(filepath):
        return os.path.normpath(filepath)
    if is_project_file(filepath) or is_project_dir(filepath):
        return os.path.normpath(os.path.join(PROJECT_ROOT, filepath))
    raise InvalidPathException(f"{filepath} is neither local nor global path.")


def mkdir_if_missing(dirname: FilePath) -> None:  # pragma: no cover
    """Creates leaf directory and all intermediates if it is missing."""
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def read_json(filepath: FilePath) -> dict[any, any] | list[any]:
    """Reads json file from a path.

    Args:
        filepath: str or path object with ending

    Returns:
        Loaded json from file as a dictionary.

    Raises:
        InvalidPathException: If filepath doesn't contain `.json` ending.
    """
    if not filepath.endswith(".json"):
        raise InvalidPathException(f"Presumed JSON file {filepath} does not have .json ending.")

    fpath = to_abspath(filepath)
    with open(fpath, encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def write_json(obj: dict, filepath: FilePath):  # pragma: no cover
    """Writes to a json file."""
    fpath = to_abspath(filepath)
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))
