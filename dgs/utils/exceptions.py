"""
Definition of custom exceptions
"""


class InvalidParameterException(Exception):
    """Exception to raise if one of the modules params is invalid."""


class InvalidConfigException(Exception):
    """Exception to raise if one value of a modules config is invalid."""


class ValidationException(Exception):
    """Exception to raise if there is something wrong with the validation object"""


class InvalidPathException(OSError):
    """Exception to raise if a given path is invalid."""


class BoundingBoxException(Exception):
    """Exception to raise if a given bounding-box is invalid."""


class DimensionMismatchException(Exception):
    """
    Exception for any custom size or dimension mismatches
    """

    def __init__(self, expected: int | list[int, ...] | tuple[int, ...], actual: int, title: str = "") -> None:
        if len(title) > 0:
            title += " - "
        super().__init__(f"{title} Dimension mismatch, expected: {expected}, actual: {actual}")


class PathException(Exception):
    """
    Exception for missing or otherwise faulty paths or folders
    """

    def __init__(self, path: str):
        super().__init__(f'Path: "{path}" does not exist.')


class FileException(Exception):
    """
    Exception for missing or otherwise faulty files
    """

    def __init__(self, file_name: str, path: str | None = None):
        text: str = "Could not find file " + str(file_name)
        if path is not None:
            text += " at path " + str(path)
        super().__init__(text)
