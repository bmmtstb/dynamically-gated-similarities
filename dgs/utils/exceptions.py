"""
Definition of custom exceptions.
"""


class InvalidParameterException(Exception):
    """Exception to raise if one of the modules params is invalid or missing."""


class InvalidConfigException(Exception):
    """Exception to raise if one value of a modules config is invalid or missing."""


class ValidationException(Exception):
    """Exception to raise if there is something wrong with the validation object."""


class InvalidPathException(OSError):
    """Exception to raise if a given path is invalid."""

    def __init__(self, *args, filepath: str = None):
        if filepath:
            super().__init__(f"{filepath} is not a valid path.")
        else:
            super().__init__(*args)
