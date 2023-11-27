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
