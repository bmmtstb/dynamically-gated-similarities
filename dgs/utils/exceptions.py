"""
Definition of custom exceptions
"""


class InvalidParameterException(Exception):
    """Exception to raise if one of the modules params is invalid."""


class InvalidConfigException(Exception):
    """Exception to raise if one value of a modules config is invalid."""
