"""
util and helpers for handling configuration files
"""

from dgs.utils.types import Config


def get_sub_config(config: Config, path: list[str]) -> Config:
    """
    Given a full configuration file in nested dict style or similar,
    return the given subtree by using the values of path as node keys.

    Args:
        config: configuration file, EasyDict stile or plain nested dict
        path: path of a subtree within this dictionary

    Returns:
        Sub configuration, an excerpt of the original configuration
    """
    if not path:
        return config
    if isinstance(config, dict) and path[0] in config:
        return get_sub_config(config[path[0]], path[1:])
    raise KeyError(f"Key {path[0]} does not exist in current configuration {config}.")
