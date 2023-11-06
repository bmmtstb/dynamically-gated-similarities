"""
util and helpers for handling configuration files
"""

from dgs.utils.types import Config


def get_sub_config(config: Config, path: list[str]) -> Config:
    """
    Given a full config file, return the subtree given by path

    Args:
        config: configuration file, EasyDict stile
        path: get the subtree at this path

    Returns:
        subconfig
    """
    if not path:
        return config
    if isinstance(config, dict) and path[0] in config:
        return get_sub_config(config[path[0]], path[1:])
    raise KeyError(f'Key {path[0]} does not exist in current configuration {config}.')
