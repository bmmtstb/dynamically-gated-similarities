"""
util and helpers for handling configuration files
"""
import yaml
from easydict import EasyDict

from dgs.utils.types import Config, FilePath


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


def load_config(filepath: FilePath, easydict: bool = True) -> Config:
    """
    Load a config.yaml file as nested dictionary.

    Args:
        filepath: Full filepath to the config.yaml file.
            Path should either start at project root or be a "real" system path (object).

            It is expected that all the configuration files are stored in the `configs` folder, but any path is valid.
        easydict: Whether to output a plain dictionary or an EasyDict object, which behaves mostly the same.
            Defaults to true, because every dict function should work with an EasyDict.
    Returns:
        Loaded configuration as nested dictionary or easydict
    """
    with open(filepath, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        if easydict:
            return EasyDict(config)
        return config
