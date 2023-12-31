"""
Util- and helper-functions for handling configuration files and passing down sub configurations to other modules.

Contains functions for validating configuration and parameter of modules.
"""
from copy import deepcopy
from typing import Union

import yaml
from easydict import EasyDict

from dgs.utils.exceptions import InvalidConfigException, InvalidPathException
from dgs.utils.files import to_abspath
from dgs.utils.types import Config, FilePath


def get_sub_config(config: Config, path: list[str]) -> Union[Config, any]:
    """
    Given a full configuration file in nested dict style, EasyDict style, or similar,
    return the given subtree by using the items in path as node keys.

    Works with regular dicts and with EasyDict.

    Args:
        config: Configuration file, EasyDict stile or plain nested dict
        path: Path of a subtree within this config dictionary

    Examples:
        With a given configuration, this would look something like this.
        Keep in mind that most configs use the type EasyDict, but works the same.

        >>> cfg: Config = {
            "bar": {
                "x": 1,
                "y": 2,
                "deeper": {
                    "ore": "iron",
                    "pickaxe": {"iron": 10, "diamond": 100.0},
        }}}
        >>> print(get_sub_config(cfg, ["bar", "deeper", "pickaxe"]))
        {"iron": 10, "diamond": 100.0}
        >>> print(get_sub_config(cfg, ["bar", "x"]))
        1

    Returns:
        Either a sub configuration, which is an excerpt of the original configuration, or a single value.
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
            Path should start at project root or be a "real" system path (object).

            It is expected that all the configuration files are stored in the `configs` folder, but any path is valid.
        easydict: Whether to output a plain dictionary or an EasyDict object, which behaves mostly the same.
            Defaults to true, because every dict function should work with an EasyDict.
    Returns:
        The loaded configuration as a nested dictionary or `EasyDict` object.
    """
    try:
        fp = to_abspath(filepath)
    except InvalidPathException as e:
        raise InvalidPathException(f"Could not load configuration from {filepath}, because no such path exists.") from e

    with open(fp, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if easydict:
        return EasyDict(config)
    return config


def fill_in_defaults(config: Config, default_cfg: Config = None) -> Config:
    """Use values of a given configuration or the default configuration,
    to fill in missing values of the current configuration.

    For the current configuration, all existing key-value pairs will stay the same.
    Additionally, keys only present in the default configuration will be added to the current configuration.

    Args:
        config: Current configuration as EasyDict or nested dict
        default_cfg: Default configuration as EasyDict or nested dict

    Returns:
        The combined configuration.
    """

    def deep_update(default_dict: Config, new_dict: Config) -> Config:
        """Modify dict.update() to work recursively for nested dicts.
        Due to update being overwritten in the EasyDict package, this works for dict and EasyDict.

        Overwrites dict1.update(dict2).

        Args:
            default_dict: default dictionary, these values will be overwritten by the values of dict2

            new_dict: new dictionary, these values will definitely be in the result

        Returns:
            A modified version of `default_dict` with recursively combined / updated values taken from `new_dict`.
        """
        if not isinstance(default_dict, dict):
            return new_dict
        for key_new, val_new in new_dict.items():
            if isinstance(val_new, dict):
                default_dict[key_new] = deep_update(default_dict.get(key_new, {}), val_new)
            else:
                default_dict[key_new] = val_new
        return default_dict

    if not isinstance(config, EasyDict | dict):
        raise InvalidConfigException(f"Config is expected to be dict or EasyDict, but is {type(config)}.")

    if not default_cfg:
        from dgs.default_config import cfg  # pylint: disable=import-outside-toplevel

        default_cfg = cfg

    # make sure to create a copy of default config, values might get overwritten!
    new_config: Config = deep_update(deepcopy(default_cfg), config)

    if isinstance(config, EasyDict) or isinstance(default_cfg, EasyDict):
        return EasyDict(new_config)
    return new_config
