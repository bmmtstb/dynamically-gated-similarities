"""
Util- and helper-functions for handling configuration files and passing down sub configurations to other modules.

Contains functions for validating configuration and parameter of modules.
"""
import os
from copy import deepcopy
from typing import Union

import yaml
from easydict import EasyDict

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidConfigException
from dgs.utils.files import is_project_file, project_to_abspath
from dgs.utils.types import Config, FilePath, Validator

VALIDATIONS: dict[str, Validator] = {
    "None": (lambda x, _: x is None),
    "not None": (lambda x, _: x is not None),
    "eq": (lambda x, d: x == d),
    "neq": (lambda x, d: x != d),
    "gt": (lambda x, d: isinstance(d, int | float) and x > d),
    "gte": (lambda x, d: isinstance(d, int | float) and x >= d),
    "lt": (lambda x, d: isinstance(d, int | float) and x < d),
    "lte": (lambda x, d: isinstance(d, int | float) and x <= d),
    "in": (lambda x, d: hasattr(d, "__contains__") and x in d),
    "not in": (lambda x, d: x not in d),
    "contains": (lambda x, d: hasattr(x, "__contains__") and d in x),
    "not contains": (lambda x, d: hasattr(x, "__contains__") and d not in x),
    "str": (lambda x, _: isinstance(x, str)),
    "int": (lambda x, _: isinstance(x, int)),
    "float": (lambda x, _: isinstance(x, float)),
    "number": (lambda x, _: isinstance(x, int | float)),
    "instance": isinstance,
    "between": (lambda x, d: isinstance(d, tuple) and len(d) == 2 and d[0] < x < d[1]),
    "within": (lambda x, d: isinstance(d, tuple) and len(d) == 2 and d[0] <= x <= d[1]),
    "outside": (lambda x, d: isinstance(d, tuple) and len(d) == 2 and x < d[0] or x > d[1]),
    "len": (lambda x, d: hasattr(x, "__len__") and len(x) == d),
    "shorter": (lambda x, d: hasattr(x, "__len__") and len(x) <= d),
    "longer": (lambda x, d: hasattr(x, "__len__") and len(x) >= d),
    "startswith": (lambda x, d: isinstance(x, str) and isinstance(d, str) and x.startswith(d)),
    "endswith": (lambda x, d: isinstance(x, str) and isinstance(d, str) and x.endswith(d)),
    "file exists": (lambda x, _: isinstance(x, FilePath) and os.path.isfile(x)),
    "file exists in project": (lambda x, _: isinstance(x, FilePath) and os.path.isfile(os.path.join(PROJECT_ROOT, x))),
    "file exists in folder": (
        lambda x, f: isinstance(x, FilePath) and isinstance(f, FilePath) and os.path.isfile(os.path.join(f, x))
    ),
    "folder exists": (lambda x, _: isinstance(x, FilePath) and os.path.isdir(x)),
    "folder exists in project": (lambda x, _: isinstance(x, FilePath) and os.path.isdir(os.path.join(PROJECT_ROOT, x))),
    "folder exists in folder": (
        lambda x, f: isinstance(x, FilePath) and isinstance(f, FilePath) and os.path.isdir(os.path.join(f, x))
    ),
    # additional logical operators for nested validations
    "not": (lambda x, d: not VALIDATIONS[d[0]](x, d[1])),
    "and": (lambda x, d: all(VALIDATIONS[d[i][0]](x, d[i][1]) for i in range(len(d)))),
    "or": (lambda x, d: any(VALIDATIONS[d[i][0]](x, d[i][1]) for i in range(len(d)))),
    "xor": (lambda x, d: bool(VALIDATIONS[d[0][0]](x, d[0][1])) != bool(VALIDATIONS[d[1][0]](x, d[1][1]))),
}


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

        >>> cfg: Config = {                                         \
            "bar": {                                                \
                "x": 1,                                             \
                "y": 2,                                             \
                "deeper": {                                         \
                    "ore": "iron",                                  \
                    "pickaxe": {"iron": 10, "diamond": 100.0},      \
        }}}
        >>> print(get_sub_config(cfg, ["bar", "deeper", "pickaxe"]))
        {"iron": 10, "diamond": 100.0}
        >>> print(get_sub_config(cfg, ["bar", "x"]))
        1

    Returns:
        Sub configuration, an excerpt of the original configuration, or single value.
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
        Loaded configuration as nested dictionary or easydict
    """
    if is_project_file(filepath):
        filepath = project_to_abspath(filepath)
    with open(filepath, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        if easydict:
            return EasyDict(config)
        return config


def fill_in_defaults(config: Config, default_cfg: Config = None) -> Config:
    """Use values of the default configuration to fill in missing values of the current configuration.

    For the current configuration, all existing key-value pairs will stay the same.
    Additionally, keys only present in the default configuration will be added to the current configuration.

    Args:
        config: Current configuration as EasyDict or nested dict
        default_cfg: Default configuration as EasyDict or nested dict

    Returns:
        Combined configuration
    """

    def deep_update(default_dict: Config, new_dict: Config) -> Config:
        """Modify dict.update() to work recursively for nested dicts.
        Due to update being overwritten in the EasyDict package, this works for dict and EasyDict.

        Overwrites dict1.update(dict2).

        Args:
            default_dict: default dictionary, these values will be overwritten by the values of dict2

            new_dict: new dictionary, these values will definitely be in the result

        Returns:
            A modified version of dictionary 1 with recursively combined / updated values
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


def validate_value(value: any, data: any, validation: str) -> bool:
    """
    Check a single value against a given predefined validation, possibly given some additional data

    Args:
        value: The value to validate.
        data: Possibly additional data needed for validation, is ignored otherwise.
        validation: The name of the validation to perform.

    Returns:
        Whether the given value is valid.
    """
    if validation not in VALIDATIONS:
        raise KeyError(f"Validation {validation} does not exist.")

    return VALIDATIONS[validation](value, data)
