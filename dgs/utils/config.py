"""
Util- and helper-functions for handling configuration files and passing down sub configurations to other modules.

Contains functions for validating configuration and parameter of modules.
"""
import os

import yaml
from easydict import EasyDict

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
    # additional logical operators for nested validations
    "not": (lambda x, d: not VALIDATIONS[d[0]](x, d[1])),
    "and": (lambda x, d: all(VALIDATIONS[d[i][0]](x, d[i][1]) for i in range(len(d)))),
    "or": (lambda x, d: any(VALIDATIONS[d[i][0]](x, d[i][1]) for i in range(len(d)))),
    "xor": (lambda x, d: bool(VALIDATIONS[d[0][0]](x, d[0][1])) != bool(VALIDATIONS[d[1][0]](x, d[1][1]))),
}


def get_sub_config(config: Config, path: list[str]) -> Config:
    """
    Given a full configuration file in nested dict style or similar,
    return the given subtree by using the values of path as node keys.

    Works with regular dicts and with EasyDict.

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
    with open(filepath, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        if easydict:
            return EasyDict(config)
        return config


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
