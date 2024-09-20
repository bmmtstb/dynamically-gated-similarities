"""
Base Class for Modules that have a ``module_name`` and ``module_type``.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

from dgs.models.module import BaseModule
from dgs.utils.loader import get_registered_class_names
from dgs.utils.types import Config, NodePath, Validations

named_module_validations: Validations = {
    "module_name": [str],
}


class NamedModule(BaseModule, ABC):
    """
    Abstract class for modules with a given ``module_name``.

    Params
    ------

    module_name (str):
        The name of the module.

    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        deepcopy(named_module_validations["module_name"]).append(("in", get_registered_class_names(self.module_type)))

        self.validate_params(named_module_validations)

    @property
    @abstractmethod
    def module_type(self) -> str:
        raise NotImplementedError

    @property
    def module_name(self) -> str:
        """Get the name of the module."""
        return self.params["module_name"]
