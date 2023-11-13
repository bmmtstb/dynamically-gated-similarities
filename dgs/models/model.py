"""
Base model class as lowest building block for dynamic modules
"""
from abc import ABC, ABCMeta, abstractmethod

import torch

from dgs.utils.config import get_sub_config
from dgs.utils.constants import PRINT_PRIO
from dgs.utils.types import Config, Path


class BaseModule(ABC, metaclass=ABCMeta):
    """
    Every Module is a building block that can be replaced with other building blocks.
    This defines a base module all of those building blocks inherit
    """

    def __init__(self, config: Config, path: Path):
        """
        Every module has access the global configuration for parameters like the modules' device

        Args:
            config: the overall configuration of the whole algorithm
            path: keys of config to the parameters of the current module
                e.g. the parameters for the pose estimator will be located in a pose-estimator subgroup of the config
                those key-based paths may be even deeper, just make sure that only information about this specific model
                is stored in params
        """
        self.config: Config = config
        self.params: Config = get_sub_config(config, path)

        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate this module's config, throws exceptions if invalid.

        Configuration should include at least the following:
            device (Device): torch device or string
            print_prio (str): printing priority, has to be in PRINT_PRIO
        """
        # validate device
        if not self.config["device"] or (  # does not exist
            not (  # is not either valid string nor existing torch.device
                (isinstance(self.config["device"], str) and self.config["device"] in ["cuda", "cpu"])
                or isinstance(self.config["device"], torch.device)
            )
        ):
            raise ValueError("Module config does not contain valid device.")
        # validate print priority
        if not self.config["print_prio"] or (  # does not exist
            self.config["print_prio"] not in PRINT_PRIO  # is not in the choices
        ):
            raise ValueError("Module config does not contain valid print priority")

    @abstractmethod
    def __call__(self, *args, **kwargs) -> any:
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        """
        Load given weights for the current model

        Args:
            weight_path: path to a loadable file with weights for this model
        """
        raise NotImplementedError

    def print(self, priority: str) -> bool:
        """
        Check whether the Module is allowed to print something with the given priority

        Args:
            priority: print priority, has to be in PRINT_PRIO

        Returns:
            Whether the module is allowed to print given its priority
        """
        try:
            index_given: int = PRINT_PRIO.index(priority)
        except ValueError as verr:
            raise ValueError(f"Priority: {priority} is not in {PRINT_PRIO}") from verr
        if priority == "none":
            raise ValueError("To print with priority of none doesn't make sense...")

        index_current: int = PRINT_PRIO.index(self.config["print_prio"])

        return index_given <= index_current
