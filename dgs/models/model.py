"""
Base model class as lowest building block for dynamic modules
"""
from abc import ABC, ABCMeta, abstractmethod

from dgs.utils.config import get_sub_config
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
