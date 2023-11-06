"""
Base model class as lowest building block for dynamic modules
"""
from abc import ABC, abstractmethod

from dgs.utils.config import get_sub_config
from dgs.utils.types import Config, Path


class BaseModule(ABC):
    """
    Every Module is a building block that can be replaced with other building blocks.
    This defines a base module all of those building blocks inherit
    """

    def __init__(self, config: Config, path: Path):
        """
        Every module has access the global configuration for parameters like device

        Args:
            config: the overall configuration of the whole algorithm
            path: keys of config to the parameters of the current module
                e.g. the parameters for the pose estimator will be located in a pose-estimator subgroup of the config
                those key-based paths may be even deeper, just make sure that only information about this specific model
                is stored in params
        """
        self.config: Config = config
        self.params: Config = get_sub_config(config, path)

    def __call__(self, *args, **kwargs):
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> ... | tuple[...]:
        """
        Module gets input and produces output.
        e.g. custom call of torch.model.forward()

        Returns:
            output of the model given inputs
            might have multiple outputs
        """
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        """
        Load given weights for current model

        Args:
            weight_path: path to a loadable file with weights for this model
        """
        raise NotImplementedError
