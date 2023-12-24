"""
Different pose based embedding generators.
"""

from torch import nn

from dgs.models.embedding_generator.embedding_generator import TorchEmbeddingGeneratorModule
from dgs.utils.types import Config, NodePath, Validations

lpbe_validations: Validations = {
    "hidden_layers": [("instance", (list, tuple, None))],
    "input_shape":   ["int", ("gt", 0)],
    # "bias": [("isinstance", bool)]
}


class LinearPoseBasedEmbeddingGenerator(TorchEmbeddingGeneratorModule):
    """Model to compute a pose-embedding given a pose, or batch of poses describing them as a single vector.

    Module name
    -----------

    LinearPBEG

    Description
    -----------

    The model consists of one or multiple linear layers followed by a single sigmoid activation function.
    The number of linear layers is determined by the length of the hidden_layers parameter.

    Params
    ------

    hidden_layers: (Union[list[int], tuple[int, ...], None])
        Respective size of every hidden layer.
        The value can be None to use only one single linear NN-layer to cast the inputs to the outputs.
    input_shape: (int)
        Size of the input.
    bias: (bool, optional)
        Whether to use a bias term in the linear layers.

    Important Inherited Params
    --------------------------

    embedding_size: (int)
        Output shape or size of the embedding.
    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(lpbe_validations)
        self.model = self._init_model()

    def _init_model(self) -> nn.Module:
        """Initialize linear pose embedding generator model."""

        # hidden layers might be given in params
        hidden_layers: list[int] = [self.params["input_shape"]]
        if not (self.params["hidden_layers"] is None or len(self.params["hidden_layers"]) == 0):
            for hidden_layer in self.params["hidden_layers"]:
                hidden_layers.append(int(hidden_layer))
        hidden_layers.append(self.embedding_size)

        # get bias from parameters or use default: True
        bias: bool = self.params.get("bias", True)

        return self.configure_torch_model(
            nn.Sequential(
                *[
                     nn.Linear(
                         in_features=hidden_layers[i], out_features=hidden_layers[i + 1], bias=bias, device=self.device
                     )
                     for i in range(len(hidden_layers) - 1)
                 ]
                 + [nn.Sigmoid()]
            )
        )
