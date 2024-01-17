"""
Different pose based embedding generators.
"""
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import convert_bounding_box_format

from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.models.module import configure_torch_module
from dgs.utils.types import Config, NodePath, Validations

lpbe_validations: Validations = {
    "hidden_layers": [("instance", (list, tuple, None))],
    "joint_shape": [("isinstance", (list, tuple)), ("len", 2), lambda tup: all(i > 0 for i in tup)],
    "bbox_format": [
        (
            "or",
            (
                ("in", ["XYXY", "XYWH", "CXCYWH", "xyxy", "xywh", "cxcywh"]),
                ("isinstance", tv_tensors.BoundingBoxFormat),
            ),
        )
    ]
    # "bias": [("isinstance", bool)]
    # "nof_kernels": ["int", ("gt", 0)]
}


def set_up_hidden_layer_sizes(input_size: int, output_size: int, hidden_layers: list[int] | None = None) -> list[int]:
    """Given input and output size of an FC-NN, create a list of the sizes containing each hidden layer in the network.
    There might be zero hidden layers.

    :param input_size: The size of the input.
    :type input_size: int
    :param output_size: The size of the output.
    :type output_size: int
    :param hidden_layers:
    :type hidden_layers: list[int] | None
    :return: The sizes of the hidden layers including input and output size.
    :rtype: list[int]
    """
    layers: list[int] = [input_size]
    if not (hidden_layers is None or len(hidden_layers) == 0):
        for hidden_layer in hidden_layers:
            layers.append(int(hidden_layer))
    layers.append(output_size)

    return layers


@configure_torch_module
class KeyPointConvolutionPBEG(EmbeddingGeneratorModule, nn.Module):
    """Create a short torch Module that has one convolutional layer reducing the key points using relational information
    and an arbitrary number of hidden fully connected layers at the end.

    Module name
    -----------

    KeyPointConvolutionPBEG

    Description
    -----------

    First, the convolution of the key points is computed using a given number of :attr:nof_kernels,
    which will return `J` values after flattening the convolution output.
    Those values are then inserted with the four bounding box values into the first fully connected layer.
    There at this point, there can be an arbitrary number of hidden FC-layers.

    Model Input: ``[B x J x j_dim]`` and ``[B x 4]``
    Model Output: ``[B x self.embedding_size]``

    Params
    ------

    hidden_layers_kp: (Union[list[int], tuple[int, ...], None])
        Respective size of every hidden layer after the convolution of the key points.
        The value can be None to use only one single convolution layer to cast the inputs before adding the bboxes.
    hidden_layers_all: (Union[list[int], tuple[int, ...], None])
        Respective size of every hidden layer after adding the bounding boxes.
        The value can be None to use only one single linear NN-layer
        to cast the convoluted key points and bboxes to the outputs.
    joint_shape: (tuple[int, int])
        Number of joints and number of dimensions of the joints as tuple.
    bias: (bool, optional, default=True)
        Whether to use a bias term in the linear layers.
    nof_kernels: (int, optional, default=5)
        Only applicable with input_type='convolved' define the number of kernels to use for convolution.
    bbox_format: (Union[str, tv_tensors.BoundingBoxFormat], optional, default='XYWH')
        The format of the bounding box coordinates.
        This will have influence on the results.

    Important Inherited Params
    --------------------------

    embedding_size: (int)
        Output shape or size of the embedding.

    """

    def __init__(self, config: Config, path: NodePath):
        nn.Module.__init__(self)
        EmbeddingGeneratorModule.__init__(self, config, path)

        self.validate_params(lpbe_validations)

        J, j_dim = self.params.get("joint_shape")
        self.nof_kernels = self.params.get("nof_kernels", 5)
        # get bias from parameters or use default: True
        bias: bool = self.params.get("bias", True)

        # define layers
        self.conv = nn.Conv2d(J, J, kernel_size=(j_dim, self.nof_kernels), bias=bias)
        self.flat = nn.Flatten()

        hidden_layers_kp = set_up_hidden_layer_sizes(
            input_size=J,
            output_size=0,  # placeholder output size will be ignored during creation
            hidden_layers=self.params.get("hidden_layers_kp"),
        )
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(
                    in_features=hidden_layers_kp[i],
                    out_features=hidden_layers_kp[i + 1],
                    bias=bias,
                )
                for i in range(len(hidden_layers_kp) - 2)
            ],
        )

        hidden_layers_all = set_up_hidden_layer_sizes(
            input_size=hidden_layers_kp[-2] + 4,  # last real layer-size of key point fc layers, defaults to J
            output_size=self.embedding_size,
            hidden_layers=self.params.get("hidden_layers_all"),
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(
                    in_features=hidden_layers_all[i],
                    out_features=hidden_layers_all[i + 1],
                    bias=bias,
                )
                for i in range(len(hidden_layers_all) - 1)
            ],
            nn.Softmax(dim=-1),
        )

    def forward(self, *data, **kwargs) -> torch.Tensor:
        """Forward pass of the custom key point convolution model."""
        if len(data) < 2:
            raise ValueError(f"Data should contain key points and bounding boxes, but has length {len(data)}.")
        # extract key points and bboxes from data
        kp, bboxes, *_args = data

        # create new last dimension for the number of kernels -> 'nof_kernels'
        x = kp.unsqueeze(-1).expand(-1, -1, -1, self.nof_kernels)
        x = self.conv(x)  # Convolve the key points. Has an out shape of ``[B x J x 1 x 1]``
        x = self.flat(x)  # flatten to have out shape of ``[B x J]``
        x = self.fc1(x)  # fc layers for key points only, defaults no nothing

        # convert bboxes to the specified type
        bboxes = convert_bounding_box_format(bboxes, new_format=self.params["bbox_format"])

        # Concatenate ``[B x (J)]`` and ``[B x 4]``, and input them into the second fc layers.
        # The activation function is called in this Sequential.
        return self.fc2(torch.cat([x, bboxes], dim=-1))


@configure_torch_module
class LinearPBEG(EmbeddingGeneratorModule, nn.Module):
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
    joint_shape: (tuple[int, int])
        Number of joints and number of dimensions of the joints as tuple.
    bias: (bool, optional, default=True)
        Whether to use a bias term in the linear layers.
    bbox_format: (Union[str, tv_tensors.BoundingBoxFormat], optional, default='XYWH')
        The format of the bounding box coordinates.
        This will have influence on the results.

    Important Inherited Params
    --------------------------

    embedding_size: (int)
        Output shape or size of the embedding.
    """

    def __init__(self, config: Config, path: NodePath):
        nn.Module.__init__(self)
        EmbeddingGeneratorModule.__init__(self, config, path)

        self.validate_params(lpbe_validations)
        self.J, self.j_dim = self.params.get("joint_shape")
        # get bias from parameters or use default: True
        self.bias: bool = self.params.get("bias", True)

        self.model = self._init_flattened_model()

    def _init_flattened_model(self) -> nn.Module:
        """Initialize linear pose embedding generator model."""

        # input is given, additional hidden layers might be given in params
        hidden_layers = set_up_hidden_layer_sizes(
            input_size=self.J * self.j_dim + 4,
            output_size=self.embedding_size,
            hidden_layers=self.params.get("hidden_layers"),
        )
        return self.configure_torch_model(  # send to the target device
            nn.Sequential(
                nn.Flatten(),  # keep batch dim and one value dim, default start_dim=-1, end_dim=1
                *[
                    nn.Linear(
                        in_features=hidden_layers[i],
                        out_features=hidden_layers[i + 1],
                        bias=self.bias,
                    )
                    for i in range(len(hidden_layers) - 1)
                ],
                nn.Softmax(dim=-1),
            )
        )

    def forward(self, *data, **kwargs) -> torch.Tensor:
        """Forward pass of the linear pose-based embedding generator."""
        if len(data) == 1:
            # expect that input is already flattened
            if self.print("debug"):
                print(
                    "In the forward call of the LinearPBEG module data only contains one single value. "
                    "It is expected that this value is the flattened and concatenated key points and pose tensor."
                )
            data, *args = data
        elif len(data) > 1:
            kp, bboxes, *args = data
            # convert bboxes to the specified type
            bboxes = convert_bounding_box_format(bboxes, new_format=self.params["bbox_format"])
            data = torch.cat([kp.flatten(start_dim=1), bboxes.data.flatten(start_dim=1)], dim=-1)
        else:
            raise ValueError(f"Data should contain key points and bounding boxes, but has length {len(data)}.")
        return self.model(data, *args, **kwargs)
