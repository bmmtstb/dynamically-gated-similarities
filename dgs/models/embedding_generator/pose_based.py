"""
Different pose based embedding generators.
"""

from typing import Union

import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms.v2 import ConvertBoundingBoxFormat

from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.torchtools import configure_torch_module
from dgs.utils.types import Config, NodePath, Validations

kpcpbeg_validations: Validations = {
    "joint_shape": [list, ("len", 2), lambda l: all(i > 0 for i in l)],
    "bbox_format": [
        "optional",
        (
            "any",
            [
                ("in", ["XYXY", "XYWH", "CXCYWH"]),
                ("instance", tv_tensors.BoundingBoxFormat),
            ],
        ),
    ],
    "bias": ["optional", bool],
    "hidden_layers": ["optional", ("instance", (list, tuple, None))],
    "hidden_layers_kp": ["optional", ("instance", (list, tuple, None))],
    "nof_kernels": ["optional", int, ("gt", 0)],
}

lpbeg_validations: Validations = {
    "joint_shape": [list, ("len", 2), ("forall", [int, ("gt", 0)])],
    "bbox_format": [
        "optional",
        (
            "any",
            [
                ("in", ["XYXY", "XYWH", "CXCYWH"]),
                ("instance", tv_tensors.BoundingBoxFormat),
            ],
        ),
    ],
    "bias": ["optional", bool],
    "hidden_layers": ["optional", ("instance", (list, tuple, None))],
    "nof_kernels": ["optional", int, ("gt", 0)],
}


def set_up_hidden_layer_sizes(
    input_size: int, output_size: int, hidden_layers: Union[list[int], None] = None
) -> list[int]:
    """Given the input and output size of an FC-NN,
    create a list of the sizes containing each hidden layer in the network.
    There might be zero hidden layers.

    Params:
        input_size: The size of the input to the FC-Layers.
        output_size: Output-size of the FC-Layers.
        hidden_layers: The dimensionality of each hidden layer in this network. Default None means no hidden layers.

    Returns:
        The sizes of the hidden layers including input and output size.
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

    Module Name
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

    joint_shape: (tuple[int, int])
        Number of joints and number of dimensions of the joints as tuple.

    Optional Params
    ---------------

    hidden_layers_kp: (Union[list[int], tuple[int, ...], None], optional)
        Respective size of every hidden layer after the convolution of the key points.
        The value can be None to use only one single convolution layer to cast the inputs before adding the bboxes.
        Default ``DEF_VAL.embed_gen.pose.KPCPBEG.hidden_layers_kp``.
    hidden_layers: (Union[list[int], tuple[int, ...], None], optional)
        Respective size of every hidden layer after adding the bounding boxes.
        The value can be None to use only one single linear NN-layer
        to cast the convoluted key points and bboxes to the outputs.
        Default ``DEF_VAL.embed_gen.pose.KPCPBEG.hidden_layers``.
    bias: (bool, optional)
        Whether to use a bias term in the linear layers.
        Default ``DEF_VAL.embed_gen.pose.KPCPBEG.bias``.
    nof_kernels: (int, optional)
        Define the number of kernels to use for convolution.
        Default ``DEF_VAL.embed_gen.pose.KPCPBEG.nof_kernels``.
    bbox_format: (Union[str, tv_tensors.BoundingBoxFormat], optional)
        The target format of the bounding box coordinates.
        This will have influence on the results.
        Default ``DEF_VAL.embed_gen.pose.KPCPBEG.bbox_format``.


    Important Inherited Params
    --------------------------

    embedding_size: (int)
        Output shape or size of the embedding.

    """

    def __init__(self, config: Config, path: NodePath):
        nn.Module.__init__(self)
        EmbeddingGeneratorModule.__init__(self, config, path)

        self.validate_params(kpcpbeg_validations)

        J, j_dim = self.params.get("joint_shape")
        self.nof_kernels = self.params.get("nof_kernels", DEF_VAL.embed_gen.pose.KPCPBEG.nof_kernels)
        # get bias from parameters or use default: True
        bias: bool = self.params.get("bias", DEF_VAL.embed_gen.pose.KPCPBEG.bias)

        self.bbox_converter = ConvertBoundingBoxFormat(
            format=self.params.get("bbox_format", DEF_VAL.embed_gen.pose.KPCPBEG.bbox_format)
        )

        # define layers
        conv = nn.Conv2d(J, J, kernel_size=(j_dim, self.nof_kernels), bias=bias)
        flat = nn.Flatten()

        hidden_layers_kp = set_up_hidden_layer_sizes(
            input_size=J,
            output_size=0,  # placeholder output size will be ignored during creation
            hidden_layers=self.params.get("hidden_layers_kp", DEF_VAL.embed_gen.pose.KPCPBEG.hidden_layers_kp),
        )
        fc1 = nn.Sequential(
            *[
                nn.Linear(
                    in_features=hidden_layers_kp[i],
                    out_features=hidden_layers_kp[i + 1],
                    bias=bias,
                )
                for i in range(len(hidden_layers_kp) - 2)
            ],
        )
        self.part1 = nn.Sequential(conv, flat, fc1)

        hidden_layers_all = set_up_hidden_layer_sizes(
            input_size=hidden_layers_kp[-2] + 4,  # last real layer-size of key point fc layers, defaults to J
            output_size=self.embedding_size,
            hidden_layers=self.params.get("hidden_layers", DEF_VAL.embed_gen.pose.KPCPBEG.hidden_layers),
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
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, self.nof_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, ds: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the custom key point convolution model.

        Params:
            ds: A :class:`State` containing the key-points and the corresponding bounding boxes.

        Returns:
            This modules' prediction.
            ``embeddings`` is describing the key-points and bounding boxes as a tensor of shape ``[B x E]``.
            ``ids`` is the probability to predict a class.
            The ids are given as a tensor of shape ``[B x num_classes]`` with values in range `[0, 1]`.
        """
        # extract key points and bboxes from data and get them into the right shape
        kp = ds.keypoints
        bboxes = ds.bbox
        # convert bboxes to the specified type
        bboxes = self.bbox_converter(bboxes)
        # create new last dimension for the number of kernels -> 'nof_kernels'
        x = kp.unsqueeze(-1).expand(-1, -1, -1, self.nof_kernels)

        # Convolve the key points, get an output shape of [B x J x 1 x 1].
        # Flatten to have out shape of [B x J].
        # Run the fc layers for the key-points only, output [B x j]
        x = self.part1(x)

        # Concatenate [B x j] and [B x 4] along the last dim.
        # Then use that as input into the second group of fc layers to obtain the embeddings.
        embeddings = self.fc2(torch.cat([x, bboxes], dim=-1))
        # Obtain the class (id) probabilities by calling the classifier on the embeddings.
        ids = self.classifier(embeddings)

        return embeddings, ids


@configure_torch_module
class LinearPBEG(EmbeddingGeneratorModule, nn.Module):
    """Model to compute a pose-embedding given a pose, or batch of poses describing them as a single vector.

    Module Name
    -----------

    LinearPBEG

    Description
    -----------

    The model consists of one or multiple linear layers followed by a single sigmoid activation function.
    The number of linear layers is determined by the length of the hidden_layers parameter.

    Params
    ------

    joint_shape: (tuple[int, int])
        The number of joints and number of dimensions of the joints as tuple.
        For data from 'COCO' the number of joints is 17 and all joints are two-dimensional.
        Therefore, resulting in ``joint_shape = (17, 2)``.

    Optional Params
    ---------------

    hidden_layers: (Union[list[int], tuple[int, ...], None], optional)
        Respective size of every hidden layer.
        The value can be None to use only one single linear NN-layer to cast the inputs to the outputs.
        Default ``DEF_VAL.embed_gen.pose.LPBEG.hidden_layers``.
    bias: (bool, optional)
        Whether to use a bias term in the linear layers.
        Default ``DEF_VAL.embed_gen.pose.LPBEG.bias``.
    bbox_format: (Union[str, tv_tensors.BoundingBoxFormat], optional)
        The target format of the bounding box coordinates.
        This will have influence on the results.
        Default ``DEF_VAL.embed_gen.pose.LPBEG.bbox_format``.

    Important Inherited Params
    --------------------------

    embedding_size: (int)
        Output shape or size of the embedding.
    """

    def __init__(self, config: Config, path: NodePath):
        nn.Module.__init__(self)
        EmbeddingGeneratorModule.__init__(self, config, path)

        self.validate_params(lpbeg_validations)

        self.J, self.j_dim = self.params.get("joint_shape")
        # get bias from parameters or use default: True
        self.bias: bool = self.params.get("bias", DEF_VAL.embed_gen.pose.LPBEG.bias)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, self.nof_classes),
            nn.Softmax(dim=-1),
        )

        self.model = self._init_flattened_model()

        self.bbox_converter = ConvertBoundingBoxFormat(
            format=self.params.get("bbox_format", DEF_VAL.embed_gen.pose.LPBEG.bbox_format)
        )

    def _init_flattened_model(self) -> nn.Module:
        """Initialize linear pose embedding generator model."""

        # input is given, additional hidden layers might be given in params
        hidden_layers = set_up_hidden_layer_sizes(
            input_size=self.J * self.j_dim + 4,
            output_size=self.embedding_size,
            hidden_layers=self.params.get("hidden_layers", DEF_VAL.embed_gen.pose.LPBEG.hidden_layers),
        )
        return self.configure_torch_module(  # send to the target device
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
            )
        )

    def forward(self, ds: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the linear pose-based embedding generator.

        Params:
            ds: Either an already flattened tensor, containing the values of the key-point coordinates and the
                bounding box as a single tensor of shape ``[B x self.J * self.j_dim + 4]``, or
                the key-point coordinates and bounding boxes as tensors of shapes ``[B x self.J]`` and ``[B x 4]``.

        Returns:
            This modules' prediction.
            ``embeddings`` is describing the key-points and bounding boxes as a tensor of shape ``[B x E]``.
            ``ids`` is the probability to predict a class.
            The ids are given as a tensor of shape ``[B x num_classes]`` with values in range `[0, 1]`.
        """
        # extract key points and bboxes from data
        kp = ds.keypoints
        bboxes = ds.bbox
        # convert bboxes to the specified target type
        bboxes = self.bbox_converter(bboxes)
        data = torch.cat([kp.flatten(start_dim=1), bboxes.data.flatten(start_dim=1)], dim=-1)

        embeddings = self.model(data)

        ids = self.classifier(embeddings)

        return embeddings, ids
