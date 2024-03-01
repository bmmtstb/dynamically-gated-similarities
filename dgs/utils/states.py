"""
definitions and helpers for pose-state(s)
"""

from collections import deque, UserDict
from collections.abc import Iterable
from typing import Union

import torch
from torchvision import tv_tensors

from dgs.utils.types import DataGetter, FilePaths, Heatmap, Image, TVImage
from dgs.utils.validation import (
    validate_bboxes,
    validate_filepath,
    validate_heatmaps,
    validate_images,
    validate_key_points,
)


class Queue:
    """A single queue containing the last states of a specific torch.Tensor up to a limit N."""

    _shape: torch.Size

    def __init__(self, N: int, states: list[torch.Tensor] = None, shape: torch.Size = None) -> None:
        if N <= 0:
            raise ValueError(f"N must be greater than 0 but got {N}")
        self._N: int = N

        self._states = deque(iterable=states if states else [], maxlen=N)

        if states is not None and len(states):
            first_shape = states[0].shape
            if shape is not None and first_shape != shape:
                raise ValueError(
                    f"First shape of the values in states {first_shape} "
                    f"must have the same shape as the given shape {shape}"
                )
            self._shape = first_shape
        else:
            self._shape = shape

    def to(self, *args, **kwargs) -> "Queue":
        """Call ``.to()`` like you do with any other ``torch.Tensor``."""
        for i, state in enumerate(self._states):
            self._states[i] = state.to(*args, **kwargs)
        return self

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._states[index]

    def append(self, state: torch.Tensor) -> None:
        """Append a new state to the Queue. Set shape if not set and make sure new states have the correct shape."""
        if self._shape:
            if state.shape != self._shape:
                raise ValueError(
                    f"The shape of the new state {state.shape} "
                    f"does not match the shape of previous states {self._shape}."
                )
        else:
            self._shape = state.shape
        self._states.append(state)

    def __len__(self) -> int:
        return len(self._states)

    def get_all(self) -> torch.Tensor:
        """Get all the states from the Queue and stack them into a single torch.Tensor."""
        if len(self) == 0:
            raise ValueError("Can not stack the items of an empty Queue.")
        return torch.stack(list(self._states))

    @property
    def shape(self) -> torch.Size:
        """Get the shape of every tensor in this Queue."""
        if self._shape is None:
            raise ValueError("Can not get the shape of an empty Queue.")
        return self._shape

    @property
    def device(self) -> torch.device:
        """Get the device of every tensor in this Queue."""
        if len(self) == 0:
            raise ValueError("Can not get the device of an empty Queue.")
        device = self._states[-1].device
        assert all(state.device == device for state in self._states), "Not all tensors are on the same device"
        return device

    @property
    def N(self) -> int:
        return self._N

    def clear(self) -> None:
        """Clear all the states from the Queue."""
        self._states.clear()

    def copy(self) -> "Queue":
        """Return a (deep) copy of self."""
        return Queue(N=self._N, states=[state.detach().clone() for state in self._states], shape=self._shape)

    def __eq__(self, other: "Queue") -> bool:
        """Return whether another Queue is equal to self."""
        return self._N == other._N and self._states == other._states and self._shape == other._shape


TrackState = dict[str, torch.Tensor]
TrackStates = dict[str, Queue]


class Track:
    """A single track containing one or multiple states that are tracked as a dictionary of Queues with a max length."""

    _states: TrackStates = {}

    def __init__(self, N: int, states: TrackStates = None) -> None:
        """
        Initialize an empty track.

        Args:
            N: The maximum number of states contained in every Queue.
                Should equal the working memory size.
            states: A dict containing an initial state.
        """
        if N <= 0:
            raise ValueError(f"N must be greater than 0 but got {N}")
        self._N: int = N

        if states is not None:
            if any(q._states.maxlen != N for q in states.values()):
                raise ValueError(f"Provided states must have max_length {N} but got {states}")
            self._states = states

    def __getitem__(self, index: int) -> TrackState:
        """Get the i-th tensor of every Query."""
        return {name: q[index] for name, q in self._states.items()}

    def __len__(self) -> int:
        """get length of this state"""
        if not self.size():
            return 0

        l: int = len(iter(self._states.values()).__next__())
        if len(self._states) > 1 and any(len(q) != l for q in self._states.values()):
            raise IndexError("Queues have different length.")
        return l

    def __eq__(self, other: "Track") -> bool:
        """Compare two tracks."""
        if not isinstance(other, Track):
            return False
        return (
            self._N == other._N
            and self._states.keys() == other._states.keys()
            and all(other.get_queue(name) == q for name, q in self._states.items())
        )

    def get_states(self) -> TrackStates:
        return self._states

    def get_state(self, index: int) -> TrackState:
        """Get the i-th state of every Query."""
        if index >= self._N:
            raise IndexError(f"Index {index} is larger than the maximum number of items in a queue {self.N}")
        if index < -self._N:
            raise IndexError(f"Index {index} is smaller than the maximum number of items in a queue {self.N}")
        if any(index >= len(q) or index < -len(q) for q in self._states.values()):
            raise IndexError(f"Index {index} is out of bounds for at least one query.")
        return {name: q[index] for name, q in self._states.items()}

    def get_queue(self, name: str) -> Queue:
        """Get the Queue with the given name."""
        if name not in self._states:
            raise KeyError(f"Queue with name {name} does not exist in the current states.")
        return self._states[name]

    def get_queues(self, names: Iterable[str]) -> TrackStates:
        """Get all the Queues given a list of their names."""
        if not names:
            return {}
        if any(name not in self._states for name in names):
            raise IndexError(f"One of the provided names does not exist in the current states {self._states.keys()}.")
        return {name: self._states[name] for name in names}

    def size(self) -> int:
        """Get the number of Queues in states"""
        return len(self._states)

    def append(self, new_state: TrackState) -> None:
        """
        Right-Append new_state to the current states, but make sure that states have max length

        Args:
            new_state: pose, jcs and bbox to append to current state
        """
        if not new_state:
            raise ValueError("Can not append an empty state")
        # append new state
        for name, t in new_state.items():
            if name in self._states:
                self._states[name].append(t)
            else:
                self._states[name] = Queue(N=self._N, states=[t])

    def to(self, *args, **kwargs) -> "Track":
        """Call ``.to()`` like you do with any other ``torch.Tensor``."""

        for name, queue in self._states.items():
            self._states[name] = queue.to(*args, **kwargs)
        return self

    @property
    def names(self) -> list[str]:
        """Get all the keys of this track."""
        return [str(state) for state in self._states]

    @property
    def N(self) -> int:
        return self._N

    def copy(self) -> "Track":
        """Create a (deep) copy of this track."""
        return Track(N=self._N, states={name: q.copy() for name, q in self._states.items()})


class DataSample(UserDict):
    """Class for storing one or multiple samples of data.

    By default, the DataSample validates all new inputs.
    If you validate elsewhere, or you don't want validation for performance reasons, it can be turned off.


    During initialization, the following keys have to be given:
    -----------------------------------------------------------

    filepath
        (tuple[str]), length ``B``

        The respective filepath of every image

    bbox
        (tv_tensors.BoundingBoxes), shape ``[B x 4]``

        One single bounding box as torchvision bounding box in global coordinates.

    keypoints
        (torch.Tensor), shape ``[B x J x 2|3]``

        The key points for this bounding box as torch tensor in global coordinates.

    person_id (optional)
        (torch.LongTensor), shape ``[B]``

        The person id, only required for training and validation


    Additional Values:
    ------------------

    The model might be given additional values during initialization,
    given at any time using the underlying dict or given setters,
    or compute further optional values:

    device (Device, optional)
        The torch device to use.
        The default is the device, bbox is on.

    heatmap (torch.FloatTensor, optional)
        The heatmap of this bounding box with a shape of shape ``[B x J x h x w]``.

    image (tv_tensor.Image, optional)
        The original image, resized to the respective shape of the key point prediction model input.
        Shape ``[B x C x H x W]``

    image_crop (tv_tensor.Image, optional)
        The content of the original image cropped using the bbox. shape ``[B x C x h x w]``

    joint_weight (torch.FloatTensor, optional)
        Some kind of joint- or key-point confidence.
        E.g., the joint confidence score (JCS) of AlphaPose or the joint visibility of the PoseTrack21 dataset.
        shape ``[B x J x 1]``

    keypoints_local (torch.Tensor, optional)
        The key points for this bounding box as torch tensor in local coordinates. shape ``[B x J x 2|3]``
    """

    # there a many attributes and they can get used, so please the linter
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        filepath: FilePaths,
        bbox: tv_tensors.BoundingBoxes,
        keypoints: torch.Tensor,
        validate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self._validate = validate

        if validate:
            filepath = validate_filepath(filepath)
            bbox = validate_bboxes(bbox)
            keypoints = validate_key_points(keypoints)

            assert bbox.device == keypoints.device

        self.data["filepath"]: FilePaths = filepath
        self.data["bbox"]: tv_tensors.BoundingBoxes = (
            bbox if "device" not in kwargs else bbox.to(device=kwargs["device"])
        )
        self.data["keypoints"]: torch.Tensor = keypoints.to(device=kwargs.get("device", bbox.device))

        for k, v in kwargs.items():
            if hasattr(DataSample, k) and getattr(DataSample, k).fset is not None:
                setattr(self, k, v)
            else:
                self.data[k] = v

    def to(self, *args, **kwargs) -> "DataSample":
        """Override torch.Tensor.to() for the whole object."""
        for attr_key, attr_value in self.items():
            if isinstance(attr_value, torch.Tensor):
                self[attr_key] = attr_value.to(*args, **kwargs)
        return self

    def __len__(self) -> int:
        """Override length to be the length of the filenames"""
        return len(self.filepath)

    @property
    def person_id(self) -> torch.LongTensor:
        return self.data["person_id"].long()

    @person_id.setter
    def person_id(self, value: Union[int, torch.Tensor]) -> None:
        self.data["person_id"] = (torch.tensor(value).long() if isinstance(value, int) else value).to(
            device=self.device, dtype=torch.long
        )

    @property
    def class_id(self) -> torch.LongTensor:
        return self.data["class_id"].long()

    @class_id.setter
    def class_id(self, value: Union[int, torch.Tensor]) -> None:
        self.data["class_id"] = (torch.tensor(value).long() if isinstance(value, int) else value).to(
            device=self.device, dtype=torch.long
        )

    @property
    def filepath(self) -> FilePaths:
        """If data filepath has a single entry, return the filepath as a string, otherwise return the list."""
        assert len(self.data["filepath"]) >= 1
        return self.data["filepath"]

    @property
    def bbox(self) -> tv_tensors.BoundingBoxes:
        return self.data["bbox"]

    @property
    def keypoints(self) -> torch.Tensor:
        return self.data["keypoints"]

    @property
    def J(self) -> int:
        """Get number of joints."""
        return self.data["keypoints"].shape[-2]

    @property
    def B(self) -> int:
        """Get the batch size."""
        return len(self.filepath)

    @property
    def joint_dim(self) -> int:
        """Get the dimensionality of the joints."""
        return self.data["keypoints"].shape[-1]

    @property
    def heatmap(self) -> Heatmap:  # pragma: no cover
        """Get the heatmaps of this sample."""
        return self.data["heatmap"]

    @heatmap.setter
    def heatmap(self, value: Heatmap) -> None:  # pragma: no cover
        """Set heatmap with a little bit of validation"""
        # make sure that heatmap has shape [B x J x h x w]
        self.data["heatmap"] = (validate_heatmaps(value) if self._validate else value).to(device=self.device)

    @property
    def keypoints_local(self) -> torch.Tensor:
        """Get local keypoints."""
        return self.data["keypoints_local"]

    @keypoints_local.setter
    def keypoints_local(self, value: torch.Tensor) -> None:
        """Set local key points with a little bit of validation."""
        # use validate_key_points to make sure local key points have the correct shape [1 x J x 2|3]
        self.data["keypoints_local"] = (
            validate_key_points(value, nof_joints=self.J, joint_dim=self.keypoints.shape[-1])
            if self._validate
            else value
        ).to(device=self.device)

    @property
    def image(self) -> TVImage:
        """Get the original image or retrieve it using filepath"""
        return self.data["image"]

    @image.setter
    def image(self, value: Image) -> None:
        self.data["image"]: TVImage = (validate_images(value) if self._validate else value).to(device=self.device)

    @property
    def image_crop(self) -> TVImage:
        return self.data["image_crop"]

    @image_crop.setter
    def image_crop(self, value: Image) -> None:
        self.data["image_crop"]: TVImage = (validate_images(value) if self._validate else value).to(device=self.device)

    @property
    def crop_path(self):
        return self.data["crop_path"]

    @crop_path.setter
    def crop_path(self, value: FilePaths):
        self.data["crop_path"]: FilePaths = validate_filepath(value) if self._validate else value

    @property
    def joint_weight(self) -> torch.Tensor:
        return self.data["joint_weight"]

    @joint_weight.setter
    def joint_weight(self, value: torch.Tensor) -> None:
        self.data["joint_weight"] = value.view(self.B, self.J, 1).to(device=self.device)

    def cast_joint_weight(
        self,
        dtype: torch.dtype = torch.float32,
        decimals: int = 0,
        overwrite: bool = False,
    ) -> torch.Tensor:
        """Cast and return the joint weight as tensor.

        The weight might have an arbitrary tensor type, this function allows getting one specific variant.

        E.g., the visibility might be a boolean value or a model certainty.

        Note:
            Keep in mind,
            that torch.round() is not really differentiable and does not really allow backpropagation.
            See https://discuss.pytorch.org/t/torch-round-gradient/28628/4 for more information.

        Args:
            dtype: The new torch dtype of the tensor.
                Default torch.float32.
            decimals: Number of decimals to round floats to, before type casting.
                Default 0 (round to integer).
                When the value of decimals is set to -1 (minus one),
                there will only be type casting and no rounding at all.
                But keep in mind that when information is compressed, e.g., when casting from float to bool,
                simply calling float might not be enough to cast 0.9 to True.
            overwrite: Whether self.joint_weight will be overwritten or not.

        Returns:
            A type-cast version of the tensor.

            If overwrite is True, the returned tensor will be the same as `self.joint_weight`,
            including the computational graph.

            If overwrite is False, the returned tensor will be a detached and cloned instance of `self.joint_weight`.
        """
        new_weights = self.joint_weight.detach().clone()
        # round
        if decimals >= 0:
            # round needs floating point tensor
            if not torch.is_floating_point(new_weights):
                new_weights = new_weights.to(dtype=torch.float32)
            new_weights.round_(decimals=decimals)
        # change dtype
        new_weights = new_weights.to(dtype=dtype)
        # overwrite existing value if requested
        if overwrite:
            self.joint_weight = new_weights
        return new_weights

    @property
    def device(self):
        """Get the device of the DataSample. Defaults to the device of self.bbox if nothing is given."""
        if "device" not in self.data:
            self.data["device"] = self.bbox.device
        return self.data["device"]

    @device.setter
    def device(self, value):
        self.data["device"] = torch.device(value)


def get_ds_data_getter(attributes: list[str]) -> DataGetter:
    """Given a list of attribute names, return a function, that gets those attributes from a given DataSample."""

    def getter(ds: DataSample) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The getter function."""
        return tuple(ds[str(attrib)] for attrib in attributes)

    return getter
