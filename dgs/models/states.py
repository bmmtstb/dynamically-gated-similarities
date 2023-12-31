"""
definitions and helpers for pose-state(s)
"""
from collections import UserDict
from typing import Union

import torch
from torchvision import tv_tensors

from dgs.utils.types import Config, Device, FilePaths, Heatmap, Image, PoseStateTuple, TVImage
from dgs.utils.validation import (
    validate_bboxes,
    validate_filepath,
    validate_heatmaps,
    validate_images,
    validate_key_points,
)


class PoseState:
    """
    PoseState = (Pose, JCS, BBox)
    """

    pose: torch.Tensor
    jcs: torch.Tensor
    bbox: torch.Tensor

    def __init__(self, pose: torch.Tensor, jcs: torch.Tensor, bbox: torch.Tensor) -> None:
        """

        Args:
            pose: pose or joint coordinates or the current state
            jcs: joint confidence scores of this pose
            bbox: bounding box coordinates relative to the full image
        """
        self.pose = pose
        self.jcs = jcs
        self.bbox = bbox

    def __eq__(self, other) -> bool:
        """Redefine the equality between two PoseState objects.

        Args:
            other: PoseState object or PoseStateTuple to compare to.

        Returns:
            bool: Whether the two PoseState are equal.
        """
        if isinstance(other, PoseState):
            return (
                torch.equal(self.pose, other.pose)
                and torch.equal(self.jcs, other.jcs)
                and torch.equal(self.bbox, other.bbox)
            )
        if isinstance(other, tuple):
            return (
                torch.equal(self.pose, other[0])
                and torch.equal(self.jcs, other[1])
                and torch.equal(self.bbox, other[2])
            )
        raise NotImplementedError(f"Equality between PoseState and {type(other)} is not defined.")

    def to(self, *args, **kwargs) -> "PoseState":
        """
        Override torch.Tensor.to() to work with PoseState class

        Args:
            See .to() of torch.

            Examples:
                device="cuda"
                dtype=torch.int
        """
        self.pose = self.pose.to(*args, **kwargs)
        self.jcs = self.jcs.to(*args, **kwargs)
        self.bbox = self.bbox.to(*args, **kwargs)
        return self

    def __getitem__(self, item: str | int) -> torch.Tensor:
        """Override PoseState["item"] to be class-specific.

        Args:
            item: Name of the value to retrieve, has to be in `["pose", "jcs", "bbox"]`.

        Returns:
            State with the given name as `torch.Tensor`.
        """
        if isinstance(item, str):
            return self.__getattribute__(str(item))
        return self.__getstate__()[item]

    def __getstate__(self) -> PoseStateTuple:
        """
        Returns:
            PoseState as tuple of torch.Tensor
        """
        return self.pose, self.jcs, self.bbox

    def __setstate__(self, state: PoseStateTuple) -> None:
        """
        Args:
            state: PoseState or tuple of three torch tensors
        """
        self.pose = state[0]
        self.jcs = state[1]
        self.bbox = state[2]


class PoseStates:
    """
    Custom "queue" of pose states with a max length.
    Every track has one PoseStates object containing the history of detected poses, joint confidence scores (JCS), and
    bounding box information (bbox).
    """

    def __init__(self, config: Config, max_length: int = 30) -> None:
        """
        Initialize empty queue with max length.

        Args:
            config: the current programs configuration
            max_length: maximum number of pose states contained. Equals working memory size.
        """
        self.config: Config = config

        self.max_length: int = max_length  # FIXME: get from params ?

        self.poses: list[torch.Tensor] = []
        self.jcss: list[torch.Tensor] = []
        self.bboxes: list[torch.Tensor] = []

    def _stack_tensor(self, lot: list[torch.Tensor], copy: bool = False) -> torch.Tensor:
        """Stack state and create a copy of the tensor"""
        if copy:
            return torch.stack(lot).detach().clone().to(self.config["device"])
        return torch.stack(lot).to(self.config["device"])

    def get_states(self, items: int | slice = None, copy: bool = False) -> PoseState:
        """Obtain a copy of the three states within this queue.
        Due to multiprocessing, we technically have to freeze appending to ensure equal length and matching indices at
        all times, but as long as it doesn't make problems, this will be postponed.

        Args:
            items: Index or slice of the state(s) to retrieve.
            copy: Whether to create a detached and cloned copy of the current states or the real tensors.

        Returns:
            Three stacked tensors of cloned and detached current state on the configured device.
        """
        if isinstance(items, int):
            # to be able to use torch.stack later, make sure to keep a list and not the single tensors
            items: slice = slice(items, items)
        elif items is None:
            items: slice = slice(len(self))

        return PoseState(
            self._stack_tensor(self.poses[items], copy=copy),
            self._stack_tensor(self.jcss[items], copy=copy),
            self._stack_tensor(self.bboxes[items], copy=copy),
        )

    def __getitem__(self, item: int | slice) -> PoseState | list[PoseState]:
        """Override get-item call (PoseStates[i]) to obtain pose state by indices.
        Supports python indexing using slices.
        Returns the exact torch tensor because it is not possible to add further parameters to this call.
        Therefore, if you want to obtain a detached and cloned tensor use self.get_state(..., copy=True)

        Args:
            item: index or slice of the states to obtain

        Returns:
            Either a single pose state given an integer item or a list of pose states given a slice.
        """
        if isinstance(item, int):
            return PoseState(self.poses[item], self.jcss[item], self.bboxes[item])
        return PoseState(self._stack_tensor(self.poses), self._stack_tensor(self.jcss), self._stack_tensor(self.bboxes))

    def __len__(self) -> int:
        """get length of this state"""
        if len(self.jcss) == len(self.poses) == len(self.bboxes):
            return len(self.jcss)
        raise IndexError("Lists in PoseStates have different length.")

    def __iadd__(self, other: PoseState):
        """
        Override += to use append()

        Args:
            other: Tuple of pose state to append to self.

        Returns:
            Updated version of self.
        """
        self.append(other)
        return self

    def append(self, new_state: PoseState | PoseStateTuple) -> None:
        """
        Right-Append new_state to the current states, but make sure that states have max length

        Args:
            new_state: pose, jcs and bbox to append to current state
        """
        # pop old state if too long
        if len(self) >= self.max_length:
            self.poses.pop(0)
            self.jcss.pop(0)
            self.bboxes.pop(0)
        # append new state
        pose, jcs, bbox = new_state
        self.poses.append(pose.to(self.config["device"]))
        self.jcss.append(jcs.to(self.config["device"]))
        self.bboxes.append(bbox.to(self.config["device"]))


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
        (torch.IntTensor), shape ``[B]``

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
        The original image, resized to the respective shape of the backbone input. Shape ``[B x C x H x W]``

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
        filepath: tuple[str, ...],
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
        for attr_value in self.values():
            if isinstance(attr_value, torch.Tensor):
                attr_value.to(*args, **kwargs)
        return self

    def __str__(self) -> str:
        """Overwrite representation to be image name."""
        return f"{self.data['filepath']} {self.data['bbox']}"

    @property
    def person_id(self) -> torch.IntTensor:
        return self.data["person_id"]

    @person_id.setter
    def person_id(self, value: Union[int, torch.IntTensor]) -> None:
        self.data["person_id"] = (torch.tensor(value).int() if isinstance(value, int) else value).to(device=self.device)

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
    def B(self):
        """Get the batch size."""
        # fixme this does not work during initialization
        # if not (self.bbox.shape[-2] == len(self.filepath) == self.keypoints.shape[-3]):
        #     raise ValueError("Obtained different batch sizes from bbox, key points and filepaths")
        return self.bbox.shape[-2]

    @property
    def joint_dim(self) -> int:
        """Get the dimensionality of the joints."""
        return self.data["keypoints"].shape[-1]

    @property
    def heatmap(self) -> Heatmap:
        """Get the heatmaps of this sample."""
        return self.data["heatmap"]

    @heatmap.setter
    def heatmap(self, value: Heatmap) -> None:
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
        device: Device = None,
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
            device: Which device the tensor is being sent to.
                Defaults to the device visibility on which the visibility tensor lies.

        Returns:
            A type-cast version of the tensor.

            If overwrite is True, the returned tensor will be the same as `self.joint_weight`,
            including the computational graph.

            If overwrite is False, the returned tensor will be a detached and cloned instance of `self.joint_weight`.
        """
        if overwrite and decimals >= 0:
            return self.joint_weight.round_(decimals=decimals).to(device=device, dtype=dtype)
        if overwrite:
            return self.joint_weight.to(device=device, dtype=dtype)
        new_weights = self.joint_weight.detach().clone().to(device=device, dtype=dtype)
        if decimals >= 0:
            new_weights.round_(decimals=decimals)
        return new_weights.to(device=device, dtype=dtype)

    @property
    def device(self):
        """Get the device of the DataSample. Defaults to the device of self.bbox if nothing is given."""
        if "device" not in self.data:
            self.data["device"] = self.bbox.device
        return self.data["device"]

    @device.setter
    def device(self, value):
        self.data["device"] = torch.device(value)
