"""
Definitions and helpers for a state.

Contains the custom collate functions to combine a list of States into a single one,
keeping custom tensor subtypes intact.
"""

from collections import UserDict
from copy import deepcopy
from typing import Callable, Type, Union

import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision import tv_tensors

from dgs.utils.image import load_image
from dgs.utils.types import DataGetter, FilePath, FilePaths, Heatmap, Image
from dgs.utils.validation import (
    validate_bboxes,
    validate_filepath,
    validate_heatmaps,
    validate_images,
    validate_key_points,
)


class State(UserDict):
    """Class for storing one or multiple samples of data as a 'State'.

    By default, this object validates all new inputs.
    If you validate elsewhere, use an existing dataset,
    or you don't want validation for performance reasons, validation can be turned off.

    The model might be given additional values during initialization,
    or at any time using the given setters or the get_item call.
    Additionally, the object can compute / load further values.

    All args and keyword args can be accessed using the States' properties.
    Additionally, the underlying dict structure can be used,
    but this does not allow validation nor on the fly computation of additional values.

    Args:
        bbox (tv_tensors.BoundingBoxes): One single bounding box as torchvision bounding box in global coordinates.
            Shape ``[B x 4]``
        kwargs: Additional keyword arguments as shown below.

    Keyword Args:
        keypoints (torch.Tensor): The key points for this bounding box as torch tensor in global coordinates.
            Shape ``[B x J x 2|3]``
        filepath (tuple[str]): The respective filepath of every image.
            Length ``B``.
        person_id (torch.Tensor): The person id, only required for training and validation.
            Shape ``[B]``.
        class_id (torch.Tensor): The class id, only required for training and validation.
            Shape ``[B]``.
        device (Device): The torch device to use.
            If the device is not given, the device the bbox is on is used as a default.
        heatmap (torch.Tensor)
            The heatmap of this bounding box with a shape of shape ``[B x J x h x w]``.
            Currently not used.
        image (tv_tensor.Image): The original image,
            resized to the respective shape of the key point prediction model input.
            Shape ``[B x C x H x W]``
        image_crop (tv_tensor.Image): The content of the original image cropped using the bbox.
           Shape ``[B x C x h x w]``
        joint_weight (torch.Tensor): Some kind of joint- or key-point confidence.
            E.g., the joint confidence score (JCS) of AlphaPose or the joint visibility of |PT21|.
            Shape ``[B x J x 1]``
        keypoints_local (torch.Tensor): The key points for this bounding box as torch tensor in local coordinates.
            Shape ``[B x J x 2|3]``
    """

    # there a many attributes and they can get used, so please the linter
    # pylint: disable=too-many-instance-attributes, too-many-public-methods

    validate: bool
    """Whether to validate the inputs into this state."""
    data: dict[str, any]
    """All the data in this state as a dict.
    Can be accessed to set its values, but as long as possible you should use the property setters."""

    def __init__(
        self,
        *args,
        bbox: tv_tensors.BoundingBoxes,
        validate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.validate = validate

        if validate:
            bbox = validate_bboxes(bbox)
            if args:
                raise NotImplementedError(f"Unknown arguments: {args}")

        self.data["bbox"]: tv_tensors.BoundingBoxes = bbox.to(device=kwargs.get("device", bbox.device))

        for k, v in kwargs.items():
            if hasattr(State, k) and getattr(State, k).fset is not None:
                setattr(self, k, v)
            else:
                self.data[k] = v

    def __len__(self) -> int:
        """Override length to be the batch-length of the bounding-boxes."""
        return self.bbox.size(-2)

    def copy(self) -> "State":
        """Obtain a copy of this state. No validation, either it was done already or it is not wanted."""
        data = {k: v.detach().clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k, v in self.data.items()}
        data["validate"] = False
        state = State(**data)
        state.validate = self.validate
        return state

    def __eq__(self, other: "State") -> bool:
        """Override State equality."""
        if not isinstance(other, State):
            return False
        return (
            self.validate == other.validate
            and self.data.keys() == other.data.keys()
            and all(
                torch.allclose(v, other.data[k]) if isinstance(v, torch.Tensor) else v == other.data[k]
                for k, v in self.data.items()
            )
        )

    @property
    def bbox(self) -> tv_tensors.BoundingBoxes:
        """Get this States bounding-box."""
        return self.data["bbox"]

    @bbox.setter
    def bbox(self, bbox: tv_tensors) -> None:
        raise NotImplementedError(
            "It is not allowed to change the bounding box of an already existing State object. "
            "Create a new object instead!"
        )

    @property
    def device(self):
        """Get the device of this State. Defaults to the device of self.bbox if nothing is given."""
        if "device" not in self.data:
            self.data["device"] = self.bbox.device
        return self.data["device"]

    @device.setter
    def device(self, value):
        """Device can be changed using to() or during initialization."""
        self.data["device"] = torch.device(value)

    # ################## #
    # REGULAR PROPERTIES #
    # ################## #

    @property
    def B(self) -> int:
        """Get the batch size."""
        return len(self)

    @property
    def J(self) -> int:
        """Get the number of joints in every skeleton."""
        if "keypoints" in self.data and self.data["keypoints"].ndim > 2:
            return self.data["keypoints"].shape[-2]
        if "keypoints_local" in self.data and self.data["keypoints_local"].ndim > 2:
            return self.data["keypoints_local"].shape[-2]
        raise NotImplementedError("There are no global or local key-points in this object.")

    @property
    def joint_dim(self) -> int:
        """Get the dimensionality of the joints."""
        if "keypoints" in self.data:
            return self.data["keypoints"].shape[-1]
        if "keypoints_local" in self.data:
            return self.data["keypoints_local"].shape[-1]
        raise NotImplementedError("There are no global or local key-points in this object.")

    # ###################### #
    # PROPERTIES AND SETTERS #
    # for regular attributes #
    # ###################### #

    @property
    def person_id(self) -> torch.Tensor:
        """Get the ID of the respective person shown on the bounding-box."""
        return self.data["person_id"].long()

    @person_id.setter
    def person_id(self, value: Union[int, torch.Tensor]) -> None:
        self.data["person_id"] = (torch.tensor(value).long() if isinstance(value, int) else value).to(
            device=self.device, dtype=torch.long
        )

    @property
    def class_id(self) -> torch.Tensor:
        """Get the class-ID of the bounding-boxes."""
        return self.data["class_id"].long()

    @class_id.setter
    def class_id(self, value: Union[int, torch.Tensor]) -> None:
        self.data["class_id"] = (torch.tensor(value).long() if isinstance(value, int) else value).to(
            device=self.device, dtype=torch.long
        )

    @property
    def track_id(self) -> torch.Tensor:
        """Get the ID of the tracks associated with the respective bounding-boxes."""
        return self.data["track_id"].long()

    @track_id.setter
    def track_id(self, value: Union[int, torch.Tensor]) -> None:
        self.data["track_id"] = (torch.tensor(value).long() if isinstance(value, int) else value).to(
            device=self.device, dtype=torch.long
        )

    @property
    def filepath(self) -> FilePaths:
        """If data filepath has a single entry, return the filepath as a string, otherwise return the list."""
        assert isinstance(self.data["filepath"], tuple), f"filepath must be a tuple but got {self.data['filepath']}"
        return self.data["filepath"]

    @filepath.setter
    def filepath(self, fp: Union[FilePath, FilePaths]) -> None:
        if isinstance(fp, tuple):
            if len(fp) != self.B:
                raise ValueError(
                    f"filepath must have the same number of entries as bounding-boxes. Got {len(fp)}, expected {self.B}"
                )
            self.data["filepath"] = validate_filepath(file_paths=fp, length=self.B) if self.validate else fp
            return
        if isinstance(fp, str):
            if self.B != 1:
                raise ValueError(
                    f"filepath must have the same number of entries as bounding-boxes. "
                    f"Got a single path, expected {self.B}"
                )
            self.data["filepath"] = validate_filepath(file_paths=fp, length=self.B) if self.validate else (fp,)
            return
        raise NotImplementedError(f"Unknown filepath format: {type(fp)}, path: {fp}")

    @property
    def keypoints(self) -> torch.Tensor:
        """Get the key-points. The coordinates are based on the coordinate-frame of the full-image."""
        return self.data["keypoints"]

    @keypoints.setter
    def keypoints(self, value: torch.Tensor) -> None:
        try:
            J = self.J
            j_dim = self.joint_dim
        except NotImplementedError:
            J = None
            j_dim = None
        self.data["keypoints"] = (
            validate_key_points(key_points=value, joint_dim=j_dim, nof_joints=J) if self.validate else value
        ).to(device=self.device)

    @property
    def heatmap(self) -> Heatmap:  # pragma: no cover
        """Get the heatmaps of this State."""
        return self.data["heatmap"]

    @heatmap.setter
    def heatmap(self, value: Heatmap) -> None:  # pragma: no cover
        """Set heatmap with a little bit of validation"""
        # make sure that heatmap has shape [B x J x h x w]
        self.data["heatmap"] = (validate_heatmaps(value) if self.validate else value).to(device=self.device)

    @property
    def keypoints_local(self) -> torch.Tensor:
        """Get the local key-points.
        The local coordinates are based on the coordinate-frame of the image crops, within the bounding-box.
        """
        return self.data["keypoints_local"]

    @keypoints_local.setter
    def keypoints_local(self, value: torch.Tensor) -> None:
        """Set local key points with a little bit of validation."""
        try:
            J = self.J
            j_dim = self.joint_dim
        except NotImplementedError:
            J = None
            j_dim = None

        # use validate_key_points to make sure local key points have the correct shape [1 x J x 2|3]
        self.data["keypoints_local"] = (
            validate_key_points(value, nof_joints=J, joint_dim=j_dim) if self.validate else value
        ).to(device=self.device)

    @property
    def image(self) -> Image:
        """Get the original image(s) of this State.
        If the images are not available, try to load them using :func:`load_image` and :attr:`filepath`.
        """
        if "image" not in self.data:
            return self.load_image()
        return self.data["image"]

    @image.setter
    def image(self, value: Image) -> None:
        self.data["image"]: Image = (validate_images(value) if self.validate else value).to(device=self.device)

    @property
    def image_crop(self) -> Image:
        """Get the image crop(s) of this State.
        If the crops are not available, try to load them using :func:`load_image_crop` and :attr:`crop_path`.
        """
        if "image_crop" not in self.data:
            self.load_image_crop()
        return self.data["image_crop"]

    @image_crop.setter
    def image_crop(self, value: Image) -> None:
        self.data["image_crop"]: Image = (validate_images(value) if self.validate else value).to(device=self.device)

    @property
    def crop_path(self):
        """Get the path to the image crops. Only necessary if the image crops are saved and not computed live."""
        return self.data["crop_path"]

    @crop_path.setter
    def crop_path(self, value: FilePaths):
        self.data["crop_path"]: FilePaths = validate_filepath(value) if self.validate else value

    @property
    def joint_weight(self) -> torch.Tensor:
        """Get the weight of the joints. Either represents the visibility or an importance score of this joint."""
        return self.data["joint_weight"]

    @joint_weight.setter
    def joint_weight(self, value: torch.Tensor) -> None:
        self.data["joint_weight"] = (value.view(self.B, self.J, 1) if self.validate else value).to(device=self.device)

    # ######### #
    # FUNCTIONS #
    # ######### #

    def extract(self, idx: int) -> "State":
        r"""Extract the i-th State from a batch B of states.

        Args:
            idx: The index of the State to retrieve.
                It is expected that :math:`-B \lte idx \lt B`.

        Returns:
            The extracted State.
        """
        if idx >= self.B or idx < -self.B:
            raise IndexError(f"Expected index to lie within ({-self.B}, {self.B - 1}), but got: {idx}")

        new_data = {"validate": self.validate}
        for k, v in self.data.items():
            ks = str(k)
            if isinstance(v, tv_tensors.TVTensor):
                # make sure tv_tensors stay, especially for bboxes
                new_data[ks] = tv_tensors.wrap(v[idx], like=v)
            elif isinstance(v, list):
                # lists stay list
                new_data[ks] = [v[idx]]
            elif isinstance(v, tuple):
                # tuples stay tuple
                new_data[ks] = (v[idx],)
            elif isinstance(v, torch.Tensor) and v.ndim == 0:
                new_data[ks] = v
            elif hasattr(v, "__getitem__"):
                # every other iterable data -> regular tensors, ...
                new_data[ks] = v[idx]
            else:
                new_data[ks] = v
        assert "bbox" in new_data, "No Bounding box given while creating the state."
        return State(**new_data)  # pylint: disable=missing-kwoa

    def split(self) -> list["State"]:
        """Given a batched State object, split it into a list of single State objects."""
        if self.B == 1:
            return [self]
        new_data = [{"validate": self.validate} for _ in range(self.B)]
        for k, v in self.data.items():
            ks = str(k)
            for idx in range(self.B):
                if isinstance(v, tv_tensors.TVTensor):
                    # make sure tv_tensors stay, especially for bboxes
                    new_data[idx][ks] = tv_tensors.wrap(v[idx], like=v)
                elif isinstance(v, list):
                    # lists stay list
                    new_data[idx][ks] = [v[idx]]
                elif isinstance(v, tuple):
                    # tuples stay tuple
                    new_data[idx][ks] = (v[idx],)
                elif hasattr(v, "__getitem__"):
                    # every other iterable data -> regular tensors, ...
                    new_data[idx][ks] = v[idx]
                else:
                    new_data[idx][ks] = v
        assert all("bbox" in d for d in new_data), "No Bounding box given while creating the state."
        return [State(**d) for d in new_data]  # pylint: disable=missing-kwoa

    def load_image_crop(self, **kwargs) -> Image:
        """Load the images crops using the crop_paths of this object. Does nothing if the crops are already present."""
        if "image_crop" in self.data and self.data["image_crop"] is not None and len(self.data["image_crop"]) == self.B:
            return self.image_crop
        if "crop_path" not in self.data:
            raise AttributeError("Could not load image crops without proper filepaths given.")
        self.image_crop = load_image(filepath=self.crop_path, device=self.device, *kwargs)
        return self.image_crop

    def load_image(self, **kwargs) -> Image:
        """Load the images using the filepaths of this object. Does nothing if the images are already present."""
        if "image" in self.data and self.data["image"] is not None and len(self.data["image"]) == self.B:
            return self.image
        if "filepath" not in self.data:
            raise AttributeError("Could not load images without proper filepaths given.")
        self.image = load_image(filepath=self.filepath, device=self.device, *kwargs)
        return self.image

    def to(self, *args, **kwargs) -> "State":
        """Override torch.Tensor.to() for the whole object."""

        for attr_key, attr_value in self.items():
            if isinstance(attr_value, torch.Tensor) or (
                hasattr(attr_value, "to") and callable(getattr(attr_value, "to"))
            ):
                self[attr_key] = attr_value.to(*args, **kwargs)
        self.device = self.bbox.device
        return self

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


def get_ds_data_getter(attributes: list[str]) -> DataGetter:
    """Given a list of attribute names,
    return a function, that gets those attributes from a given :class:`State`.
    """

    def getter(ds: State) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The getter function."""
        return tuple(ds[str(attrib)] for attrib in attributes)

    return getter


def collate_devices(batch: list[torch.device], *_args, **_kwargs) -> torch.device:
    """Collate a batch of devices into a single device."""
    return batch[0]


def collate_tensors(batch: list[torch.Tensor], *_args, **_kwargs) -> torch.Tensor:
    """Collate a batch of tensors into a single one.

    Will use torch.cat() if the first dimension has a shape of one, otherwise torch.stack()
    """
    if len(batch) == 0:
        return torch.empty(0)
    if len(batch[0].shape) > 0 and batch[0].shape[0] == 1:
        return torch.cat(batch)
    return torch.stack(batch)


def collate_bboxes(batch: list[tv_tensors.BoundingBoxes], *_args, **_kwargs) -> tv_tensors.BoundingBoxes:
    """Collate a batch of bounding boxes into a single one.
    It is expected that all bounding boxes have the same canvas size and format.
    """
    if len(batch) == 0:
        return tv_tensors.BoundingBoxes(torch.empty((0, 4)), canvas_size=(0, 0), format="XYXY")
    bb_format: tv_tensors.BoundingBoxFormat = batch[0].format
    canvas_size = batch[0].canvas_size

    return tv_tensors.BoundingBoxes(
        torch.cat(batch),
        canvas_size=canvas_size,
        format=bb_format,
    )


def collate_tvt_tensors(
    batch: list[Union[tv_tensors.Image, tv_tensors.Mask, tv_tensors.Video]], *_args, **_kwargs
) -> Union[tv_tensors.TVTensor, tv_tensors.Image, tv_tensors.Mask, tv_tensors.Video]:
    """Collate a batch of tv_tensors into a batched version of it."""
    if len(batch) == 0:
        return tv_tensors.TVTensor([])
    if len(batch[0].shape) > 0 and batch[0].size(0) == 1:
        return tv_tensors.wrap(torch.cat(batch), like=batch[0])
    return tv_tensors.wrap(torch.stack(batch), like=batch[0])


CUSTOM_COLLATE_MAP: dict[Type, Callable] = default_collate_fn_map.copy()
CUSTOM_COLLATE_MAP.update(  # pragma: no cover
    {
        str: lambda str_batch, *args, **kwargs: tuple(s for s in str_batch),
        tuple: lambda t_batch, *args, **kwargs: sum(t_batch, ()),
        tv_tensors.BoundingBoxes: collate_bboxes,
        (tv_tensors.Image, tv_tensors.Video, tv_tensors.Mask): collate_tvt_tensors,
        torch.device: collate_devices,
        torch.Tensor: collate_tensors,  # override regular tensor collate to *not* add another dimension
    }
)


def collate_states(batch: Union[list["State"], "State"]) -> "State":
    """Collate function for multiple States, to flatten / squeeze the shapes and keep the tv_tensors classes.

    The default collate function messes up a few of the dimensions and removes custom tv_tensor classes.
    Therefore, add custom collate functions for the tv_tensors classes.
    Additionally, custom torch tensor collate, which stacks tensors only if first dimension != 1, cat otherwise.

    Args:
        batch: A list of :class:`.State`, each State contains the data belonging to a single bounding-box.

    Returns:
        One single `State` object, containing a batch of data belonging to the bounding-boxes.
    """
    if isinstance(batch, State):
        return batch

    c_batch: dict[str, any] = collate(batch, collate_fn_map=CUSTOM_COLLATE_MAP)

    # skip validation, because either every State has been validated before or validation is not required.
    s = State(**c_batch, validate=False)
    # then set the validation to the correct value
    s.validate = batch[0].validate
    return s
