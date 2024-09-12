"""
Definitions and helpers for a state.

Contains the custom collate functions to combine a list of States into a single one,
keeping custom tensor subtypes intact.
"""

import os
import warnings
from collections import UserDict
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, Type, Union

import torch as t
from matplotlib import pyplot as plt
from torch.utils.data._utils.collate import collate
from torchvision import tv_tensors as tvte
from torchvision.transforms.v2.functional import convert_image_dtype
from torchvision.utils import save_image

from dgs.utils.constants import COLORS, SKELETONS
from dgs.utils.files import is_file, mkdir_if_missing
from dgs.utils.image import load_image, load_image_list
from dgs.utils.types import DataGetter, FilePath, FilePaths, Image, Images
from dgs.utils.utils import extract_crops_from_images, replace_file_type
from dgs.utils.validation import (
    validate_bboxes,
    validate_filepath,
    validate_image,
    validate_images,
    validate_key_points,
)
from dgs.utils.visualization import show_image_with_additional


class State(UserDict):
    """Class for storing one or multiple samples of data as a 'State'.

    Batch Size
    ----------

    Even if the batch size of a State is 1, or even zero (!),
    the dimension containing the batch size should always be present.

    Validation
    ----------

    By default, this object validates all new inputs.
    If you validate elsewhere, use an existing dataset,
    or you don't want validation for performance reasons, validation can be turned off.


    Additional Values
    -----------------

    The model might be given additional values during initialization,
    or at any time using the given setters or the get_item call.
    Additionally, the object can compute / load further values.

    All args and keyword args can be accessed using the States' properties.
    Additionally, the underlying dict structure ('self.data') can be used,
    but this does not allow validation nor on the fly computation of additional values.
    So make sure you do so, if needed.

    keypoints (:class:`torch.Tensor`)
        The key points for this bounding box as torch tensor in global coordinates.

        Shape ``[B x J x 2|3]``

    filepath (:obj:`.FilePaths`)
        The respective filepath(s) of every image.

        Length ``B``.

    person_id (:class:`torch.Tensor`)
        The person id, only required for training and validation.

        Shape ``[B]``.

    class_id (:class:`torch.Tensor`)
        The class id, only required for training and validation.

        Shape ``[B]``.

    device (:obj:`.Device`)
        The torch device to use.
        If the device is not given, the device of :attr:`bbox` is used as the default.

    heatmap (:class:`torch.Tensor`)
        The heatmap of this bounding box.
        Currently not used.

        Shape ``[B x J x h x w]``.

    image (:obj:`.Images`)
        A list containing the original image(s) as :class:`tv_tensors.Image` object.

        A list of length ``B`` containing images of shape ``[1 x C x H x W]``.

    image_crop (:obj:`.Image`)
        The content of the original image cropped using the bbox.

        Shape ``[B x C x h x w]``

    joint_weight (:class:`torch.Tensor`)
        Some kind of joint- or key-point confidence.
        E.g., the joint confidence score (JCS) of AlphaPose or the joint visibility of |PT21|.

        Shape ``[B x J x 1]``

    keypoints_local (:class:`torch.Tensor`)
        The key points for this bounding box as torch tensor in local coordinates.

        Shape ``[B x J x 2|3]``

    Args:
        bbox (tv_tensors.BoundingBoxes): One single bounding box as torchvision bounding box in global coordinates.

            Shape ``[B x 4]``
        kwargs: Additional keyword arguments as shown in the 'Additional Values' section.
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
        bbox: tvte.BoundingBoxes,
        validate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.validate = validate

        if validate:
            bbox = validate_bboxes(bbox)
            if args:
                raise NotImplementedError(f"Unknown arguments: {args}")

        self.data["bbox"]: tvte.BoundingBoxes = bbox.to(device=kwargs.get("device", bbox.device))

        for k, v in kwargs.items():
            if v is None:
                continue
            if hasattr(State, k) and getattr(State, k).fset is not None:
                setattr(self, k, v)
            else:
                self.data[k] = v

    def __len__(self) -> int:
        """Override length to be the batch-length of the bounding-boxes."""
        return self.bbox.size(-2)

    def copy(self) -> "State":
        """Obtain a copy of this state. No validation, either it was done already or it is not wanted."""
        data = {k: v.detach().clone() if isinstance(v, t.Tensor) else deepcopy(v) for k, v in self.data.items()}
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
                (
                    # tensor equality
                    t.allclose(v, other.data[k])
                    if isinstance(v, t.Tensor)
                    else (
                        # check for iterable of tensors
                        len(other.data[k]) == len(v)
                        and all(t.allclose(sub_v, sub_o) for sub_v, sub_o in zip(v, other.data[k]))
                        if isinstance(v, Iterable) and any(isinstance(sub_v, t.Tensor) for sub_v in v)
                        # regular equality
                        else v == other.data[k]
                    )
                )
                for k, v in self.data.items()
            )
        )

    def __getitem__(self, item: any) -> any:
        """Override the getitem call.
        Use strings to get the keys of this dict.
        Use integers or slices to extract parts of this :class:`State`.

        Returns:
            A :class:`State` if item is int or slice, and any if item is a string.
        """
        if isinstance(item, str):
            return self.data[item]
        if isinstance(item, int):
            return self.extract(item)
        if isinstance(item, slice):
            return collate_states(self.split()[item])
        raise NotImplementedError(f"Expected item to be str, int or slice, got {type(item)}")

    @property
    def bbox(self) -> tvte.BoundingBoxes:
        """Get this States bounding-box."""
        return self.data["bbox"]

    @bbox.setter
    def bbox(self, bbox: tvte) -> None:
        if not isinstance(bbox, tvte.BoundingBoxes):
            raise TypeError(f"Expected bounding box, got {type(bbox)}")
        if bbox.shape != self.bbox.shape:
            raise ValueError(f"Can't switch bbox shape. Expected {self.bbox.shape} but got {bbox.shape}")

        # switch device if new bbox is on another device
        if bbox.device != self.data["bbox"].device:
            self.device = bbox.device
            self.to(device=bbox.device)

        # set new bbox
        self.data["bbox"] = bbox

    @property
    def device(self):
        """Get the device of this State. Defaults to the device of self.bbox if nothing is given."""
        if "device" not in self.data:
            self.data["device"] = self.bbox.device
        return self.data["device"]

    @device.setter
    def device(self, value):
        """Device can be changed using to() or during initialization."""
        self.data["device"] = t.device(value)

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
    def person_id(self) -> t.Tensor:
        """Get the ID of the respective person shown on the bounding-box."""
        return self.data["person_id"].long()

    @person_id.setter
    def person_id(self, value: Union[int, t.Tensor]) -> None:
        self.data["person_id"] = (t.tensor(value).flatten().long() if isinstance(value, int) else value).to(
            device=self.device, dtype=t.long
        )

    @property
    def class_id(self) -> t.Tensor:
        """Get the class-ID of the bounding-boxes."""
        return self.data["class_id"].long()

    @class_id.setter
    def class_id(self, value: Union[int, t.Tensor]) -> None:
        self.data["class_id"] = (t.tensor(value).flatten().long() if isinstance(value, int) else value).to(
            device=self.device, dtype=t.long
        )

    @property
    def track_id(self) -> t.Tensor:
        """Get the ID of the tracks associated with the respective bounding-boxes."""
        return self.data["track_id"].long()

    @track_id.setter
    def track_id(self, value: Union[int, t.Tensor]) -> None:
        self.data["track_id"] = (t.tensor(value).flatten().long() if isinstance(value, int) else value).to(
            device=self.device, dtype=t.long
        )

    @property
    def filepath(self) -> FilePaths:
        """If data filepath has a single entry, return the filepath as a string, otherwise return the list."""
        assert "filepath" in self, "filepath not set"
        assert isinstance(self.data["filepath"], tuple), f"filepath must be a tuple but got {self.data['filepath']}"
        return self.data["filepath"]

    @filepath.setter
    def filepath(self, fp: Union[FilePath, FilePaths]) -> None:
        if not self.validate:
            self.data["filepath"] = fp if isinstance(fp, tuple) else (fp,)
            return
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
    def keypoints(self) -> t.Tensor:
        """Get the key-points.
        The coordinates are based on the coordinate-frame of the full-image.

        Optionally loads the key-points from the 'keypoints_path' if given.
        Otherwise, tries to load the key-points from the 'crop_path' with '_glob.pt' ending if given.
        If one of the loading methods is used, the `joint_weight` will be set.
        """
        if "keypoints" in self:
            return self.data["keypoints"]

        if "keypoints_path" in self:
            if isinstance(self["keypoints_path"], str):
                self["keypoints_path"] = tuple(self["keypoints_path"] for _ in range(self.B))
            if not isinstance(self["keypoints_path"], tuple):
                raise NotImplementedError("Unknown format of keypoints_path.")

            self.keypoints = self.keypoints_and_weights_from_paths(self["keypoints_path"])
            return self["keypoints"]

        if "crop_path" in self:
            if isinstance(self["crop_path"], str):
                self["crop_path"] = tuple(self["crop_path"] for _ in range(self.B))
            if not isinstance(self["crop_path"], tuple):
                raise NotImplementedError("Unknown crop_path format.")

            self.keypoints = self.keypoints_and_weights_from_paths(
                tuple(replace_file_type(cp, new_type="_glob.pt") for cp in self["crop_path"])
            )
            return self["keypoints"]

        raise KeyError("There are no key-points in this object.")

    @keypoints.setter
    def keypoints(self, value: t.Tensor) -> None:
        try:
            J = self.J
            j_dim = self.joint_dim
        except NotImplementedError:
            J = None
            j_dim = None
        self.data["keypoints"] = (
            validate_key_points(key_points=value, length=self.B, joint_dim=j_dim, nof_joints=J)
            if self.validate
            else value
        ).to(device=self.device)

    @property
    def keypoints_local(self) -> t.Tensor:
        """Get the local key-points.
        The local coordinates are based on the coordinate-frame of the image crops, within the bounding-box.

        Optionally loads the local key-points from the 'keypoints_local_path' if given.
        Otherwise, tries to load the local key-points from the 'crop_path' with '.pt' ending if given.
        If one of the loading methods is used, the `joint_weight` will be set.
        """
        if "keypoints_local" in self:
            return self.data["keypoints_local"]

        if "keypoints_local_path" in self:
            if isinstance(self["keypoints_local_path"], str):
                self["keypoints_local_path"] = tuple(self["keypoints_local_path"] for _ in range(self.B))
            if not isinstance(self["keypoints_local_path"], tuple):
                raise NotImplementedError("Unknown format of keypoints_local_path.")

            self.keypoints_local = self.keypoints_and_weights_from_paths(self["keypoints_local_path"])
            return self.data["keypoints_local"]

        if "crop_path" in self:
            if isinstance(self["crop_path"], str):
                self["crop_path"] = tuple(self["crop_path"] for _ in range(self.B))
            if not isinstance(self["crop_path"], tuple):
                raise NotImplementedError("Unknown crop_path format.")

            self.keypoints_local = self.keypoints_and_weights_from_paths(
                tuple(replace_file_type(cp, new_type=".pt") for cp in self["crop_path"])
            )
            return self.data["keypoints_local"]

        raise KeyError("There are no local key-points in this object.")

    @keypoints_local.setter
    def keypoints_local(self, value: t.Tensor) -> None:
        """Set local key points with a little bit of validation."""
        try:
            J = self.J
            j_dim = self.joint_dim
        except NotImplementedError:
            J = None
            j_dim = None
        # use validate_key_points to make sure local key points have the correct shape [1 x J x 2|3]
        self.data["keypoints_local"] = (
            validate_key_points(key_points=value, length=self.B, nof_joints=J, joint_dim=j_dim)
            if self.validate
            else value
        ).to(device=self.device)

    @property
    def image(self) -> Images:
        """Get the original image(s) of this State.
        If the images are not available, try to load them using :func:`load_image` and :attr:`filepath`.
        """
        if "image" not in self.data:
            return self.load_image()
        return self.data["image"]

    @image.setter
    def image(self, value: Images) -> None:
        imgs = validate_images(value) if self.validate else value
        self.data["image"]: Images = [tvte.Image(v.to(device=self.device)) for v in imgs]

    @property
    def image_crop(self) -> Image:
        """Get the image crop(s) of this State.
        If the crops are not available, try to load them using :func:`load_image_crop` and :attr:`crop_path`.
        """
        if "image_crop" not in self.data:
            return self.load_image_crop()
        return self.data["image_crop"]

    @image_crop.setter
    def image_crop(self, value: Image) -> None:
        self.data["image_crop"]: Image = (validate_image(value) if self.validate else value).to(device=self.device)

    @property
    def crop_path(self):
        """Get the path to the image crops. Only necessary if the image crops are saved and not computed live."""
        return self.data["crop_path"]

    @crop_path.setter
    def crop_path(self, value: FilePaths):
        self.data["crop_path"]: FilePaths = validate_filepath(value) if self.validate else value

    @property
    def joint_weight(self) -> t.Tensor:
        """Get the weight of the joints. Either represents the visibility or an importance score of this joint."""
        return self.data["joint_weight"]

    @joint_weight.setter
    def joint_weight(self, value: t.Tensor) -> None:
        try:
            J = self.J
        except NotImplementedError:
            J = -1
        self.data["joint_weight"] = (value.reshape(self.B, J, 1) if self.validate else value).to(device=self.device)

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
        # pylint: disable=too-many-branches
        if idx >= self.B or idx < -self.B:
            raise IndexError(f"Expected index to lie within ({-self.B}, {self.B - 1}), but got: {idx}")

        new_data = {"validate": self.validate}
        for k, v in self.data.items():
            ks = str(k)
            if isinstance(v, tvte.TVTensor):
                # make sure tv_tensors stay, especially for bboxes
                new_data[ks] = tvte.wrap(v[idx], like=v)
            elif isinstance(v, list):
                # lists stay list
                new_data[ks] = [v[idx]]
            elif isinstance(v, tuple):
                # tuples stay tuple
                new_data[ks] = (v[idx],)
            elif isinstance(v, t.Tensor) and v.ndim > 1:
                new_data[ks] = v[idx].unsqueeze(0)
            elif isinstance(v, t.Tensor) and v.ndim == 1:
                new_data[ks] = v[idx].flatten()
            elif isinstance(v, t.Tensor) and v.ndim == 0:
                new_data[ks] = v
            elif isinstance(v, (dict, set)):  # dict and set stay the same
                new_data[ks] = v
            elif isinstance(v, str):
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
        # pylint: disable=too-many-branches
        if self.B == 1:
            return [self]
        new_data = [{"validate": self.validate} for _ in range(self.B)]
        for k, v in self.data.items():
            ks = str(k)
            for idx in range(self.B):
                if isinstance(v, list):
                    # lists stay list
                    new_data[idx][ks] = [v[idx]]
                elif isinstance(v, tuple):
                    # tuples stay tuple
                    new_data[idx][ks] = (v[idx],)
                elif isinstance(v, t.Tensor) and v.ndim > 1:
                    new_data[idx][ks] = v[idx].unsqueeze(0)
                elif isinstance(v, t.Tensor) and v.ndim == 1:
                    new_data[idx][ks] = v[idx].flatten()
                elif isinstance(v, t.Tensor) and v.ndim == 0:
                    new_data[idx][ks] = v
                elif isinstance(v, (dict, set)):  # dict and set stay the same
                    new_data[idx][ks] = v
                elif isinstance(v, str):
                    new_data[idx][ks] = v
                elif hasattr(v, "__getitem__"):
                    # every other iterable data -> regular tensors, ...
                    new_data[idx][ks] = v[idx]
                else:
                    new_data[idx][ks] = v

                # if it was a tv_tensor, make sure to wrap it again
                if isinstance(v, tvte.TVTensor):
                    # make sure tv_tensors stay, especially for bboxes
                    new_data[idx][ks] = tvte.wrap(new_data[idx][ks], like=v)

        assert all("bbox" in d for d in new_data), "No Bounding box given while creating the state."
        return [State(**d) for d in new_data]  # pylint: disable=missing-kwoa

    def load_image_crop(self, store: bool = False, **kwargs) -> Image:
        """Load the images crops using the crop_paths of this object. Does nothing if the crops are already present.

        Keyword Args:
            crop_size: The size of the image crops.
                Default ``DEF_VAL.images.crop_size``.
        """
        if (
            "image_crop" in self
            and self.data["image_crop"] is not None
            and len(self.data["image_crop"]) == self.B
            and "keypoints_local" in self
        ):
            return self.image_crop

        if self.B == 0:
            crop = t.empty((0, 3, 0, 0), device=self.device, dtype=t.long)
            if store:
                self.data["image_crop"] = crop
            return crop

        if "crop_path" in self:
            if len(self.crop_path) == 0:
                crop = []
                loc_kps = t.empty((0, 1, 2), dtype=t.long, device=self.device)
            else:
                # allow changing the crop_size and other params via kwargs
                crop = load_image(filepath=self.crop_path, device=self.device, **kwargs)

                kps_paths = tuple(replace_file_type(sub_path, new_type=".pt") for sub_path in self.crop_path)
                if all(is_file(path) for path in kps_paths):
                    loc_kps = self.keypoints_and_weights_from_paths(kps_paths, save_weights=store)
                else:
                    loc_kps = None
            if store:
                self.data["image_crop"] = crop
                if loc_kps is not None:
                    self.data["keypoints_local"] = loc_kps
            return crop

        try:
            kps = self.keypoints if "keypoints" in self.data else None
            crop, loc_kps = extract_crops_from_images(imgs=self.image, bboxes=self.bbox, kps=kps, **kwargs)
            if store:
                self.image_crop = crop
                if kps is not None:
                    self.keypoints_local = loc_kps
            return crop
        except AttributeError as e:
            raise AttributeError(
                "Could not load image crops without either a proper filepath given or an image and bbox given."
            ) from e

    def load_image(self, store: bool = False) -> Images:
        """Load the images using the filepaths of this object. Does nothing if the images are already present."""
        if "image" in self.data and self.data["image"] is not None:
            return self.image
        if "filepath" not in self.data:
            raise AttributeError("Could not load images without proper filepaths given.")
        if len(self.filepath) == 0:
            imgs: Images = []
        else:
            imgs: Images = load_image_list(filepath=self.filepath, device=self.device)
        if store:
            self.image = imgs
        return imgs

    def to(self, *args, **kwargs) -> "State":
        """Override torch.Tensor.to() for the whole object."""

        for attr_key, attr_value in self.items():
            if isinstance(attr_value, t.Tensor) or (hasattr(attr_value, "to") and callable(getattr(attr_value, "to"))):
                self[attr_key] = attr_value.to(*args, **kwargs)
        self.device = self.bbox.device
        return self

    def keypoints_and_weights_from_paths(self, paths: FilePaths, save_weights: bool = True) -> t.Tensor:
        """Given a tuple of paths, load the (local) key-points and weights from these paths.
        Does change ``self.joint_weight``,
        but does not change ``self.keypoints`` or ``self.keypoints_local`` respectively.

        Args:
            paths: A tuple of paths to the .pt files containing the key-points and weights.
            save_weights: Whether to save the weights if they were provided.

        Returns:
            The (local) key-points as :class:`~torch.Tensor`.

        Raises:
            ValueError: If the number of paths does not match the batch size.
            FileExistsError: If one of the paths does not exist.
        """
        if len(paths) != self.B:
            raise ValueError(f"There must be a path for every bounding box. Got B: {self.B} and paths: {paths}")

        kps, weights = [], []
        try:
            J = self.J
            j_dim = self.joint_dim
        except NotImplementedError:
            J, j_dim = None, None

        for path in paths:
            if not is_file(path):
                raise FileExistsError(f"Keypoint file: '{path}' is missing.")

            kp_data = t.load(os.path.normpath(path)).to(device=self.device)
            if J is not None and j_dim is not None and kp_data.size(-1) != 2:
                kp, jw = kp_data.reshape((1, J, j_dim + 1)).split([2, 1], dim=-1)
            elif j_dim == kp_data.size(-1) or kp_data.size(-1) == 2:
                kp = kp_data
                jw = None
            else:
                kp, jw = kp_data.split([2, 1], dim=-1)
            kps.append(kp)
            weights.append(jw)

        keypoints = t.cat(kps, dim=0).to(self.device)
        # save weights of all are not None
        if all(w is not None for w in weights):
            weights = t.cat(weights, dim=0).to(self.device)
            if "joint_weight" in self and not t.allclose(weights, self.joint_weight):
                raise ValueError(f"Expected old and new weights to be close, got: {self.joint_weight} and {weights}")
            if save_weights:
                self.joint_weight = weights

        return keypoints

    def cast_joint_weight(
        self,
        dtype: t.dtype = t.float32,
        decimals: int = 0,
        overwrite: bool = False,
    ) -> t.Tensor:
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
            if not t.is_floating_point(new_weights):
                new_weights = new_weights.to(dtype=t.float32)
            new_weights.round_(decimals=decimals)
        # change dtype
        new_weights = new_weights.to(dtype=dtype)
        # overwrite existing value if requested
        if overwrite:
            self.joint_weight = new_weights
        return new_weights

    @t.no_grad()
    def draw(
        self, save_path: FilePath, show_kp: bool = True, show_skeleton: bool = True, show_bbox: bool = True, **kwargs
    ) -> None:  # pragma: no cover
        """Draw the bboxes and key points of this State on the first image.

        This method uses torchvision to draw the information of this State on the first image in :attr:`self.image`.
        The drawing of key points, the respective connectivity / skeleton, and the bounding boxes can be disabled.
        Additionally, many keyword arguments can be set,
        see the docstring for :func:`.show_image_with_additional` for more information.

        Notes:
            In the case that :attr:`B` is ``0``,
            this method can still draw an empty image if an image or filepath is set.
            This works iff :attr:`validation` is ``False``.
            The :class:`PoseTrack21_Image` dataset uses this trick to draw the images that aren't annotated.
        """
        # make sure the full image is loaded
        img: Images = self.load_image(store=False)

        if len(img) == 0:
            warnings.warn(f"There are no images to be drawn for save_path: '{save_path}'")

        # get the original image - with B > 0, there might be multiple images; they all should be equal
        if len(img) > 1 and not all(t.allclose(self.image[i - 1], self.image[i]) for i in range(1, len(self.image))):
            raise ValueError(
                "If there are more than one image in this state, it is expected, that all the images are equal."
            )
        orig_img = img[0].detach().clone()

        save_dir = os.path.dirname(os.path.abspath(save_path))
        mkdir_if_missing(save_dir)

        img_kwargs = {
            "img": orig_img,
            "show": kwargs.pop("show", False),  # whether to show the image, the image will be saved nevertheless
        }
        if show_bbox:
            img_kwargs["bboxes"] = self.bbox
        if show_kp and "keypoints" in self.data:
            img_kwargs["key_points"] = self.keypoints
        if show_kp and "joint_weight" in self.data:
            img_kwargs["kp_visibility"] = self.joint_weight
        if show_skeleton and "skeleton_name" in self.data:
            img_kwargs["kp_connectivity"] = SKELETONS[
                (
                    self.data["skeleton_name"]
                    if isinstance(self.data["skeleton_name"], str)
                    else self.data["skeleton_name"][0]
                )
            ]
        if "pred_tid" in self.data:
            img_kwargs["bbox_labels"] = [str(tid) for tid in self["pred_tid"].tolist()]
            # make sure to map the same PID to the same color all the time
            colors = [COLORS[int(i) % len(COLORS)] for i in self["pred_tid"].tolist()]
            img_kwargs["bbox_colors"] = kwargs.pop("bbox_colors", colors)
            img_kwargs["kp_colors"] = kwargs.pop("kp_colors", colors)
            # fixme kind of useless, move to sub function
            img_kwargs["bbox_font"] = kwargs.pop("bbox_font", "./data/freemono/FreeMono.ttf")
            img_kwargs["bbox_font_size"] = kwargs.pop("bbox_font_size", min(self.bbox.canvas_size) // 10)
            img_kwargs["bbox_width"] = kwargs.pop("bbox_width", min(self.bbox.canvas_size) // 100)

        # add kwargs
        img_kwargs.update(kwargs)
        # draw bboxes and key points
        int_img = show_image_with_additional(**img_kwargs)

        # save the resulting image
        # ('save_image' expects a float32 image and is immediately converting it back to byte...)
        save_image(convert_image_dtype(int_img), fp=save_path)

    @t.no_grad()
    def draw_individually(self, save_path: Union[FilePath, FilePaths], **kwargs) -> None:  # pragma: no cover
        """Split the state and draw the detections of the image(s) independently.

        Args:
            save_path: Directory path to save the images to.
        """
        # validate save_path and create folders if necessary
        if isinstance(save_path, str):
            mkdir_if_missing(os.path.abspath(save_path))
            save_path = tuple(os.path.join(os.path.abspath(save_path), f"{i}.jpg") for i in range(self.B))
        elif isinstance(save_path, tuple):
            if len(save_path) != self.B:
                raise ValueError(
                    f"When giving multiple paths, it is expected that exactly B={self.B} are given, "
                    f"but got {len(save_path)}."
                )
            for path in save_path:
                mkdir_if_missing(os.path.dirname(os.path.abspath(path)))
        else:
            raise ValueError(f"Expected either a single path or a tuple of paths, but got: {save_path}")

        states = self.split()

        show = kwargs.pop("show", False)

        for i, (state, path) in enumerate(zip(states, save_path)):
            state.draw(save_path=path, show=show, **kwargs)
        plt.show()

    def clean(self, keys: Union[list[str], str] = None) -> "State":
        """Given a state, remove one or more keys to free up memory.

        Args:
            keys: The name of the keys to remove.
                If a key is not present in self.data, the key is ignored.
                If keys is None, the default keys ``["image", "image_crop"]`` are removed.
                If keys is "all", all keys that contain tensors are removed except for the bounding box.
        """
        if keys is None:
            keys = ["image", "image_crop"]
        elif keys == "all":
            keys = [
                k
                for k, v in self.data.items()
                if k != "bbox"
                and (
                    isinstance(v, t.Tensor)
                    or (isinstance(v, (list, tuple)) and all(isinstance(sub_v, t.Tensor) for sub_v in v))
                )
            ]
        elif isinstance(keys, str):
            keys = [keys]
        if "bbox" in keys:
            raise ValueError("Cannot clean bounding box!")
        for key in keys:
            if key in self.data:
                del self.data[key]
            self.data.pop(key, None)
        return self


EMPTY_STATE = State(bbox=tvte.BoundingBoxes(t.empty((0, 4)), canvas_size=(0, 0), format="XYXY"), validate=False)


def get_ds_data_getter(attributes: list[str]) -> DataGetter:
    """Given a list of attribute names,
    return a function, that gets those attributes from a given :class:`State`.
    """

    def getter(ds: State) -> Union[t.Tensor, tuple[t.Tensor, ...]]:
        """The getter function."""
        return tuple(ds[str(attrib)] for attrib in attributes)

    return getter


def collate_devices(batch: list[t.device], *_args, **_kwargs) -> t.device:
    """Collate a batch of devices into a single device."""
    return batch[0]


def collate_tensors(batch: list[t.Tensor], *_args, **_kwargs) -> t.Tensor:
    """Collate a batch of tensors into a single one.

    Will use torch.cat() if the first dimension has a shape of one, otherwise torch.stack()
    """
    if len(batch) == 0 or all(b.shape and len(b) == 0 for b in batch):
        return t.empty(0)
    return t.cat([b if b.shape else b.flatten() for b in batch if (not b.shape) or (b.shape and len(b))])


def collate_tvt_tensors(
    batch: list[Union[tvte.Image, tvte.Mask, tvte.Video]], *_args, **_kwargs
) -> Union[tvte.TVTensor, tvte.Image, tvte.Mask, tvte.Video]:
    """Collate a batch of tv_tensors into a batched version of it."""
    if len(batch) == 0 or all(b.shape and len(b) == 0 for b in batch):
        return tvte.TVTensor([])
    return tvte.wrap(
        t.cat([b if b.shape else b.flatten() for b in batch if (not b.shape) or (b.shape and len(b))]),
        like=batch[0],
    )


def collate_bboxes(batch: list[tvte.BoundingBoxes], *_args, **_kwargs) -> tvte.BoundingBoxes:
    """Collate a batch of bounding boxes into a single one.
    It is expected that all bounding boxes have the same canvas size and format.
    """
    if len(batch) == 0 or all(b.shape and len(b) == 0 for b in batch):
        return tvte.BoundingBoxes(t.empty((0, 4)), canvas_size=(0, 0), format="XYXY")
    bb_format: tvte.BoundingBoxFormat = batch[0].format
    canvas_size = batch[0].canvas_size

    return tvte.BoundingBoxes(
        t.cat(batch),
        canvas_size=canvas_size,
        format=bb_format,
    )


CUSTOM_COLLATE_MAP: dict[Type, Callable] = {  # pragma: no cover
    str: lambda str_batch, *args, **kwargs: tuple(s for s in str_batch),
    list: lambda l_batch, *args, **kwargs: sum(l_batch, []),
    tuple: lambda t_batch, *args, **kwargs: sum(t_batch, ()),
    tvte.BoundingBoxes: collate_bboxes,
    (tvte.Image, tvte.Video, tvte.Mask): collate_tvt_tensors,
    t.device: collate_devices,
    t.Tensor: collate_tensors,  # override regular tensor collate to *not* add another dimension
}


def collate_states(batch: Union[list[State], State]) -> State:
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

    # remove all empty states and return early for length 0 or 1
    batch = [b for b in batch if b.B != 0]

    if len(batch) == 0:
        return EMPTY_STATE.copy()

    if len(batch) == 1:
        return batch[0]

    c_batch: dict[str, any] = collate(batch, collate_fn_map=CUSTOM_COLLATE_MAP)

    # skip validation, because either every State has been validated before or validation is not required.
    s = State(**c_batch, validate=False)
    # then set the validation to the correct value
    s.validate = batch[0].validate
    return s


def collate_lists(batch: Union[State, list[State], list[list[State]]]) -> list[State]:
    """Collate function used to not concatenate a batch of States.

    Given the batch data, return a list of states.
    If the batch data is a single state, a list with the single state is returned.
    If the batch data is a list of states, the list is returned.
    if the batch data is a list containing list of states,
    the sub-list of states is collated using the regular :func:`collate_states` function.
    Thus, a list of collated states is returned.

    Args:
        batch: Either a list of States, a single State, or a list containing list of States.

    Returns:
        A list of States. Every State can have a different number of items.
    """
    if isinstance(batch, State):
        return [batch]
    if isinstance(batch, list) and all(isinstance(b, State) for b in batch):
        return batch
    if isinstance(batch, list) and all(
        isinstance(b, list) and all(isinstance(sub_state, State) for sub_state in b) for b in batch
    ):
        return [collate_states(b) for b in batch]
    raise NotImplementedError


def collate_list_of_history(batch: Union[State, list[State], list[list[State]]]) -> list[State]:
    """Collate function used to combine the data returned in the format of a class:`ImageHistoryDataset`.

    With ``N`` detections, a batch of data contains ``N`` lists, each with ``L+1`` States.
    This functions collates the ``i``-th State of each of the ``N`` lists into a single list of States
    of length ``L+1``.

    Args:
        batch: Either a single list with ``L+1`` :class:`States`
            or a list containing ``N`` list, each containing ``L+1`` :class:`States`.
            A single State is also supported, even though, this shouldn't really be feasible.

    Returns:
        A list of States.
        Because there can be a different number of detections, every State can still have a different number of items.
    """
    if isinstance(batch, State):
        return [batch]
    if isinstance(batch, list) and all(isinstance(b, State) for b in batch):
        return batch
    # a list containing a single list of states -> no collating necessary
    if (
        isinstance(batch, list)
        and len(batch) == 1
        and isinstance(batch[0], list)
        and all(isinstance(i, State) for i in batch[0])
    ):
        return batch[0]
    # a list containing multiple list of states, all with the same length, containing states
    if isinstance(batch, list) and all(
        isinstance(b, list) and len(b) == len(batch[0]) and all(isinstance(sub_state, State) for sub_state in b)
        for b in batch
    ):
        return [collate_states([batch[i][l] for i in range(len(batch))]) for l in range(len(batch[0]))]

    raise NotImplementedError(f"Unknown format of batch - length: {len(batch)} type: {type(batch)}")
