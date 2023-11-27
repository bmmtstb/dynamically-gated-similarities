"""
definitions and helpers for pose-state(s)
"""
import torch

from dgs.models.bbox import BoundingBox
from dgs.utils.types import Config, PoseStateTuple


class PoseState:
    """
    # PoseState = (Pose, JCS, BBox)
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
        """
        Equality between two PoseState objects

        Args:
            other: PoseState object or PoseStateTuple to compare to

        Returns:
            boolean, whether the two PoseState are equal
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
        """
        Override PoseState["item"] to be class-specific

        Args:
            item: name of value to retrieve, has to be one value from ["pose", "jcs", "bbox"]

        Returns:
            State with the given name as torch.Tensor
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
        """
        Obtain a copy of the three states within this queue.
        Due to multiprocessing, we technically have to freeze appending to ensure equal length and matching indices at
        all times, but as long as it doesn't make problems, this will be postponed.

        Args:
            items: slice
            copy: whether to create a detached and cloned copy of the current states or the real tensors

        Returns:
            Three stacked tensors of cloned and detached current state on the configured device
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
        """
        Overwrite get-item call (PoseStates[i]) to obtain pose state by indices.
        Supports python indexing using slices.
        Returns the exact torch tensor because it is not possible to add further parameters to this call.
        Therefore, if you want to obtain a detached and cloned tensor use self.get_state(..., copy=True)

        Args:
            item: index or slice of the states to obtain

        Returns:
            Single pose state given integer item
            List of pose states given slice
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
        Overwrite += to use append()

        Args:
            other: tuple of pose state to append to self

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


class BackboneOutput:
    """Class for storing backbone outputs.

    Original image shape: `[B x C x H x W]`
    Cropped image shape: `[C x h x w]`
    """

    # there a many attributes and they can get used, so please the linter
    # pylint: disable=too-many-instance-attributes

    def __init__(self, **kwargs) -> None:
        self.img_orig: torch.Tensor = torch.squeeze(kwargs.get("img_orig"))
        self.img_name: str = kwargs.get("img_name")
        self._img_crop: torch.Tensor = kwargs.get("img_crop", None)
        self.heatmaps: torch.Tensor = kwargs.get("hm", None)
        self.ids = kwargs.get("ids", None)  # fixme type?
        self.jcs: torch.Tensor = kwargs.get("jcs", None)
        self.visibility: torch.Tensor = kwargs.get("vis", None)
        self.bbox: BoundingBox = kwargs.get("bbox", None)

    def to(self, *args, **kwargs) -> "BackboneOutput":
        """Override torch tensor to for the whole object."""
        for name in ["img_orig", "_img_crop", "heatmaps", "jcs", "bbox"]:
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self

    def __str__(self) -> str:
        """Overwrite representation to be image name."""
        return self.img_name

    def contains_img_name(self, name: str) -> bool:
        """Returns whether the image has given name"""
        return self.img_name == name

    def contains_id(self, id_) -> bool:
        """Returns whether the id is part of this object's ids"""
        return id_ in self.ids

    @property
    def J(self) -> int:
        """Get number of joints"""
        return self.heatmaps.shape[0]

    @property
    def C(self) -> int:
        """Get number of channels in image"""
        # might use 0 or 1, because image can have dimension for batch size (of one)
        return self.img_orig.shape[-3]

    @property
    def W(self) -> int:
        """Get the width of the original image"""
        return self.img_orig.shape[-1]

    @property
    def img_crop(self) -> torch.Tensor:
        """Get the plain image crop or compute it if it's not yet available.

        One might want to include image crop reshaping after this step
        """
        if self._img_crop is None:
            # compute image crop using global / original coordinates
            x1, y1, x2, y2 = self.bbox.corners((self.W, self.H))
            self.img_crop = self.img_orig[:, y1:y2, x1:x2]
            # fixme: do we have to reshape crop to specific size?
        return self._img_crop

    @img_crop.setter
    def img_crop(self, crop: torch.Tensor) -> None:
        """Set image crop."""
        self._img_crop = crop

    @property
    def w(self) -> int:
        """Get the width of the cropped image"""
        return self.img_crop.shape[-1]

    @property
    def H(self) -> int:
        """Get the height of the original image"""
        return self.img_orig.shape[-2]

    @property
    def h(self) -> int:
        """Get the height of the cropped image"""
        return self.img_crop.shape[-2]
