"""Methods and helpers for handling bounding boxes.

Implements a simple bounding box class to handle bbox objects within dgs.

Additionally, this implements all conversion functions to convert between the different bbox formats.
See class description or BBOX_FORMATS constant in dgs.utils.constants for a list of all possible bbox formats.
An explanation can be found in the BoundingBox class.
"""
import torch

from dgs.utils.constants import BBOX_FORMATS
from dgs.utils.exceptions import BoundingBoxException


class BoundingBox:
    r"""Class for storing and modifying bounding boxes.

    The values of a created bounding box are static and therefore cannot be changed.

    BBox-Formats
    ------------

    Initially, only one format is given.
    If you intend to stay in the same format the whole time, this removes redundant computation.
    Other formats and additional values will only be computed on request,
    or if the transformation to another format needs specific values.
    If intermediate values are computed once, they are stored inside the class and can therefore be reused easily.

    Looking at a regular image, the width of the bounding box lies on the x-axis and the height lies on the y-axis.
    There exist a multitude of formats:

    * xyxy: [x\ :sub:`min`, y\ :sub:`min`, x\ :sub:`max`, y\ :sub:`max`]

      Pixel values of left [0], top [1], right [2], and bottom [3] corner.
      Sometimes called ltrb, referencing the left top right and bottom sides.

    * xywh: [x\ :sub:`min`, y\ :sub:`min`, width, height]

      Used in COCO

    * xyah: [x\ :sub:`min`, y\ :sub:`min`, aspect ratio, height]

      Instead of providing the width, use the aspec ratio `( w / h )`.
      Because the aspect ratio is not as much influenced by pixel noise as the width,
      this has some advantages during training.

    * yolo: [x\ :sub:`center`, y\ :sub:`center`, width, height]

      ?

    """

    # there a many attributes and they can get used, so please the linter
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(self, **kwargs) -> None:
        """Initialize the bounding-box object.

        Iff you have multiple bounding box formats precomputed, you can pass multiple values to the initializer.
        If multiple values are given, they will be validated against each other.
        They are expected to reference the same bounding box!

        Most cases will only provide one single format and will convert it to another.

        Keyword Args:
            xyxy: Torch tensor containing bbox information in xyxy format
            xyah: Torch tensor containing bbox information in xyah format
            xywh: Torch tensor containing bbox information in xywh format
            yolo: Torch tensor containing bbox information in yolo format

        Raises:
             BoundingBoxException if there is anything wrong with the provided values.
        """
        # validate kwargs
        if len(kwargs) == 0:
            raise BoundingBoxException("Bounding box format has to be specified. Make sure to use kwargs not args...")

        if any(k not in BBOX_FORMATS for k in kwargs):
            raise BoundingBoxException(
                f"Unknown bbox format during initialization. Specified formats: {[kwargs.keys()]}"
            )

        self._xyxy = None
        self._xyah = None
        self._xywh = None
        self._yolo = None
        #
        self._device = None
        #
        self._left = None
        self._right = None
        self._top = None
        self._bottom = None
        self._width = None
        self._height = None
        self._center = None
        self._aspect = None

        self._validate_inputs(**kwargs)

    def _validate_inputs(self, **kwargs) -> None:
        """Validates and sets the input kwargs as class attributes.

        Validations:
          - existence of at least one value that is not None,
          - length of four
          - compare value to against other values if multiple valid kwargs are provided
        """
        value_set = False
        for key, value in kwargs.items():
            if value is not None:
                if len(value) != 4:
                    raise BoundingBoxException("Bounding box input tensors have to have length of 4.")

                # this value seems correct, set it using custom setter function
                value_set = True
                setter_func = getattr(self, f"_set_{str(key)}")
                setter_func(value)

                # validate device
                if self._device and self._device != value.device:
                    raise BoundingBoxException("Multiple values were given, but they are not on the same device.")
                if not self._device:
                    self._device = value.device

                # if there are multiple kwargs given, validate against the others
                for other_key, other_value in kwargs.items():
                    # don't compare against self and skip if prev already validated
                    if other_key == key:
                        continue
                    if getattr(self, "_" + str(other_key), None) is not None:
                        continue
                    conversion_func = getattr(self, f"to_{other_key}")
                    if not torch.allclose(conversion_func(), other_value):
                        raise BoundingBoxException(
                            "Multiple values were given, but they do not have the same value."
                            f"Noticed while converting value from {key} to {other_key}."
                        )

        if not value_set:
            raise BoundingBoxException(
                f"At least one bbox has to have a value. "
                f"Format is expected to be in {BBOX_FORMATS} but was {kwargs.keys()}"
            )

    def corners(self, image_size: tuple[int, int]) -> torch.IntTensor:
        """Obtain the integer coordinates of the four bbox corners.

        Use the four side properties to compute the corners in xyxy / ltrb format.

        Args:
            image_size: (H, W) width and height of the original image, as upper bound for the corners.

        Returns:
            output shape: [4] - integer values

            Integer values of the four bbox corners in xyxy format.
        """
        corners = torch.round(torch.FloatTensor([self.left, self.top, self.right, self.bottom]))
        H, W = image_size
        # create upper image shape
        # make sure to use max - 1 to be in range
        img_shape: torch.IntTensor = torch.IntTensor([W, H, W, H], device=self._device) - 1
        return torch.clamp(corners, torch.zeros_like(img_shape), img_shape).to(dtype=torch.int)

    @property
    def xyxy(self) -> torch.FloatTensor:
        """Get bbox coordinates in xyxy format"""
        if self._xyxy is None:
            return self.to_xyxy()
        return self._xyxy

    def _set_xyxy(self, new_value: torch.FloatTensor) -> None:
        """Set bbox coordinates in xyxy format.

        Should not be called outside of init because it does not validate against the other properties.
        """
        self._xyxy = new_value
        self._left = new_value[0]
        self._top = new_value[1]
        self._right = new_value[2]
        self._bottom = new_value[3]

        if self._left >= self._right:
            raise BoundingBoxException("Cannot create bounding box with left greater than than right.")
        if self._top >= self._bottom:
            raise BoundingBoxException("Cannot create bounding box with top greater than than bottom.")

    def to_xyxy(self) -> torch.FloatTensor:
        """Convert bbox format to xyxy"""
        xyxy: torch.FloatTensor = torch.zeros(4).to(dtype=torch.float32, device=self._device)
        xyxy[0] = self.left
        xyxy[1] = self.top
        xyxy[2] = self.right
        xyxy[3] = self.bottom
        # store and return result
        self._xyxy = xyxy
        return self._xyxy

    @property
    def xywh(self) -> torch.FloatTensor:
        """Get bbox coordinates in xywh format"""
        if self._xywh is None:
            return self.to_xywh()
        return self._xywh

    def _set_xywh(self, new_value: torch.FloatTensor) -> None:
        """Set bbox coordinates in xywh format.

        Should not be called outside of init because it does not validate against the other properties.
        """
        self._xywh = new_value
        self._left = new_value[0]
        self._top = new_value[1]
        self._width = new_value[2]
        self._height = new_value[3]
        if self._width <= 0:
            raise BoundingBoxException("Cannot create bounding box with width of 0")
        if self._height <= 0:
            raise BoundingBoxException("Cannot create bounding box with height of 0")

    def to_xywh(self) -> torch.FloatTensor:
        """Convert bbox format to xywh"""
        xywh: torch.FloatTensor = torch.zeros(4).to(dtype=torch.float32, device=self._device)
        xywh[0] = self.left
        xywh[1] = self.top
        xywh[2] = self.width
        xywh[3] = self.height
        # store and return result
        self._xywh = xywh
        return self._xywh

    @property
    def xyah(self) -> torch.FloatTensor:
        """Get bbox coordinates in xyah format"""
        if self._xyah is None:
            return self.to_xyah()
        return self._xyah

    def _set_xyah(self, new_value: torch.FloatTensor) -> None:
        """Set bbox coordinates in xyah format.

        Should not be called outside of init because it does not validate against the other properties.
        """
        self._xyah = new_value
        self._left = new_value[0]
        self._top = new_value[1]
        self._aspect = new_value[2]
        self._height = new_value[3]
        if self._aspect <= 0:
            raise BoundingBoxException("Cannot create bounding box with aspect ratio smaller than 0")
        if self._height <= 0:
            raise BoundingBoxException("Cannot create bounding box with height of 0")

    def to_xyah(self) -> torch.FloatTensor:
        """Convert bbox format to xyah"""
        xyah: torch.FloatTensor = torch.zeros(4).to(dtype=torch.float32, device=self._device)
        xyah[0] = self.left
        xyah[1] = self.top
        xyah[2] = self.aspect_ratio
        xyah[3] = self.height
        # store and return result
        self._xyah = xyah
        return self._xyah

    @property
    def yolo(self) -> torch.FloatTensor:
        """Get bbox coordinates in yolo format"""
        if self._yolo is None:
            return self.to_yolo()
        return self._yolo

    def _set_yolo(self, new_value: torch.FloatTensor) -> None:
        """Set bbox coordinates in yolo format.

        Should not be called outside of init because it does not validate against the other properties.
        """
        self._yolo = new_value
        self._center = torch.FloatTensor([new_value[0], new_value[1]], device=new_value.device)
        self._width = new_value[2]
        self._height = new_value[3]
        if self._width <= 0:
            raise BoundingBoxException("Cannot create bounding box with width of 0")
        if self._height <= 0:
            raise BoundingBoxException("Cannot create bounding box with height of 0")

    def to_yolo(self) -> torch.FloatTensor:
        """Convert bbox format to yolo"""
        yolo: torch.FloatTensor = torch.zeros(4).to(dtype=torch.float32, device=self._device)
        yolo[0] = self.center_x
        yolo[1] = self.center_y
        yolo[2] = self.width
        yolo[3] = self.height
        # store and return result
        self._yolo = yolo
        return self._yolo

    @property
    def left(self) -> torch.FloatTensor:
        """Return position of the left side.

        The left side is the beginning of the x-axis.
        """
        if isinstance(self._left, torch.FloatTensor):
            return self._left
        # xyxy, xyah, and xywh set this during init
        if isinstance(self._yolo, torch.FloatTensor):
            return self._yolo[0] - self.width / 2
        raise NotImplementedError

    @property
    def right(self) -> torch.FloatTensor:
        """Return position of the right side.

        The right side is the end of the x-axis.
        """
        if isinstance(self._right, torch.FloatTensor):
            return self._right
        # xyxy sets this during init
        if isinstance(self._xyah, torch.FloatTensor):
            return self._xyah[0] + self._xyah[2] * self._xyah[3]
        if isinstance(self._xywh, torch.FloatTensor):
            return self._xywh[0] + self._xywh[2]
        if isinstance(self._yolo, torch.FloatTensor):
            return self._yolo[0] + self.width / 2
        raise NotImplementedError

    @property
    def top(self) -> torch.FloatTensor:
        """Return position of the top side.

        The bottom is the beginning of the y-axis.
        """
        if isinstance(self._top, torch.FloatTensor):
            return self._top
        # xyxy, xyah, and xywh set this during init
        if isinstance(self._yolo, torch.FloatTensor):
            return self._yolo[1] - self.height / 2
        raise NotImplementedError

    @property
    def bottom(self) -> torch.FloatTensor:
        """Return position of the bottom side.

        The bottom is the end of the y-axis.
        """
        if isinstance(self._bottom, torch.FloatTensor):
            return self._bottom
        # xyxy sets this during init
        if isinstance(self._xyah, torch.FloatTensor):
            return self._xyah[1] + self.height
        if isinstance(self._xywh, torch.FloatTensor):
            return self._xywh[1] + self.height
        if isinstance(self._yolo, torch.FloatTensor):
            return self._yolo[1] + self.height / 2
        raise NotImplementedError

    @property
    def width(self) -> torch.FloatTensor:
        """Retrieve the width of the bbox.

        Width is parallel to the x-axis.
        """
        if isinstance(self._width, torch.FloatTensor):
            return self._width
        # yolo and xywh set this during init
        if isinstance(self._xyah, torch.FloatTensor):
            return self.aspect_ratio * self.height  # width = aspect * height
        if isinstance(self._xyxy, torch.FloatTensor):
            return self._xyxy[2] - self._xyxy[0]
        raise NotImplementedError

    @property
    def height(self) -> torch.FloatTensor:
        """Retrieve the height of the bbox.

        Height is parallel to the y-axis.
        """
        if isinstance(self._height, torch.FloatTensor):
            return self._height
        # xywh, xyah, and yolo set this during init
        if isinstance(self._xyxy, torch.FloatTensor):
            return self._xyxy[3] - self._xyxy[1]
        raise NotImplementedError

    @property
    def center_x(self) -> torch.FloatTensor:
        """Retrieve the x coordinate of the center."""
        if isinstance(self._center, torch.FloatTensor):
            return self._center[0]
        # yolo sets this during init
        if isinstance(self._xyah, torch.FloatTensor):
            return self._xyah[0] + self.width / 2
        if isinstance(self._xywh, torch.FloatTensor):
            return self._xywh[0] + self.width / 2
        if isinstance(self._xyxy, torch.FloatTensor):
            return (self._xyxy[0] + self._xyxy[2]) / 2
        raise NotImplementedError

    @property
    def center_y(self) -> torch.FloatTensor:
        """Retrieve the y coordinate of the center."""
        if isinstance(self._center, torch.FloatTensor):
            return self._center[1]
        # yolo sets this during init
        if isinstance(self._xyah, torch.FloatTensor):
            return self._xyah[1] + self.height / 2
        if isinstance(self._xywh, torch.FloatTensor):
            return self._xywh[1] + self.height / 2
        if isinstance(self._xyxy, torch.FloatTensor):
            return (self._xyxy[1] + self._xyxy[3]) / 2
        raise NotImplementedError

    @property
    def center(self) -> torch.FloatTensor:
        """Retrieve the center of the bounding box."""
        if isinstance(self._center, torch.FloatTensor):
            return self._center
        # yolo sets this during init
        return torch.FloatTensor([self.center_x, self.center_y], device=self._device)

    @property
    def aspect_ratio(self) -> torch.FloatTensor:
        """Retrieve the aspect ratio of the bounding box.

        The aspect ratio is always computed as width / height.
        """
        if isinstance(self._aspect, torch.FloatTensor):
            return self._aspect
        # xyah sets this during init
        if isinstance(self._yolo, torch.FloatTensor):
            return self._yolo[2] / self._yolo[3]
        if isinstance(self._xywh, torch.FloatTensor):
            return self._xywh[2] / self._xywh[3]
        if isinstance(self._xyxy, torch.FloatTensor):
            return (self._xyxy[0] + self._xyxy[2]) / (self._xyxy[1] + self._xyxy[3])
        raise NotImplementedError

    def contains(self, point: torch.Tensor) -> bool:
        """Return whether this bounding box contains a given point."""
        return bool(self.top <= point[1] <= self.bottom and self.left <= point[0] <= self.right)

    def iou(self, other: "BoundingBox") -> torch.FloatTensor:
        """Compute intersection over union between self and other."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a representation of the bounding box"""
        return (
            f"left: {self.left}, right: {self.right}, height: {self.height}, "
            f"top: {self.top}, bottom: {self.bottom}, width: {self.width}, "
            f"center: {self.center}"
        )

    def __eq__(self, other: "BoundingBox") -> bool:
        """Return whether two bboxes are equal."""
        if isinstance(other, BoundingBox):
            # make sure they are on the same device
            if self._device != other._device:
                return False
            return torch.allclose(self.xyxy, other.xyxy)
        raise NotImplementedError(
            f"Equality is only defined between two BoundingBox objects. Other has type: {type(other)}"
        )
