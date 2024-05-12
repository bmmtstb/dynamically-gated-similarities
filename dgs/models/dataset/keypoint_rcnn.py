"""
Use :func:`.keypointrcnn_resnet50_fpn` to predict the key points and bounding boxes of each image.

References:
    https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html
"""

import os
from abc import ABC
from typing import Union

import torch
from torch import nn
from torchvision import tv_tensors as tvte
from torchvision.io import VideoReader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset, ImageDataset, VideoDataset
from dgs.utils.config import DEF_VAL
from dgs.utils.constants import IMAGE_FORMATS, VIDEO_FORMATS
from dgs.utils.files import is_dir, is_file
from dgs.utils.image import CustomToAspect, load_image
from dgs.utils.state import State
from dgs.utils.types import Config, FilePath, FilePaths, Image, Images, NodePath, Validations
from dgs.utils.utils import extract_crops_from_images

rcnn_validations: Validations = {
    "data_path": [("any", [str, ("all", [list, ("forall", str)])])],
    # optional
    "threshold": ["optional", float, ("within", (0.0, 1.0))],
    "crop_mode": ["optional", str, ("in", CustomToAspect.modes)],
    "crop_size": ["optional", tuple, ("len", 2), ("forall", (int, ("gt", 0)))],
}


class KeypointRCNNBackbone(BaseDataset, nn.Module, ABC):
    """Metaclass for the torchvision Key Point RCNN backbone model.

    This class sets up the RCNN model and validates and sets the basic modules parameters.

    Params
    ------

    threshold (float):
        Detections with a score lower than the threshold will be ignored.
        Default ``DEF_VAL.dataset.kprcnn.threshold``.

    """

    def __init__(self, config: Config, path: NodePath) -> None:
        BaseDataset.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(rcnn_validations)

        self.threshold: float = self.params.get("threshold", DEF_VAL.dataset.kprcnn.threshold)

        self.logger.debug("Loading Keypoint-RCNN Model")
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
        self.register_module("model", model)
        self.configure_torch_module(module=self.model, train=False)

        self.img_id: int = 1

    @torch.no_grad()
    def images_to_states(self, images: Images) -> list[State]:
        """Given a list of images, use the key-point-RCNN model to predict key points and bounding boxes,
        then create a :class:`State` containing the available information.

        Notes:
            Does not add the original image to the new State, to reduce memory / GPU usage.
        """

        outputs = self.model(images)

        states: list[State] = []
        canvas_size = (max(i.shape[-2] for i in images), max(i.shape[-1] for i in images))

        for output, image in zip(outputs, images):
            # for every image (output), get the indices where the score is bigger than the threshold
            indices = output["scores"] > self.threshold

            # bbox given in XYXY format
            bbox = tvte.BoundingBoxes(output["boxes"][indices], format="XYXY", canvas_size=canvas_size)
            # keypoints in [x,y,v] format -> kp, vis
            kps, vis = (
                output["keypoints"][indices]
                .to(device=self.device, dtype=self.precision)
                .reshape((-1, 17, 3))
                .split([2, 1], dim=-1)
            )

            crops, loc_kps = extract_crops_from_images(
                imgs=[tvte.Image(image.unsqueeze(0)) for _ in range(len(bbox))],
                bboxes=bbox,
                kps=kps,
                crop_size=self.params.get("crop_size", DEF_VAL.images.crop_size),
                crop_mode=self.params.get("crop_mode", DEF_VAL.images.crop_mode),
            )

            if crops.ndim == 3:
                crops = tvte.wrap(crops.unsqueeze(0), like=crops)

            B = len(bbox)

            data = {
                "validate": False,
                "bbox": bbox,
                "image_crop": crops,
                "keypoints": kps,
                "keypoints_local": loc_kps,
                "joint_weight": vis,
                "scores": output["scores"],
                "skeleton_name": tuple("coco" for _ in range(B)),
                "image_id": torch.ones(B, device=self.device, dtype=torch.long) * self.img_id,
                "frame_id": torch.ones(B, device=self.device, dtype=torch.long) * self.img_id,
                "person_id": torch.ones(B, device=self.device, dtype=torch.long) * -1,
            }
            self.img_id += 1
            states.append(State(**data))

        return states

    def terminate(self) -> None:
        if hasattr(self, "model"):
            self.model = None


# pylint: disable=too-many-ancestors
class KeypointRCNNImageBackbone(KeypointRCNNBackbone, ImageDataset):
    """

    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html

    Params
    ------

    threshold (float):
        Detections with a score lower than the threshold will be ignored.
        Default ``DEF_VAL.dataset.kprcnn.threshold``.

    Optional Params
    ---------------

    crop_size (:obj:`ImgSize`):
        The size, the image crop should have.
        Default ``DEF_VAL.images.crop_size``.

    crop_mode (str):
        The mode to use when cropping the image.
        Default ``DEF_VAL.images.crop_mode``.

    """

    data: list[FilePath]

    def __init__(self, config: Config, path: NodePath) -> None:
        KeypointRCNNBackbone.__init__(self, config=config, path=path)
        ImageDataset.__init__(self, config=config, path=path)

        # load data - data_path is either a directory, a single image file, or a list of image filepaths
        self.data = []
        data_path = self.params["data_path"]
        if isinstance(data_path, list):
            assert all(isinstance(p, str) for p in data_path), "Path is a list but not all values are string"
            assert all(
                any(p.lower().endswith(end) for end in IMAGE_FORMATS) for p in data_path
            ), "Not all values are images"
            self.data = sorted(data_path)
        elif isinstance(data_path, str):
            data_path = self.get_path_in_dataset(data_path)
            if is_file(data_path):
                # single image
                if any(data_path.lower().endswith(ending) for ending in IMAGE_FORMATS):
                    self.data = [data_path]
                # video file
                elif any(data_path.lower().endswith(ending) for ending in VIDEO_FORMATS):
                    raise TypeError(f"Got Video file, but is an Image Dataset. File: {data_path}")
                else:
                    raise NotImplementedError(f"Unknown file type. Got '{data_path}'")
            elif is_dir(data_path):
                # directory of images
                self.data = [
                    os.path.normpath(os.path.join(data_path, child_path))
                    for child_path in tqdm(sorted(os.listdir(data_path)), desc="Loading images")
                    if any(child_path.lower().endswith(ending) for ending in IMAGE_FORMATS)
                ]
            else:
                raise NotImplementedError(f"string is neither file nor dir. Got '{data_path}'.")
        else:
            raise NotImplementedError(
                f"Unknown path object, expected filepath, dirpath, or list of filepaths. Got {type(data_path)}"
            )

    def arbitrary_to_ds(self, a: Union[FilePath, FilePaths], idx: int) -> list[State]:
        """Given a filepath, predict the bounding boxes and key-points of the respective image.
        Return a State containing all the available information.
        Because the state is known, the image is not saved in the State, to reduce the space-overhead on the GPU.
        """
        if isinstance(a, str):
            a = (a,)
        # the torch model expects a list of 3D images
        images = [
            convert_image_dtype(tvte.Image(load_image(fp).squeeze(0), device=self.device), dtype=torch.float32)
            for fp in a
        ]

        states = self.images_to_states(images=images)

        for fp, state in zip(a, states):
            state.filepath = tuple(fp for _ in range(state.B))

        return states


# pylint: disable=too-many-ancestors
class KeypointRCNNVideoBackbone(KeypointRCNNBackbone, VideoDataset):
    """A Dataset that gets the path to a single Video file and predicts the bounding boxes and key points of the Video.

    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html

    Params
    ------

    threshold (float):
        Detections with a score lower than the threshold will be ignored.
        Default ``DEF_VAL.dataset.kprcnn.threshold``.

    Optional Params
    ---------------

    crop_size (:obj:`.ImgSize`):
        The size, the image crop should have.
        Default ``DEF_VAL.images.crop_size``.

    crop_mode (str):
        The mode to use when cropping the image.
        Default ``DEF_VAL.images.crop_mode``.
    """

    data: VideoReader

    def __init__(self, config: Config, path: NodePath) -> None:
        KeypointRCNNBackbone.__init__(self, config=config, path=path)
        VideoDataset.__init__(self, config=config, path=path)
        # the data has already been loaded in the VideoDataset
        # the model and threshold has been loaded in KeypointRCNNBackbone

    def arbitrary_to_ds(self, a: Image, idx: int) -> list[State]:
        """Given a frame of a video, return the resulting state after running the RCNN model."""
        if not isinstance(a, torch.Tensor):
            raise NotImplementedError
        if a.ndim == 3:
            a = a.unsqueeze(0)
        # the torch RCNN model expects a list of 3D images
        images = [convert_image_dtype(img, torch.float32) for img in a]

        states = self.images_to_states(images=images)

        for img, state in zip(a, states):
            state.image = [img.unsqueeze(0) for _ in range(state.B)]

        return states
