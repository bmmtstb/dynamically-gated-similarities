"""
Use 'keypointrcnn_resnet50_fpn' from PyTorch.

References:
    https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
"""

import os
from abc import ABC

import torch
from torch.nn import Module as TorchModule
from torchvision import tv_tensors as tvte
from torchvision.io import VideoReader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset, VideoDataset
from dgs.utils.config import DEF_CONF
from dgs.utils.constants import IMAGE_FORMATS, VIDEO_FORMATS
from dgs.utils.files import is_dir, is_file
from dgs.utils.image import CustomToAspect, load_image
from dgs.utils.state import collate_states, State
from dgs.utils.types import Config, FilePath, Image, Images, NodePath, Validations
from dgs.utils.utils import extract_crops_from_images

rcnn_validations: Validations = {
    "path": [("any", [str, [list, ("forall", str)]])],
    # optional
    "threshold": ["optional", float, ("within", (0.0, 1.0))],
    "crop_mode": ["optional", str, ("in", CustomToAspect.modes)],
    "crop_size": ["optional", tuple, ("len", 2), ("forall", (int, ("gt", 0)))],
}


class KeypointRCNNBackbone(BaseDataset, TorchModule, ABC):

    def __init__(self, config: Config, path: NodePath) -> None:
        BaseDataset.__init__(self, config, path)
        TorchModule.__init__(self)

        self.validate_params(rcnn_validations)

        self.threshold: float = self.params.get("threshold", DEF_CONF.backbone.kprcnn.threshold)

        self.logger.info("Loading Keypoint-RCNN Model")
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
        self.model.eval()
        self.model.to(self.device)

    def outputs_to_states(self, outputs: list[dict], images: Images) -> State:
        """"""
        states = []
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
            new_images = [tvte.Image(image.unsqueeze(0)) for _ in range(len(bbox))]

            crops, loc_kps = extract_crops_from_images(
                imgs=new_images,
                bboxes=bbox,
                kps=kps,
                crop_size=self.params.get("crop_size", DEF_CONF.images.crop_size),
                crop_mode=self.params.get("crop_mode", DEF_CONF.images.crop_mode),
            )

            if crops.ndim == 3:
                crops = tvte.wrap(crops.unsqueeze(0), like=crops)

            data = {
                "validate": True,  # fixme remove
                # "validate": False,
                "bbox": bbox,
                "image_crop": crops,
                "keypoints": kps,
                "keypoints_local": loc_kps,
                "joint_weight": vis,
                "scores": output["scores"],
            }
            states.append(State(**data))

        return collate_states(states)


class KeypointRCNNImageBackbone(KeypointRCNNBackbone):
    """

    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn

    Params
    ------

    threshold (float):
        Detections with a score lower than the threshold will be ignored.
        Default `DEF_CONF.backbone.kprcnn.threshold`.

    Optional Params
    ---------------

    crop_size (:obj:`ImgSize`):
        The size, the image crop should have.
        Default `DEF_CONF.images.crop_size`

    crop_mode (str):
        The mode to use when cropping the image.
        Default `DEF_CONF.images.crop_mode`

    """

    data: list[FilePath]

    def __init__(self, config: Config, path: NodePath) -> None:
        KeypointRCNNBackbone.__init__(self, config=config, path=path)

        # load data - path is either a directory, a single image file, or a list of image filepaths
        self.data = []
        path = self.params["path"]
        if isinstance(path, list):
            self.data = path
        elif isinstance(path, str):
            path = self.get_path_in_dataset(path)
            if is_file(path):
                # single image
                if any(path.lower().endswith(ending) for ending in IMAGE_FORMATS):
                    self.data = [path]
                # video file
                elif any(path.lower().endswith(ending) for ending in VIDEO_FORMATS):
                    raise TypeError(f"Got Video file, but is an Image Dataset. File: {path}")
                else:
                    raise NotImplementedError(f"Unknown file type. Got '{path}'")
            elif is_dir(path):
                # directory of images
                self.data = [
                    os.path.normpath(os.path.join(path, child_path))
                    for child_path in tqdm(os.listdir(path), desc="Loading images", total=len(os.listdir(path)))
                    if any(child_path.lower().endswith(ending) for ending in IMAGE_FORMATS)
                ]
            else:
                raise ValueError(f"string is neither file nor dir. Got '{path}'.")  # pragma: no cover
        else:
            raise TypeError(
                f"Unknown path object, expected filepath, dirpath, or list of filepaths. Got {type(path)}"
            )  # pragma: no cover

    def arbitrary_to_ds(self, a: FilePath, idx: int) -> State:
        """Given a filepath, predict the bounding boxes and key-points of the respective image.
        Return a State containing all the available information.
        Because the state is known, the image is not saved in the State, to reduce the space-overhead on the GPU.
        """
        # the torch model expects a 3D image
        images = [convert_image_dtype(tvte.Image(load_image(a), device=self.device), dtype=torch.float32)]

        outputs = self.model(images)

        s = self.outputs_to_states(outputs=outputs, images=images)

        s.filepath = tuple(a for _ in range(len(s)))

        return s


class KeypointRCNNVideoBackbone(KeypointRCNNBackbone, VideoDataset):
    """A Dataset that gets the path to a single Video file and predicts the bounding boxes and key points of the Video.

    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn

    Params
    ------

    threshold (float):
        Detections with a score lower than the threshold will be ignored.
        Default `DEF_CONF.backbone.kprcnn.threshold`.

    Optional Params
    ---------------

    crop_size (:obj:`.ImgSize`):
        The size, the image crop should have.
        Default `DEF_CONF.images.crop_size`

    crop_mode (str):
        The mode to use when cropping the image.
        Default `DEF_CONF.images.crop_mode`
    """

    data: VideoReader

    def __init__(self, config: Config, path: NodePath) -> None:
        KeypointRCNNBackbone.__init__(self, config=config, path=path)
        VideoDataset.__init__(self, config, path)
        # the data has already been loaded in the VideoDataset
        # the model and threshold has been loaded in KeypointRCNNBackbone

    def arbitrary_to_ds(self, a: Image, idx: int) -> State:
        """Given a frame of a video, return the resulting state after running the RCNN model."""
        # the torch RCNN model expects a list of 3D images
        images = [convert_image_dtype(a, torch.float32)]

        outputs = self.model(images)

        s = self.outputs_to_states(outputs=outputs, images=images)

        s.image = [a.unsqueeze(0) for _ in range(len(s))]

        return s
