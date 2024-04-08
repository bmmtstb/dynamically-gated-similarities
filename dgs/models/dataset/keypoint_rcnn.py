"""
Use 'keypointrcnn_resnet50_fpn' from PyTorch.

References:
    https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
"""

import os
from typing import Union

import torch
from torch.nn import Module as TorchModule
from torchvision import tv_tensors as tvte
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset
from dgs.utils.config import DEF_CONF
from dgs.utils.constants import IMAGE_FORMATS, VIDEO_FORMATS
from dgs.utils.files import is_dir, is_file
from dgs.utils.image import load_image, load_video
from dgs.utils.state import collate_states, State
from dgs.utils.types import Config, FilePath, Image, NodePath, Validations, Video
from dgs.utils.utils import extract_crops_from_images

rcnn_validations: Validations = {
    "path": [("any", [str, [list, ("forall", str)]])],
    # optional
    "threshold": ["optional", float, ("within", (0.0, 1.0))],
}


class KeypointRCNNBackbone(BaseDataset, TorchModule):
    """

    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn

    Params
    ------

    threshold (float):
        Detections with a score lower than the threshold will be ignored.
        Default `DEF_CONF.backbone.kprcnn.threshold`.
    """

    data: Union[list, Video]

    def __init__(self, config: Config, path: NodePath) -> None:
        BaseDataset.__init__(self, config, path)
        TorchModule.__init__(self)

        self.validate_params(rcnn_validations)

        self.threshold: float = self.params.get("threshold", DEF_CONF.backbone.kprcnn.threshold)

        self.logger.info("Loading Keypoint-RCNN Model")
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
        self.model.eval()
        self.model.to(self.device)

        # load data - path is either a directory, a single file, or a list of image filepaths
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
                    self.logger.info(f"Loading video file: '{path}'")
                    self.data = load_video(path, device=self.device)
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

    def arbitrary_to_ds(self, a: Union[Image, FilePath], idx: int) -> State:
        """Given a single image, a batch of images, or a filepath,
        predict the bounding boxes and key-points of the image(s).
        Return a State containing all the available information.
        """
        if isinstance(a, str):
            a: Image = load_image(a)
        elif not isinstance(a, torch.Tensor):
            raise TypeError(f"Expected input to be an image, got {type(a)}.")

        # the torch model expects a 3D image
        if a.ndim == 3:
            assert a.size(-3) == 3, "Image should be RGB"
            images = [convert_image_dtype(tvte.Image(a, device=self.device), dtype=torch.float32)]
        elif a.ndim == 4:
            assert a.size(-3) == 3, "Image should be RGB"
            images = [convert_image_dtype(tvte.Image(img, device=self.device), dtype=torch.float32) for img in a]
        else:
            raise NotImplementedError(f"Got image with unknown shape {a.shape}")

        outputs = self.model(images)

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

            crop, loc_kps = extract_crops_from_images(
                imgs=new_images,
                bboxes=bbox,
                kps=kps,
                crop_size=self.params.get("crop_size", DEF_CONF.images.crop_size),
                crop_mode=self.params.get("crop_mode", DEF_CONF.images.crop_mode),
            )

            data = {
                "validate": False,
                "bbox": bbox,
                "image": new_images,
                "image_crop": crop,
                "keypoints": kps,
                "keypoints_local": loc_kps,
                "joint_weight": vis,
                "scores": output["scores"],
            }
            states.append(State(**data))

        return collate_states(states)
