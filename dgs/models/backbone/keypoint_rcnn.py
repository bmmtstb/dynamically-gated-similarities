"""
Use 'keypointrcnn_resnet50_fpn' from PyTorch.

References:
    https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
"""

from typing import Union

import torch
from torch.nn import Module as TorchModule
from torchvision import tv_tensors as tvte
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

from dgs.models.backbone.backbone import BaseBackboneModule
from dgs.utils.config import DEF_CONF
from dgs.utils.image import load_image_list
from dgs.utils.state import collate_states, State
from dgs.utils.types import Config, FilePath, FilePaths, NodePath, Validations

rcnn_validations: Validations = {"threshold": ["optional", float, ("within", (0.0, 1.0))]}


class KeypointRCNNBackbone(BaseBackboneModule, TorchModule):
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

    def __init__(self, config: Config, path: NodePath) -> None:
        BaseBackboneModule.__init__(self, config, path)
        TorchModule.__init__(self)

        self.validate_params(rcnn_validations)

        self.threshold: float = self.params.get("threshold", DEF_CONF.backbone.kprcnn.threshold)

        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
        self.model.eval()
        self.model.to(self.device)

    def forward(self, image_paths: Union[FilePath, FilePaths], *args, **kwargs) -> State:
        """Given a tuple of image paths, return a State containing the bounding boxes and the predicted key-points."""
        if isinstance(image_paths, str):
            image_paths = (image_paths,)
        assert len(image_paths) > 0, "No images provided"

        images = [
            img.squeeze(0) for img in load_image_list(filepath=image_paths, dtype=torch.float32, device=self.device)
        ]
        outputs = self.model(images)

        states = []
        canvas_size = (max(i.shape[-2] for i in images), max(i.shape[-1] for i in images))

        for output, path in zip(outputs, image_paths):
            # for every image (output), get the indices where score is bigger than the threshold
            idx = torch.where(output["scores"] > self.threshold)

            # bbox given in XYXY format
            bbox = tvte.BoundingBoxes(output["boxes"][idx], format="XYXY", canvas_size=canvas_size)
            # keypoints in [x,y,v] format -> kp, vis
            kp, vis = (
                output["keypoints"][idx]
                .to(device=self.device, dtype=self.precision)
                .reshape((-1, 17, 3))
                .split([2, 1], dim=-1)
            )

            # todo compute image crops

            data = {
                "validate": False,
                "bbox": bbox,
                "filepath": tuple(path for _ in range(len(bbox))),  # there might be multiple detections
                "keypoints": kp,
                "joint_weight": vis,
                # "scores": output["scores"],
            }
            states.append(State(**data))

        return collate_states(states)
