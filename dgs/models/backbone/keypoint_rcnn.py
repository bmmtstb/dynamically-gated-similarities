"""
Use 'keypointrcnn_resnet50_fpn' from PyTorch.

References:
    https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
"""

from typing import Union

import torch
from torchvision import tv_tensors as tvte
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

from dgs.models.backbone.backbone import BaseBackboneModule
from dgs.utils.image import load_image_list
from dgs.utils.state import collate_states, State
from dgs.utils.types import Config, FilePath, FilePaths, NodePath


class KeypointRCNNBackbone(BaseBackboneModule):
    """
    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config, path)

        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
        model.eval()
        model.to(self.device)
        self.model = model

    def forward(self, image_paths: Union[FilePath, FilePaths], *args, **kwargs) -> State:
        """Given a tuple of image paths, return a State containing the bounding boxes and the predicted key-points."""
        assert len(image_paths) > 0, "No images provided"

        images = load_image_list(filepath=image_paths, dtype=torch.float32, device=self.device)
        outputs = self.model(images)

        states = []
        canvas_size = (max(i.shape[-2] for i in images), max(i.shape[-1] for i in images))

        for output, path in zip(outputs, image_paths):
            # bbox given in XYXY format
            bbox = tvte.BoundingBoxes(output["boxes"], format="XYXY", canvas_size=canvas_size)
            # keypoints in [x,y,v] format -> kp, vis
            kp, vis = (
                output["keypoints"]
                .to(device=self.device, dtype=self.precision)
                .reshape((-1, 17, 3))
                .split([2, 1], dim=-1)
            )

            # todo compute image crops
            # todo skip if scores to low?

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
