"""
Use :func:`.keypointrcnn_resnet50_fpn` to predict the key points and bounding boxes of each image.

References:
    https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html
"""

import os
from abc import ABC
from typing import Union

import torch as t
from imagesize import imagesize
from torch import nn
from torchvision import tv_tensors as tvte
from torchvision.io import VideoReader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_dtype
from tqdm import tqdm

from dgs.models.dataset.dataset import BaseDataset, ImageDataset, VideoDataset
from dgs.utils.config import DEF_VAL
from dgs.utils.constants import IMAGE_FORMATS, VIDEO_FORMATS
from dgs.utils.files import is_dir, is_file, read_json
from dgs.utils.image import create_mask_from_polygons, CustomToAspect, load_image
from dgs.utils.state import EMPTY_STATE, State
from dgs.utils.types import Config, FilePath, FilePaths, Image, Images, ImgShape, NodePath, Validations
from dgs.utils.utils import extract_crops_from_images

rcnn_validations: Validations = {
    "data_path": [("any", [str, ("all", [list, ("forall", str)])])],
    # optional
    "score_threshold": ["optional", float, ("within", (0.0, 1.0))],
    "iou_threshold": ["optional", float, ("within", (0.0, 1.0))],
    "force_reshape": ["optional", bool],
    "image_mode": ["optional", str, ("in", CustomToAspect.modes)],
    "image_size": ["optional", tuple, ("len", 2), ("forall", [int, ("gt", 0)])],
    "crop_mode": ["optional", str, ("in", CustomToAspect.modes)],
    "crop_size": ["optional", tuple, ("len", 2), ("forall", [int, ("gt", 0)])],
    "bbox_min_size": ["optional", float, ("gte", 1.0)],
    "mask_path": ["optional", str],
    "weights": ["optional", ("instance", KeypointRCNN_ResNet50_FPN_Weights)],
}


class KeypointRCNNBackbone(BaseDataset, nn.Module, ABC):
    """Metaclass for the torchvision Key Point RCNN backbone model.

    This class sets up the RCNN model and validates and sets the basic modules parameters.

    Params
    ------

    data_path (FilePath):
        A single path or a list of paths.
        The path is either a directory, a single image file, or a list of image filepaths.

    Optional Params
    ---------------

    score_threshold (float, optional):
        Detections with a score lower than the threshold will be ignored.
        Default ``DEF_VAL.dataset.kprcnn.score_threshold``.
    iou_threshold (float, optional):
        Bounding-boxes with IoU above this threshold will be ignored.
        Default ``DEF_VAL.dataset.kprcnn.iou_threshold``.
    force_reshape (bool, optional):
        Whether to force reshape all the input images.
        Change the size and mode via ``image_mode`` and ``image_size`` parameters, iff ``force_reshape`` is `True`.
        Default ``DEF_VAL.images.force_reshape``.
    image_size (:obj:`ImgSize`, optional):
        The size, the loaded image should have, iff ``force_reshape`` is `True`.
        Default ``DEF_VAL.images.image_size``.
    image_mode (str, optional):
        The mode to use when loading the image, iff ``force_reshape`` is `True`.
        Default ``DEF_VAL.images.image_mode``.
    crop_size (:obj:`ImgSize`, optional):
        The size, the image crop should have.
        Default ``DEF_VAL.images.crop_size``.
    crop_mode (str, optional):
        The mode to use when cropping the image.
        Default ``DEF_VAL.images.crop_mode``.
    bbox_min_size (float, optional):
        The minimum side length a bounding box should have in pixels.
        Smaller detections will be discarded.
        Works in addition to the ``threshold`` parameter.
        If you do not want to discard smaller bounding boxes, make sure to set ``bbox_min_size`` to ``1.0``.
        The size of the bounding boxes is in relation to the original image.
        Default ``DEF_VAL.images.bbox_min_size``.
    weights (KeypointRCNN_ResNet50_FPN_Weights, optional):
        The weights to load for the model.
        Default ``KeypointRCNN_ResNet50_FPN_Weights.COCO_V1``.
    """

    model: nn.Module

    def __init__(self, config: Config, path: NodePath) -> None:
        BaseDataset.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(rcnn_validations)

        self.score_threshold: float = self.params.get(
            "score_threshold", DEF_VAL["dataset"]["kprcnn"]["score_threshold"]
        )

        self.logger.debug("Loading Keypoint-RCNN Model")
        weights = self.params.get("weights", KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
        model = keypointrcnn_resnet50_fpn(weights=weights, progress=True)
        self.register_module("model", self.configure_torch_module(module=model, train=False))

        self.img_id: t.Tensor = t.tensor(1, dtype=t.long, device=self.device)

        bbox_min_size: float = float(self.params.get("bbox_min_size", DEF_VAL["images"]["bbox_min_size"]))
        self.bbox_cleaner = v2.Compose(
            [
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(
                    min_size=bbox_min_size,
                    labels_getter=lambda y: (y["keypoints"], y["scores"], y["keypoints_scores"], y["labels"]),
                ),
            ]
        )

        self.iou_threshold: float = self.params.get("iou_threshold", DEF_VAL["dataset"]["kprcnn"]["iou_threshold"])

        # image loading params
        self.force_reshape: bool = self.params.get("force_reshape", DEF_VAL["images"]["force_reshape"])
        self.image_size: ImgShape = self.params.get("image_size", DEF_VAL["images"]["image_size"])
        self.image_mode: str = self.params.get("image_mode", DEF_VAL["images"]["image_mode"])

    @t.no_grad()
    def images_to_states(self, images: Images) -> list[State]:
        """Given a list of images, use the key-point-RCNN model to predict key points and bounding boxes,
        then create a :class:`.State` containing the available information.

        Notes:
            Does not add the original image to the new State, to reduce memory / GPU usage.
            With the filepath given in the state, the image can be reloaded if required.
        """
        # make sure all images are float
        images = [tvte.Image(to_dtype(img, dtype=t.float32, scale=True)) for img in images]

        # predicts a list of {boxes: XYXY[N], labels: Int64[N], scores: [N], keypoints: Float[N,J,(x|y|vis)]}
        # every image in images can have multiple predictions
        outputs: list[dict[str, t.Tensor]] = self.model.forward(images)

        states: list[State] = []
        canvas_size = (max(i.shape[-2] for i in images), max(i.shape[-1] for i in images))

        for output, image in zip(outputs, images):
            # get the output for every image independently

            # bbox given in XYXY format
            output["boxes"] = tvte.BoundingBoxes(output["boxes"], format="XYXY", canvas_size=canvas_size)

            # first sanitize and clamp the bboxes, while cleaning up the respective other data as well
            sanitized = self.bbox_cleaner(output)
            scores = sanitized["scores"]  # score of each instance

            # Get the sanitized bboxes and compute the IoU.
            bbox = tvte.BoundingBoxes(sanitized["boxes"], format="XYXY", canvas_size=canvas_size)
            iou = box_iou(bbox, bbox).tril(diagonal=-1)  # lower tri excluding diag
            # Filter the bboxes using an IoU threshold.
            # Additionally, use only the indices where the score ('certainty') is bigger than the given score_threshold.
            # Because the output of KeypointRCNN is sorted by score,
            # using the lower triangular matrix will remove the lower score.
            indices = t.logical_and(
                t.logical_not(t.any(iou > self.iou_threshold, dim=1)),  # iou smaller than
                scores > self.score_threshold,  # score > thresh
            )

            # get final bbox and B after double sanitizing
            bbox = tvte.BoundingBoxes(sanitized["boxes"][indices], format="XYXY", canvas_size=canvas_size)

            B: int = int(t.count_nonzero(indices).item())

            data = {
                "validate": False,
                "image_id": t.ones(max(B, 1), device=self.device, dtype=t.long) * self.img_id,
                "frame_id": t.ones(max(B, 1), device=self.device, dtype=t.long) * self.img_id,
            }
            self.img_id += t.tensor(1, dtype=t.long, device=self.device)  # increment counter

            # skip if there aren't any detections
            if B == 0:
                es = EMPTY_STATE.copy()
                es.update(data)
                states.append(es)
                continue

            # keypoints in [x,y,v] format -> kp, vis
            kps, vis = (
                sanitized["keypoints"][indices]
                .to(device=self.device, dtype=self.precision)
                .reshape((-1, 17, 3))
                .split([2, 1], dim=-1)
            )
            assert kps.shape[-2:] == (17, 2), kps.shape[-2:]

            crops, loc_kps = extract_crops_from_images(
                imgs=[tvte.Image(image.unsqueeze(0)) for _ in range(B)],
                bboxes=bbox,
                kps=kps,
                crop_size=self.params.get("crop_size", DEF_VAL["images"]["crop_size"]),
                crop_mode=self.params.get("crop_mode", DEF_VAL["images"]["crop_mode"]),
            )

            assert loc_kps is not None

            data = dict(
                data,
                **{
                    "skeleton_name": tuple("coco" for _ in range(B)),
                    "scores": sanitized["keypoints_scores"][indices, :],  # B x 17
                    "score": scores[indices],
                    "bbox": bbox,
                    "image_crop": crops,
                    "keypoints": kps,
                    "keypoints_local": loc_kps,
                    "joint_weight": vis,
                    "person_id": t.ones(B, device=self.device, dtype=t.long) * -1,  # set as -1
                },
            )

            states.append(State(**data))

        return states

    def terminate(self) -> None:  # pragma: no cover
        if hasattr(self, "model"):
            del self.model


# pylint: disable=too-many-ancestors
class KeypointRCNNImageBackbone(KeypointRCNNBackbone, ImageDataset):
    """Predicts 17 key-points (like COCO).

    Optional Params
    ---------------

    mask_path (str, optional):
        The path to a PT21 json file containing the ``ignore_regions``.
        Note that currently only PT21 ignore regions are supported.

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html
    """

    __doc__ += KeypointRCNNBackbone.__doc__

    data: list[FilePath]
    masks: list[Union[tvte.Mask, None]]

    def __init__(self, config: Config, path: NodePath) -> None:
        KeypointRCNNBackbone.__init__(self, config=config, path=path)
        ImageDataset.__init__(self, config=config, path=path)

        # load data - data_path is either a directory, a single image file, or a list of image filepaths
        self.data = []
        data_path: any = self.params["data_path"]
        if isinstance(data_path, list):
            self.data = sorted(data_path)
        elif isinstance(data_path, str):
            data_path: FilePath = self.get_path_in_dataset(data_path)
            if is_file(data_path):
                # single image
                if data_path.lower().endswith(IMAGE_FORMATS):
                    self.data = [data_path]
                # video file
                elif data_path.lower().endswith(VIDEO_FORMATS):
                    raise TypeError(f"Got Video file, but is an Image Dataset. File: {data_path}")
                else:
                    raise NotImplementedError(f"Unknown file type. Got '{data_path}'")
            elif is_dir(data_path):
                # directory of images
                self.data = [
                    os.path.normpath(os.path.join(data_path, child_path))
                    for child_path in tqdm(sorted(os.listdir(data_path)), desc="Loading images", leave=False)
                    if child_path.lower().endswith(IMAGE_FORMATS)
                ]
            else:
                raise NotImplementedError(f"string is neither file nor dir. Got '{data_path}'.")
        else:
            raise NotImplementedError(
                f"Unknown path object, expected filepath, dirpath, or list of filepaths. Got {type(data_path)}"
            )

        # fixme what about other masking types?
        if "mask_path" in self.params and self.force_reshape:
            self.masks = [
                self.transform_resize_image()(
                    {
                        "image": tvte.Image(
                            create_mask_from_polygons(
                                img_size=imagesize.get(self.get_path_in_dataset(img["file_name"]))[::-1],
                                polygons_x=img["ignore_regions_x"],
                                polygons_y=img["ignore_regions_y"],
                                device=self.device,
                            ).unsqueeze(0)
                        ),
                        "box": tvte.BoundingBoxes(
                            t.ones((1, 4), dtype=t.float32), canvas_size=self.image_size, format="XYWH", dtype=t.float32
                        ),
                        "keypoints": t.ones((1, 15, 2)),
                        "output_size": self.image_size,
                        "mode": self.image_mode,
                    }
                )["image"]
                .squeeze(0)
                .to(dtype=t.bool)
                for img in read_json(self.params["mask_path"])["images"]
            ]
        elif "mask_path" in self.params:
            self.masks = [
                create_mask_from_polygons(
                    img_size=imagesize.get(self.get_path_in_dataset(img["file_name"]))[::-1],
                    polygons_x=img["ignore_regions_x"],
                    polygons_y=img["ignore_regions_y"],
                    device=self.device,
                )
                for img in read_json(self.params["mask_path"])["images"]
            ]
        else:
            self.masks = [None for _ in range(len(self.data))]

    def arbitrary_to_ds(self, a: FilePath, idx: int) -> list[State]:
        """Given a filepath, predict the bounding boxes and key-points of the respective image.
        Return a State containing all the available information.
        Because the state is known, the image is not saved in the State, to reduce the space-overhead on the GPU.

        Args:
            a: A single path to an image file.
            idx: The index of the file path within ``self.data``.

        Returns:
            A list containing one single :class:`.State` that describes zero or more detections of the given image.
        """
        img = load_image(
            filepath=a,
            force_reshape=self.force_reshape,
            output_size=self.image_size,
            mode=self.image_mode,
            device=self.device,
            dtype=t.float32,
        ).squeeze(0)

        if self.masks[idx] is not None:
            # Get the mask with the same size as the image.
            # True, where the image should be ignored.
            mask = self.masks[idx]
            assert not self.force_reshape or mask.shape == t.Size(self.image_size), (mask.shape, self.image_size)
        else:
            mask = t.zeros(img.shape[-2:], device=self.device, dtype=t.bool)

        # create the image by using the unmasked area of the image and the masked area of a black image
        masked_img = tvte.Image(img * t.bitwise_not(mask) + t.zeros_like(img) * mask, device=self.device)

        # the torch model expects a list of 3D images
        states = self.images_to_states(images=[masked_img])

        for state in states:
            state.filepath = tuple(a for _ in range(max(state.B, 1)))

        return states

    def __getitems__(self, indices: list[int]) -> list[State]:
        """Get a batch of predictions from the dataset. It is expected that all images have the same shape.

        Returns:
            A list containing one :class:`.State` per image / index.
            Every State describes zero or more detections of the respective image.
        """
        fps: FilePaths = tuple(self.data[idx] for idx in indices)
        masks = [self.masks[idx] for idx in indices]
        images = load_image(
            fps,
            force_reshape=self.force_reshape,
            output_size=self.image_size,
            mode=self.image_mode,
            device=self.device,
            dtype=t.float32,
        )
        assert not self.force_reshape or all(
            mask.shape == t.Size(self.image_size) for mask in masks if mask is not None
        )
        # the torch model expects a list of 3D images
        masked_images = [
            (img * t.bitwise_not(mask) + t.zeros_like(img) * mask).squeeze(0) if mask is not None else img.squeeze(0)
            for img, mask in zip(images.split(1, dim=0), masks)
        ]

        states = self.images_to_states(images=masked_images)

        for fp, state in zip(fps, states):
            state.filepath = tuple(fp for _ in range(max(state.B, 1)))

        return states


# pylint: disable=too-many-ancestors
class KeypointRCNNVideoBackbone(KeypointRCNNBackbone, VideoDataset):
    """A Dataset that gets the path to a single Video file and predicts the bounding boxes and key points of the Video.

    Predicts 17 key-points (like COCO).

    References:
        https://pytorch.org/vision/0.17/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html
    """

    __doc__ += KeypointRCNNBackbone.__doc__

    data: VideoReader

    def __init__(self, config: Config, path: NodePath) -> None:
        KeypointRCNNBackbone.__init__(self, config=config, path=path)
        VideoDataset.__init__(self, config=config, path=path)
        # the data has already been loaded in the VideoDataset
        # the model and threshold has been loaded in KeypointRCNNBackbone

    def arbitrary_to_ds(self, a: Image, idx: int) -> list[State]:
        """Given a frame of a video, return the resulting state after running the RCNN model."""
        if not isinstance(a, t.Tensor):
            raise NotImplementedError
        # the torch RCNN model expects a list of 3D images
        if self.force_reshape:
            # reshape the images iff requested to the desired size
            images = [
                self.transform_resize_image()(
                    {
                        "image": tvte.Image(a),
                        "box": tvte.BoundingBoxes(
                            t.ones((1, 4), dtype=t.float32),
                            canvas_size=self.image_size,
                            format="XYWH",
                            dtype=t.float32,
                        ),
                        "keypoints": t.ones((1, 15, 2)),
                        "output_size": self.image_size,
                        "mode": self.image_mode,
                    }
                )["image"],
            ]
        else:
            images = [a]

        states = self.images_to_states(images=images)

        for img, state in zip(images, states):
            state.image = [tvte.Image(img.unsqueeze(0)) for _ in range(state.B)]

        return states
