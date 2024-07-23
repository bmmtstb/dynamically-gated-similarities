import os
import unittest
import warnings
from unittest.mock import patch

import torch
from torch.nn import Module as TorchModule
from torchvision.io import VideoReader
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

from dgs.models.dataset.dataset import BaseDataset, ImageDataset, VideoDataset
from dgs.models.dataset.keypoint_rcnn import KeypointRCNNBackbone, KeypointRCNNImageBackbone, KeypointRCNNVideoBackbone
from dgs.models.loader import get_data_loader
from dgs.utils.config import DEF_VAL, fill_in_defaults
from dgs.utils.state import State
from helper import get_test_config, load_test_image

IMG_FOLDER_PATH = os.path.abspath("./tests/test_data/images/")
VID_FOLDER_PATH = os.path.abspath("./tests/test_data/videos/")

IMAGE_PATH = os.path.abspath(os.path.join(IMG_FOLDER_PATH, "torch_person.jpg"))
IMAGE = load_test_image("torch_person.jpg")

VIDEO_PATH = os.path.abspath(os.path.join(VID_FOLDER_PATH, "3209828-sd_426_240_25fps.mp4"))


class TestKPRCNNModel(unittest.TestCase):

    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1

    @patch.multiple(KeypointRCNNBackbone, __abstractmethods__=set())
    def test_init_model(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"data_path": IMAGE_PATH, "dataset_path": "", "weights": self.weights},
            },
            get_test_config(),
        )
        m = KeypointRCNNBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m, BaseDataset))
        self.assertTrue(isinstance(m, KeypointRCNNBackbone))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.score_threshold > 0.0)

    def test_init_image_backbone(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"data_path": IMAGE_PATH, "dataset_path": "", "weights": self.weights},
            },
            get_test_config(),
        )
        m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m, KeypointRCNNBackbone))
        self.assertTrue(isinstance(m, ImageDataset))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.score_threshold > 0.0)

    def test_init_video_backbone(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"data_path": VIDEO_PATH, "dataset_path": "", "weights": self.weights},
            },
            get_test_config(),
        )
        m = KeypointRCNNVideoBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m, KeypointRCNNBackbone))
        self.assertTrue(isinstance(m, VideoDataset))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.score_threshold > 0.0)

    def test_init_image_data(self):
        for path, length in [
            (IMAGE_PATH, 1),  # string
            ([IMAGE_PATH, IMAGE_PATH], 2),  # list of file names
            (IMG_FOLDER_PATH, 14),  # directory
        ]:
            with self.subTest(msg="path: {}, length: {}".format(path, length)):
                cfg = fill_in_defaults(
                    {"kprcnn": {"data_path": path, "dataset_path": "", "weights": self.weights}}, get_test_config()
                )
                m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
                self.assertTrue(isinstance(m.data, list))
                self.assertEqual(len(m.data), length)

    def test_init_video_data(self):
        cfg = fill_in_defaults(
            {
                "kprcnn": {"data_path": VIDEO_PATH, "dataset_path": "", "weights": self.weights},
            },
            get_test_config(),
        )
        m = KeypointRCNNVideoBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m.data, VideoReader))
        self.assertEqual(len(m), 345)

    def test_init_data_loading_exceptions(self):
        for path, exception, err_msg in [
            (os.path.join(IMG_FOLDER_PATH, "license.txt"), NotImplementedError, "Unknown file type. Got"),
            (VIDEO_PATH, TypeError, "Got Video file, but is an Image Dataset"),
        ]:
            with self.subTest(msg="path: {}, exception: {}, err_msg: {}".format(path, exception, err_msg)):
                cfg = fill_in_defaults(
                    {
                        "kprcnn": {
                            "data_path": path,
                            "dataset_path": "",
                            "weights": self.weights,
                        }
                    },
                    get_test_config(),
                )
                with self.assertRaises(exception) as e:
                    _ = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
                self.assertTrue(err_msg in str(e.exception), msg=e.exception)

    def test_predict_empty(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "score_threshold": 0.9,
                    "data_path": [os.path.abspath(os.path.join(IMG_FOLDER_PATH, "866-256x256.jpg"))],
                    "dataset_path": "",
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )
        m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
        out_list: list[State]

        self.assertEqual(len(m), 1)

        for out_list in m:
            self.assertTrue(isinstance(out_list, list))
            self.assertEqual(len(out_list), 1)

            out: State = out_list[0]
            self.assertEqual(out.B, 0)
            self.assertEqual(out.filepath, (cfg["kprcnn"]["data_path"][0],))
            self.assertEqual(len(out["image_id"]), 1)
            self.assertEqual(len(out["frame_id"]), 1)
            for key in ["keypoints", "keypoints_local", "image", "skeleton_name", "scores"]:
                self.assertTrue(key not in out)

    def test_image_iou(self):
        for iou_thresh, detections in [(0.4, 1), (0.5, 2)]:
            with self.subTest(msg="iou_thresh: {}".format(iou_thresh)):
                cfg = fill_in_defaults(
                    {
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "kprcnn": {
                            "score_threshold": 0.0,
                            "iou_threshold": iou_thresh,
                            "data_path": [IMAGE_PATH],
                            "dataset_path": "",
                            "weights": self.weights,
                        },
                    },
                    get_test_config(),
                )
                m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
                out_list: list[State]
                for out_list in m:
                    self.assertTrue(isinstance(out_list, list))
                    self.assertEqual(len(out_list), 1)
                    out: State = out_list[0]

                    self.assertTrue(isinstance(out, State))
                    self.assertEqual(out.B, detections, f"expected {detections} detections, but got {out.B}")
                    self.assertEqual(len(out.person_id), detections)
                    self.assertEqual(len(out["image_id"]), detections)
                    self.assertEqual(len(out["frame_id"]), detections)
                    self.assertEqual(len(out["skeleton_name"]), detections)
                    self.assertEqual(len(out["scores"]), detections)
                    self.assertEqual(len(out["score"]), detections)
                    self.assertEqual(len(out.bbox), detections)
                    self.assertEqual(len(out.image_crop), detections)
                    self.assertEqual(len(out.keypoints), detections)
                    self.assertEqual(len(out.keypoints_local), detections)
                    self.assertEqual(len(out.joint_weight), detections)
                    self.assertTrue(
                        torch.any(out["scores"] > 0.9), f"at least one score should be high, got: {out['scores']}"
                    )

    def test_masked(self):
        # mask to crop the skateboarder from the torch_person image is in pt21_dummy_1.json
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "score_threshold": 0.2,
                    "iou_threshold": 1.0,
                    "data_path": [IMAGE_PATH],
                    "dataset_path": "./tests/test_data/",
                    "mask_path": "./tests/test_data/pt21/pt21_dummy_1.json",
                    "force_reshape": True,
                    "image_size": (1024, 1024),  # [H,W]
                    "image_mode": "zero-pad",
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )
        m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
        out_list: list[State]
        detections = 0

        for out_list in m:
            self.assertTrue(isinstance(out_list, list))
            self.assertEqual(len(out_list), 1)
            out: State = out_list[0]

            self.assertTrue(isinstance(out, State))
            self.assertEqual(out.B, detections, f"expected {detections} detections, but got {out.B}")

    def test_masked_batched(self):
        # mask to crop the skateboarder from the torch_person image is in pt21_dummy_2.json
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "module_name": "KeypointRCNNImageBackbone",
                    "score_threshold": 0.2,
                    "iou_threshold": 1.0,
                    "data_path": [IMAGE_PATH, IMAGE_PATH],
                    "dataset_path": "./tests/test_data/",
                    "mask_path": "./tests/test_data/pt21/pt21_dummy_2.json",
                    "force_reshape": True,
                    "image_size": (1024, 1024),  # [H,W]
                    "image_mode": "zero-pad",
                    "batch_size": 2,
                    "return_lists": True,
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )
        m = get_data_loader(config=cfg, path=["kprcnn"])
        out_list: list[State]
        detections = 0

        for out_list in m:
            self.assertTrue(isinstance(out_list, list))
            self.assertEqual(len(out_list), 2)

            for out in out_list:
                self.assertTrue(isinstance(out, State))
                self.assertEqual(out.B, detections, f"expected {detections} detections, but got {out.B}")

    def test_dataset_image(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "score_threshold": 0.5,
                    "iou_threshold": 0.5,
                    "data_path": [IMAGE_PATH, IMAGE_PATH],
                    "dataset_path": "",
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )
        detections = 1
        m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
        out_list: list[State]
        for out_list in m:
            self.assertTrue(isinstance(out_list, list))
            self.assertEqual(len(out_list), 1)

            out: State = out_list[0]
            self.assertTrue(isinstance(out, State))
            self.assertEqual(out.B, 1)
            self.assertTrue("bbox" in out.data)
            self.assertTrue("keypoints" in out.data)
            self.assertTrue("keypoints_local" in out.data)
            self.assertTrue("joint_weight" in out.data)
            self.assertTrue("filepath" in out.data)
            self.assertTrue("image_crop" in out.data)

            self.assertEqual(len(out.person_id), detections)
            self.assertEqual(len(out["image_id"]), detections)
            self.assertEqual(len(out["frame_id"]), detections)
            self.assertEqual(len(out["skeleton_name"]), detections)
            self.assertEqual(len(out["scores"]), detections)
            self.assertEqual(len(out["score"]), detections)
            self.assertEqual(len(out.bbox), detections)
            self.assertEqual(len(out.image_crop), detections)
            self.assertEqual(len(out.keypoints), detections)
            self.assertEqual(len(out.keypoints_local), detections)
            self.assertEqual(len(out.joint_weight), detections)

            self.assertEqual(out.image[0].ndim, 4)
            self.assertEqual(out.image[0].size(0), detections)
            self.assertEqual(out.image_crop.ndim, 4)
            self.assertEqual(out.image_crop.size(0), detections)
            self.assertEqual(out.bbox.shape, torch.Size((detections, 4)))
            self.assertEqual(out.keypoints.shape, torch.Size((detections, 17, 2)))
            self.assertEqual(out.joint_weight.shape, torch.Size((detections, 17, 1)))

    def test_dataloader_image(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "module_name": "KeypointRCNNImageBackbone",
                    "score_threshold": 0.5,
                    "data_path": [IMAGE_PATH, IMAGE_PATH],
                    "dataset_path": "",
                    "batch_size": 2,
                    "return_lists": True,
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )
        detections = 1
        m = get_data_loader(config=cfg, path=["kprcnn"])
        self.assertEqual(len(m), 1)

        for out_list in m:
            self.assertTrue(isinstance(out_list, list))
            self.assertEqual(len(out_list), 2)

            for out in out_list:
                self.assertTrue(isinstance(out, State))
                self.assertEqual(out.B, 1)
                self.assertTrue("bbox" in out.data)
                self.assertTrue("keypoints" in out.data)
                self.assertTrue("keypoints_local" in out.data)
                self.assertTrue("joint_weight" in out.data)
                self.assertTrue("filepath" in out.data)
                self.assertTrue("image_crop" in out.data)

                self.assertEqual(out.image[0].ndim, 4)
                self.assertEqual(out.image[0].size(0), detections)
                self.assertEqual(out.image_crop.ndim, 4)
                self.assertEqual(out.image_crop.size(0), detections)
                self.assertEqual(out.bbox.shape, torch.Size((detections, 4)))
                self.assertEqual(out.keypoints.shape, torch.Size((detections, 17, 2)))
                self.assertEqual(out.joint_weight.shape, torch.Size((detections, 17, 1)))

    def test_dataset_video(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "score_threshold": 0.5,
                    "data_path": VIDEO_PATH,
                    "dataset_path": "",
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )

        m = KeypointRCNNVideoBackbone(config=cfg, path=["kprcnn"])

        i = 0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Accurate seek is not implemented for pyav backend", category=UserWarning
            )
            out_list: list[State]
            for out_list in m:
                self.assertTrue(isinstance(out_list, list))
                self.assertEqual(len(out_list), 1)

                out: State = out_list[0]
                self.assertTrue(isinstance(out, State))
                self.assertEqual(out.image[0].shape, torch.Size((1, 3, 240, 426)))
                self.assertEqual(out.image_crop[0].shape, torch.Size((3, *DEF_VAL["images"]["crop_size"])))
                i += 1
                if i >= 2:
                    break

    def test_dataloader_video(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {
                    "module_name": "KeypointRCNNVideoBackbone",
                    "score_threshold": 0.5,
                    "data_path": VIDEO_PATH,
                    "dataset_path": "",
                    "batch_size": 2,
                    "return_lists": True,
                    "weights": self.weights,
                },
            },
            get_test_config(),
        )

        m = get_data_loader(config=cfg, path=["kprcnn"])
        i = 0

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Accurate seek is not implemented for pyav backend", category=UserWarning
            )

            for out_list in m:
                self.assertTrue(isinstance(out_list, list))
                self.assertEqual(len(out_list), 2)
                for out in out_list:
                    self.assertTrue(isinstance(out, State))
                    self.assertEqual(out.image[0].shape, torch.Size((1, 3, 240, 426)))
                    self.assertEqual(out.image_crop[0].shape, torch.Size((3, *DEF_VAL["images"]["crop_size"])))
                i += 1
                if i >= 2:
                    break


if __name__ == "__main__":
    unittest.main()
