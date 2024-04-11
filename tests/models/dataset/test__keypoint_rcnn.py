import os
import unittest
import warnings
from unittest.mock import patch

import torch
from torch.nn import Module as TorchModule
from torchvision.io import VideoReader

from dgs.models.dataset.dataset import BaseDataset, VideoDataset
from dgs.models.dataset.keypoint_rcnn import KeypointRCNNBackbone, KeypointRCNNImageBackbone, KeypointRCNNVideoBackbone
from dgs.utils.config import DEF_CONF, fill_in_defaults
from dgs.utils.state import State
from helper import get_test_config, load_test_image

IMG_FOLDER_PATH = os.path.abspath("./tests/test_data/images/")
VID_FOLDER_PATH = os.path.abspath("./tests/test_data/videos/")

IMAGE_PATH = os.path.abspath(os.path.join(IMG_FOLDER_PATH, "torch_person.jpg"))
IMAGE = load_test_image("torch_person.jpg")

VIDEO_PATH = os.path.abspath(os.path.join(VID_FOLDER_PATH, "3209828-sd_426_240_25fps.mp4"))


class TestKPRCNNModel(unittest.TestCase):

    @patch.multiple(KeypointRCNNBackbone, __abstractmethods__=set())
    def test_init_model(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"path": IMAGE_PATH, "dataset_path": ""},
            },
            get_test_config(),
        )
        m = KeypointRCNNBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m, BaseDataset))
        self.assertTrue(isinstance(m, KeypointRCNNBackbone))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.threshold > 0.0)

    def test_init_image_backbone(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"path": IMAGE_PATH, "dataset_path": ""},
            },
            get_test_config(),
        )
        m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m, KeypointRCNNBackbone))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.threshold > 0.0)

    def test_init_video_backbone(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"path": VIDEO_PATH, "dataset_path": ""},
            },
            get_test_config(),
        )
        m = KeypointRCNNVideoBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m, KeypointRCNNBackbone))
        self.assertTrue(isinstance(m, VideoDataset))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.threshold > 0.0)

    def test_init_image_data(self):
        for path, length in [
            (IMAGE_PATH, 1),  # string
            ([IMAGE_PATH, IMAGE_PATH], 2),  # list of file names
            (IMG_FOLDER_PATH, 14),  # directory
        ]:
            with self.subTest(msg="path: {}, length: {}".format(path, length)):
                cfg = fill_in_defaults({"kprcnn": {"path": path, "dataset_path": ""}}, get_test_config())
                m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
                self.assertTrue(isinstance(m.data, list))
                self.assertEqual(len(m.data), length)

    def test_init_video_data(self):
        cfg = fill_in_defaults({"kprcnn": {"path": VIDEO_PATH, "dataset_path": ""}}, get_test_config())
        m = KeypointRCNNVideoBackbone(config=cfg, path=["kprcnn"])
        self.assertTrue(isinstance(m.data, VideoReader))
        self.assertEqual(len(m), 345)

    def test_init_data_loading_exceptions(self):
        for path, exception, err_msg in [
            (os.path.join(IMG_FOLDER_PATH, "license.txt"), NotImplementedError, "Unknown file type. Got"),
            (VIDEO_PATH, TypeError, "Got Video file, but is an Image Dataset"),
        ]:
            with self.subTest(msg="path: {}, exception: {}, err_msg: {}".format(path, exception, err_msg)):
                cfg = fill_in_defaults({"kprcnn": {"path": path, "dataset_path": ""}}, get_test_config())
                with self.assertRaises(exception) as e:
                    _ = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
                self.assertTrue(err_msg in str(e.exception), msg=e.exception)

    def test_dataset_image(self):
        cfg = fill_in_defaults(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "kprcnn": {"threshold": 0.5, "path": [IMAGE_PATH, IMAGE_PATH], "dataset_path": ""},
            },
            get_test_config(),
        )
        detections = 1
        m = KeypointRCNNImageBackbone(config=cfg, path=["kprcnn"])
        for out in m:
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
                "kprcnn": {"threshold": 0.5, "path": VIDEO_PATH, "dataset_path": ""},
            },
            get_test_config(),
        )

        m = KeypointRCNNVideoBackbone(config=cfg, path=["kprcnn"])

        i = 0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Accurate seek is not implemented for pyav backend", category=UserWarning
            )
            for out in m:
                self.assertTrue(isinstance(out, State))
                self.assertEqual(out.image[0].shape, torch.Size((1, 3, 240, 426)))
                self.assertEqual(out.image_crop[0].shape, torch.Size((3, *DEF_CONF.images.crop_size)))
                i += 1
                if i >= 2:
                    break


if __name__ == "__main__":
    unittest.main()
