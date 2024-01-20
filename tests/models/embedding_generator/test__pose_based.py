import unittest

import torch
from torchvision import tv_tensors

from dgs.models.embedding_generator.pose_based import KeyPointConvolutionPBEG, LinearPBEG
from dgs.utils.config import fill_in_defaults
from dgs.utils.types import Device
from helper import test_multiple_devices


class TestPoseBased(unittest.TestCase):
    @test_multiple_devices
    def test_linear_PBEG_out_shape(self, device: Device):
        for batch_size, params, out_shape in [
            (
                32,
                {
                    "hidden_layers": [],
                    "joint_shape": (21, 2),
                    "nof_kernels": 10,
                    "embedding_size": 4,
                    "bbox_format": "xyxy",
                },
                (32, 4),
            ),
            (
                8,
                {
                    "hidden_layers": [15],
                    "joint_shape": (17, 2),
                    "bias": False,
                    "embedding_size": 13,
                    "bbox_format": tv_tensors.BoundingBoxFormat.XYWH,
                },
                (8, 13),
            ),
        ]:
            with self.subTest(msg=f"params: {params}"):
                cfg = fill_in_defaults({"pose_embedding_generator": params, "batch_size": batch_size, "device": device})
                m = LinearPBEG(config=cfg, path=["pose_embedding_generator"])
                kp = torch.rand((batch_size, *params["joint_shape"])).to(device=device)
                bbox = tv_tensors.BoundingBoxes(torch.rand((batch_size, 4)), format="XYWH", canvas_size=(100, 100)).to(
                    device=device
                )
                res = m.forward(kp, bbox)
                self.assertEqual(res.device.type, device.type)
                self.assertEqual(res.shape, out_shape)

    def test_linear_PBEG_flattened(self):
        batch_size = 7
        params = {
            "hidden_layers": [],
            "joint_shape": (21, 2),
            "nof_kernels": 10,
            "embedding_size": 5,
            "bbox_format": "xyxy",
        }
        cfg = fill_in_defaults({"pose_embedding_generator": params, "batch_size": batch_size})
        m = LinearPBEG(config=cfg, path=["pose_embedding_generator"])
        kp = torch.rand((batch_size, *params["joint_shape"])).reshape((batch_size, -1))
        bbox = torch.rand((batch_size, 4)).reshape((batch_size, -1))
        res = m.forward(torch.hstack([kp, bbox]).reshape((batch_size, -1)))
        self.assertEqual(res.shape, (batch_size, 5))

    def test_linear_PBEG_raises(self):
        batch_size = 7
        params = {
            "hidden_layers": [],
            "joint_shape": (21, 2),
            "nof_kernels": 10,
            "embedding_size": 5,
            "bbox_format": "xyxy",
        }
        cfg = fill_in_defaults({"pose_embedding_generator": params, "batch_size": batch_size})
        m = LinearPBEG(config=cfg, path=["pose_embedding_generator"])
        with self.assertRaises(ValueError) as e:
            m.forward()
        self.assertTrue("Data should contain key points and bounding boxes, but has length" in str(e.exception))

    def test_KPCPBEG_raises(self):
        batch_size = 7
        params = {
            "hidden_layers": [],
            "joint_shape": (21, 2),
            "nof_kernels": 10,
            "embedding_size": 4,
            "bbox_format": "xyxy",
        }
        cfg = fill_in_defaults({"pose_embedding_generator": params, "batch_size": batch_size})
        m = KeyPointConvolutionPBEG(config=cfg, path=["pose_embedding_generator"])
        kp = torch.rand((batch_size, *params["joint_shape"]))
        bbox = tv_tensors.BoundingBoxes(torch.rand((batch_size, 4)), format="XYWH", canvas_size=(100, 100))

        with self.assertRaises(ValueError) as e:
            m.forward(kp, bbox, "dummy")
        self.assertTrue("Data should contain key points and bounding boxes, but has length" in str(e.exception))

    @test_multiple_devices
    def test_KPCPBEG_out_shape(self, device: Device):
        for batch_size, params, out_shape in [
            (
                32,
                {
                    "hidden_layers_kp": [],
                    "hidden_layers_all": [],
                    "joint_shape": (21, 2),
                    "nof_kernels": 10,
                    "embedding_size": 4,
                    "bbox_format": "xyxy",
                },
                (32, 4),
            ),
            (
                8,
                {
                    "hidden_layers_kp": [],
                    "hidden_layers_all": [15, 12, 11],
                    "joint_shape": (17, 2),
                    "bias": False,
                    "embedding_size": 13,
                    "bbox_format": tv_tensors.BoundingBoxFormat.XYWH,
                },
                (8, 13),
            ),
            (
                32,
                {
                    "hidden_layers_kp": [15],
                    "hidden_layers_all": [],
                    "joint_shape": (21, 2),
                    "nof_kernels": 10,
                    "bias": True,
                    "embedding_size": 8,
                    "bbox_format": "xyxy",
                },
                (32, 8),
            ),
            (
                32,
                {
                    "hidden_layers_kp": [15, 11],
                    "hidden_layers_all": [15, 11],
                    "joint_shape": (21, 2),
                    "nof_kernels": 10,
                    "embedding_size": 4,
                    "bbox_format": "xyxy",
                },
                (32, 4),
            ),
        ]:
            with self.subTest(msg=f"params: {params}"):
                cfg = fill_in_defaults({"pose_embedding_generator": params, "batch_size": batch_size, "device": device})
                m = KeyPointConvolutionPBEG(config=cfg, path=["pose_embedding_generator"])
                kp = torch.rand((batch_size, *params["joint_shape"])).to(device=cfg["device"])
                bbox = tv_tensors.BoundingBoxes(torch.rand((batch_size, 4)), format="XYWH", canvas_size=(100, 100)).to(
                    device=device
                )
                res = m.forward(kp, bbox)
                self.assertEqual(res.device.type, device.type)
                self.assertEqual(res.shape, out_shape)


if __name__ == "__main__":
    unittest.main()
