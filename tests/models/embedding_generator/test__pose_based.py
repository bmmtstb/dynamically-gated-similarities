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
