import unittest

import torch as t

from dgs.models.alpha.combined import SequentialCombinedAlpha
from dgs.utils.config import insert_into_config
from dgs.utils.state import State
from helper import get_test_config
from utils.state import B, DUMMY_BBOX, DUMMY_BBOX_BATCH, DUMMY_KP, DUMMY_KP_BATCH


class TestSequentialCombinedAlphaModule(unittest.TestCase):

    def setUp(self):
        self.sequential_name = "sequential"
        self.first_path = ["alpha"]
        self.second_path = ["small"]
        self.conv_path = "pose_coco_conv1o15k2fc1"
        self.default_cfg = get_test_config().copy()
        self.default_cfg[self.first_path[0]] = {
            "module_name": "FullyConnectedAlpha",
            "name": "bbox",
            "hidden_layers": [4, 2],
            "bias": False,
        }
        self.default_cfg[self.second_path[0]] = {
            "module_name": "FullyConnectedAlpha",
            "hidden_layers": [2, 1],
            "bias": False,
        }
        self.default_cfg[self.sequential_name] = {
            "module_name": "SequentialCombinedAlpha",
            "name": "bbox",
            "paths": ["Identity", self.first_path, self.second_path],
        }

        self.default_cfg[self.conv_path] = {
            "module_name": "SequentialCombinedAlpha",
            "name": "keypoints",
            "paths": [
                {"Conv1d": {"in_channels": 17, "out_channels": 15, "kernel_size": 2, "groups": 1, "bias": True}},
                "Flatten",
                [self.conv_path, "fc"],
            ],
            "fc": {
                "module_name": "FullyConnectedAlpha",
                "hidden_layers": [15, 1],
                "bias": False,
            },
        }
        self.m = SequentialCombinedAlpha(path=[self.sequential_name], config=self.default_cfg)
        self.keypoint_model = SequentialCombinedAlpha(path=[self.conv_path], config=self.default_cfg)

    def tearDown(self):
        del self.default_cfg

    def test_init(self):
        for m, length in [
            (self.m, 3),
            (self.keypoint_model, 3),
        ]:
            with self.subTest(msg="m: {}, l: {}".format(m, length)):
                self.assertTrue(isinstance(m, SequentialCombinedAlpha))
                self.assertTrue(hasattr(m, "model"))
                self.assertTrue(isinstance(m.model, t.nn.Sequential))
                self.assertEqual(len(m.model), length)

    def test_init_raises(self):
        for cfg, exception, msg in [
            (
                {"module_name": "SequentialCombinedAlpha", "name": "bbox", "paths": ["Dummy"]},
                AttributeError,
                "Tried to load non-existent torch module",
            ),
            (
                {"module_name": "SequentialCombinedAlpha", "name": "bbox", "paths": [{"ReLU": None, "Flatten": None}]},
                ValueError,
                "Expected submodule config to be a single dict",
            ),
            (
                {"module_name": "SequentialCombinedAlpha", "name": "bbox", "paths": [{"Dummy": {}}]},
                AttributeError,
                "Tried to load non-existent torch module",
            ),
        ]:
            with self.subTest(msg="cfg: {}, exception: {}, msg: {}".format(cfg, exception, msg)):
                cfg = insert_into_config(path=["k"], value=cfg, original=self.default_cfg, copy=True)
                with self.assertRaises(exception) as e:
                    _ = SequentialCombinedAlpha(path=["k"], config=cfg)
                self.assertTrue(msg in str(e.exception), msg=e.exception)

    def test_forward(self):
        for m, data, res_shape in [
            (self.m, {"bbox": DUMMY_BBOX}, (1, 1)),
            (self.m, {"bbox": DUMMY_BBOX_BATCH}, (B, 1)),
            (self.keypoint_model, {"bbox": DUMMY_BBOX, "keypoints": DUMMY_KP}, (1, 1)),
            (self.keypoint_model, {"bbox": DUMMY_BBOX_BATCH, "keypoints": DUMMY_KP_BATCH}, (B, 1)),
        ]:
            with self.subTest(msg="data: {}, res_shape: {}".format(data, res_shape)):
                s = State(**data)
                out = m.forward(s)
                self.assertEqual(out.shape, t.Size(res_shape))


if __name__ == "__main__":
    unittest.main()
