import os
import shutil
import unittest

import gdown
import torch as t
from torch import nn
from torch.nn.functional import softmax as f_softmax
from torchvision.tv_tensors import BoundingBoxes

from dgs.models.dgs.dgs import DGSModule
from dgs.utils.config import fill_in_defaults, load_config
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_abs_dir, mkdir_if_missing
from dgs.utils.state import State
from dgs.utils.utils import HidePrint
from helper import load_test_image, load_test_images

J = 17
PATH = ["dgs"]


class TestDGSModule(unittest.TestCase):

    def test_init(self):
        cfg = load_config("./tests/test_data/configs/test_config_dgs.yaml")

        with HidePrint():
            m = DGSModule(config=cfg, path=PATH)

        self.assertTrue(isinstance(m, DGSModule))
        self.assertTrue(isinstance(m.combine, nn.Module))

        self.assertEqual(len(m.params["names"]), 3)

    def test_init_variants(self):

        cfg = fill_in_defaults(
            {PATH[0]: {"new_track_weight": 0.43}},
            load_config("./tests/test_data/configs/test_config_dgs.yaml"),
        )

        with HidePrint():
            m = DGSModule(config=cfg, path=PATH)

        self.assertTrue(isinstance(m.new_track_weight, t.Tensor))
        self.assertTrue(t.allclose(m.new_track_weight, t.tensor(0.43)))

    def test_forward_equal_inputs(self):
        cfg = load_config("./tests/test_data/configs/test_config_dgs.yaml")

        img_name = "866-256x256.jpg"
        img_crop_path = ("./tests/test_data/images/" + img_name,)

        with HidePrint():
            m = DGSModule(config=cfg, path=PATH)

        ds_input = State(
            filepath=img_crop_path,
            bbox=BoundingBoxes([0, 0, 100, 100], canvas_size=(256, 256), format="XYXY"),
            keypoints=t.ones((J, 2)),
            joint_weight=t.ones((J, 1)),
            image_crop=load_test_image(img_name),
        )

        r = m.forward(ds_input, ds_input)

        self.assertEqual(list(r.shape), [1, 2])
        self.assertTrue(t.allclose(r, t.tensor([1.0, 0], dtype=t.float32)))

    def test_forward(self):
        cfg = load_config("./tests/test_data/configs/test_config_dgs.yaml")

        vis_mod, pose_mod, box_mod = cfg["combine_sims"]["alpha"]

        with HidePrint():
            m = DGSModule(config=cfg, path=["dgs"])

        for msg, ds_input, ds_target, out_values, inv_out in [
            (
                "image differs",
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((5, J, 2)),
                    joint_weight=t.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=("file_example_PNG_500kB.png", "866-256x256.jpg"),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(2, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((2, J, 2)),
                    joint_weight=t.ones((2, J, 1)),
                    validate=False,
                ),
                t.ones((5, 2)) * 0.5 * (pose_mod + box_mod) + t.tensor([0.0, 1.0]).repeat(5, 1) * vis_mod,
                0.2 * t.ones((2, 5)),
            ),
            (
                "bbox differs",
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((5, J, 2)),
                    joint_weight=t.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(2)),
                    bbox=BoundingBoxes(t.tensor([[0, 0, 5, 5], [1, 1, 6, 6]]), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((2, J, 2)),
                    joint_weight=t.ones((2, J, 1)),
                    validate=False,
                ),
                t.ones((5, 2)) * 0.5 * (vis_mod + pose_mod)
                + f_softmax(t.tensor([1.0, 16 / 34]).repeat(5, 1), dim=-1) * box_mod,
                0.2 * t.ones((2, 5)),
            ),
            (
                "pose differs",
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((5, J, 2)),
                    joint_weight=t.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(2)),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(2, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.stack([-1 * t.ones((1, J, 2)), t.ones(1, J, 2)]),
                    joint_weight=t.ones((2, J, 1)),
                    validate=False,
                ),
                t.ones((5, 2)) * 0.5 * (vis_mod + box_mod)
                + f_softmax(t.tensor([0.0, 1.0]).repeat(5, 1), dim=-1) * pose_mod,
                0.2 * t.ones((2, 5)),
            ),
            (
                "joint weight differs",
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((5, J, 2)),
                    joint_weight=t.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(2)),
                    bbox=BoundingBoxes(t.tensor([0, 0, 5, 5]).repeat(2, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=t.ones((2, J, 2)),
                    joint_weight=t.stack([t.zeros((1, J, 1)), t.ones(1, J, 1)]),
                    validate=False,
                ),
                t.ones((5, 2)) * 0.5 * (vis_mod + box_mod)
                + f_softmax(t.tensor([0.0, 1.0]).repeat(5, 1), dim=-1) * pose_mod,
                0.2 * t.ones((2, 5)),
            ),
        ]:
            target_shape = t.Size((len(ds_input), len(ds_target) + len(ds_input)))
            target_inv_shape = t.Size((len(ds_target), len(ds_input) + len(ds_target)))

            with self.subTest(msg="msg: {}".format(msg)):
                # make sure the image crops are loaded
                ds_input.image_crop = load_test_images(ds_input.filepath, force_reshape=True, output_size=(256, 256))
                ds_target.image_crop = load_test_images(ds_target.filepath, force_reshape=True, output_size=(256, 256))

                r = m.forward(ds_input, ds_target)
                r_inv = m.forward(ds_target, ds_input)

                self.assertEqual(r.shape, target_shape)
                self.assertEqual(r_inv.shape, target_inv_shape)

                # combined softmax is True
                out_values = f_softmax(out_values, dim=-1)
                # add zeros for empty states
                out_values = t.cat([out_values, t.zeros(len(ds_input), len(ds_input))], dim=-1)
                self.assertTrue(t.allclose(r, out_values, rtol=1e-3), (r[0], out_values[0]))

                inv_out = t.cat([inv_out, t.zeros(len(ds_target), len(ds_target))], dim=-1)
                self.assertTrue(t.allclose(r_inv, inv_out, rtol=1e-3), (r_inv, inv_out))

    def setUp(self):
        mkdir_if_missing(os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_dgs/"))
        weights_dir = os.path.join(PROJECT_ROOT, "./weights/")
        mkdir_if_missing(weights_dir)
        cached_file = os.path.join(weights_dir, "osnet_x0_25_imagenet.pth")

        if not os.path.exists(cached_file):
            gdown.download("https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs", cached_file, quiet=False)

    def tearDown(self):
        dir_path = os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_dgs/")
        if is_abs_dir(dir_path):
            shutil.rmtree(dir_path)


if __name__ == "__main__":
    unittest.main()
