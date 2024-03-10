import unittest

import torch
from torch import nn
from torch.nn.functional import softmax as f_softmax
from torchvision.tv_tensors import BoundingBoxes

from dgs.models.dgs.dgs import DGSModule
from dgs.utils.config import fill_in_defaults, load_config
from dgs.utils.state import State
from dgs.utils.utils import HidePrint
from helper import load_test_image, load_test_images

J = 17
PATH = ["dgs"]


class TestDGSModule(unittest.TestCase):

    def test_init(self):
        cfg = load_config("./tests/test_data/test_config_dgs.yaml")

        with HidePrint():
            m = DGSModule(config=cfg, path=PATH)

        self.assertTrue(isinstance(m, DGSModule))
        self.assertTrue(isinstance(m.combine, nn.Module))

        self.assertEqual(len(m.params["names"]), 3)

    def test_init_variants(self):

        cfg = fill_in_defaults(
            {PATH[0]: {"similarity_softmax": True, "combined_softmax": True}},
            load_config("./tests/test_data/test_config_dgs.yaml"),
        )

        with HidePrint():
            m = DGSModule(config=cfg, path=PATH)

        self.assertTrue(len(m.combined_softmax) == 1)
        self.assertTrue(len(m.similarity_softmax) == 1)

    def test_forward_equal_inputs(self):
        cfg = load_config("./tests/test_data/test_config_dgs.yaml")

        img_name = "866-256x256.jpg"
        img_crop_path = ("./tests/test_data/" + img_name,)

        with HidePrint():
            m = DGSModule(config=cfg, path=PATH)

        ds_input = State(
            filepath=img_crop_path,
            bbox=BoundingBoxes([0, 0, 100, 100], canvas_size=(256, 256), format="XYXY"),
            keypoints=torch.ones((J, 2)),
            joint_weight=torch.ones((J, 1)),
            image_crop=load_test_image(img_name),
        )

        r = m.forward(ds_input, ds_input)

        self.assertEqual(list(r.shape), [1, 2])
        self.assertTrue(torch.allclose(r, torch.tensor([1.0, 0], dtype=torch.float32)))

    def test_forward(self):
        cfg = load_config("./tests/test_data/test_config_dgs.yaml")

        vis_mod, pose_mod, box_mod = cfg["combine_sims"]["alpha"]

        with HidePrint():
            m = DGSModule(config=cfg, path=["dgs"])

        for ds_input, ds_target, out_values, inv_out in [
            (  # image differs
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((5, J, 2)),
                    joint_weight=torch.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=("file_example_PNG_500kB.png", "866-256x256.jpg"),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(2, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((2, J, 2)),
                    joint_weight=torch.ones((2, J, 1)),
                    validate=False,
                ),
                torch.ones((5, 2)) * 0.5 * (pose_mod + box_mod) + torch.tensor([0.0, 1.0]).repeat(5, 1) * vis_mod,
                0.2 * torch.ones((2, 5)),
            ),
            (  # bbox differs
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((5, J, 2)),
                    joint_weight=torch.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(2)),
                    bbox=BoundingBoxes(torch.tensor([[0, 0, 5, 5], [1, 1, 6, 6]]), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((2, J, 2)),
                    joint_weight=torch.ones((2, J, 1)),
                    validate=False,
                ),
                torch.ones((5, 2)) * 0.5 * (vis_mod + pose_mod)
                + f_softmax(torch.tensor([1.0, 16 / 34]).repeat(5, 1), dim=-1) * box_mod,
                0.2 * torch.ones((2, 5)),
            ),
            (  # pose differs
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((5, J, 2)),
                    joint_weight=torch.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(2)),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(2, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.stack([-1 * torch.ones((1, J, 2)), torch.ones(1, J, 2)]),
                    joint_weight=torch.ones((2, J, 1)),
                    validate=False,
                ),
                torch.ones((5, 2)) * 0.5 * (vis_mod + box_mod)
                + f_softmax(torch.tensor([0.0, 1.0]).repeat(5, 1), dim=-1) * pose_mod,
                0.2 * torch.ones((2, 5)),
            ),
            (  # joint weight differs
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(5)),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(5, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((5, J, 2)),
                    joint_weight=torch.ones((5, J, 1)),
                    validate=False,
                ),
                State(
                    filepath=tuple("866-256x256.jpg" for _ in range(2)),
                    bbox=BoundingBoxes(torch.tensor([0, 0, 5, 5]).repeat(2, 1), canvas_size=(10, 10), format="XYXY"),
                    keypoints=torch.ones((2, J, 2)),
                    joint_weight=torch.stack([torch.zeros((1, J, 1)), torch.ones(1, J, 1)]),
                    validate=False,
                ),
                torch.ones((5, 2)) * 0.5 * (vis_mod + box_mod)
                + f_softmax(torch.tensor([0.0, 1.0]).repeat(5, 1), dim=-1) * pose_mod,
                0.2 * torch.ones((2, 5)),
            ),
        ]:
            target_shape = torch.Size((len(ds_input), len(ds_target) + 1))
            target_inv_shape = torch.Size((len(ds_target), len(ds_input) + 1))

            with self.subTest(
                msg="ds_input: {}, ds_target: {}, out_values: {}".format(
                    ds_input.filepath[0], ds_target.filepath[0], out_values
                )
            ):
                # make sure the image crops are loaded
                ds_input.image_crop = load_test_images(ds_input.filepath, force_reshape=True, output_size=(256, 256))
                ds_target.image_crop = load_test_images(ds_target.filepath, force_reshape=True, output_size=(256, 256))

                r = m.forward(ds_input, ds_target)
                r_inv = m.forward(ds_target, ds_input)

                self.assertEqual(r.shape, target_shape)
                self.assertEqual(r_inv.shape, target_inv_shape)

                out_values = torch.cat([out_values, torch.zeros(len(ds_input), 1)], dim=-1)
                self.assertTrue(torch.allclose(r, out_values, rtol=1e-3), (r, out_values))

                inv_out = torch.cat([inv_out, torch.zeros(len(ds_target), 1)], dim=-1)
                self.assertTrue(torch.allclose(r_inv, inv_out, rtol=1e-3), (r_inv, inv_out))


if __name__ == "__main__":
    unittest.main()
