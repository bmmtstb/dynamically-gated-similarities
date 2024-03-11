import os
import unittest

import torch
from torch import nn
from torchvision.tv_tensors import BoundingBoxes

from dgs.models.embedding_generator.torchreid import TorchreidEmbeddingGenerator
from dgs.utils.config import fill_in_defaults
from dgs.utils.state import State
from dgs.utils.utils import HidePrint
from helper import get_test_config, load_test_image, load_test_images

J = 17


class TestTorchreidEmbeddingGenerator(unittest.TestCase):

    def test_osnet_0_25_single_input(self):

        nof_classes = 2
        cfg = fill_in_defaults(
            {
                "is_training": False,
                "device": "cpu",
                "embed_gen": {"module_name": "torchreid", "model_name": "osnet_x0_25", "nof_classes": nof_classes},
            },
            get_test_config(),
        )

        with HidePrint():
            m = TorchreidEmbeddingGenerator(config=cfg, path=["embed_gen"])

        self.assertTrue(isinstance(m, nn.Module))
        self.assertFalse(m.training)
        self.assertTrue(isinstance(m.model, nn.Module))
        self.assertFalse(m.model.training)

        file_name = "866-256x256.jpg"
        fp = os.path.join("./tests/test_data/images/", file_name)
        img = load_test_image(file_name)
        ds = State(
            filepath=(fp,),
            validate=False,
            keypoints=torch.ones(1, J, 2),
            bbox=BoundingBoxes([0, 0, 1, 1], format="xyxy", canvas_size=(1, 1)),
            crop_path=(fp,),
            image_crop=img,
        )
        pred_embed = m.forward(ds)
        self.assertEqual(list(pred_embed.shape), [1, 512])
        pred_class_probs = m.model.classifier(pred_embed)
        self.assertEqual(list(pred_class_probs.shape), [1, 2])

    def test_osnet_0_25_batched_input(self):
        nof_classes = 2
        cfg = fill_in_defaults(
            {
                "is_training": False,
                "device": "cpu",
                "embed_gen": {"module_name": "torchreid", "model_name": "osnet_x0_25", "nof_classes": nof_classes},
            },
            get_test_config(),
        )

        with HidePrint():
            m = TorchreidEmbeddingGenerator(config=cfg, path=["embed_gen"])

        self.assertTrue(isinstance(m, nn.Module))
        self.assertFalse(m.training)
        self.assertTrue(isinstance(m.model, nn.Module))
        self.assertFalse(m.model.training)

        B = 3
        file_names = ["866-256x256.jpg" for _ in range(B)]
        fps = tuple(os.path.join("./tests/test_data/images/", fn) for fn in file_names)
        img = load_test_images(file_names)
        ds = State(
            filepath=fps,
            validate=False,
            keypoints=torch.ones(1, J, 2),
            bbox=BoundingBoxes(torch.ones((B, 4)), format="xyxy", canvas_size=(1, 1)),
            crop_path=fps,
            image_crop=img,
        )
        pred_embed = m.forward(ds)
        self.assertEqual(list(pred_embed.shape), [B, 512])
        pred_class_probs = m.model.classifier(pred_embed)
        self.assertEqual(list(pred_class_probs.shape), [B, nof_classes])


if __name__ == "__main__":
    unittest.main()
