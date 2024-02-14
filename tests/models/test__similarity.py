import unittest

import torch

from dgs.models.similarity import (
    CosineSimilarityModule,
    DotProductModule,
    EuclideanDistanceModule,
    PairwiseDistanceModule,
    PNormDistanceModule,
    SimilarityModule,
)
from dgs.utils.config import fill_in_defaults
from helper import get_default_config


class TestSimilarity(unittest.TestCase):
    default_cfg = get_default_config()

    def test_similarity(self):
        # fixme why are the shapes so chaotic?
        for module, kwargs, out_shape in [
            (CosineSimilarityModule, {}, (7, 2)),
            (DotProductModule, {}, (7, 17, 7, 17)),
            (EuclideanDistanceModule, {}, (7, 17, 17)),
            (PairwiseDistanceModule, {}, (7, 17)),
            (PNormDistanceModule, {"kwargs": {"p": 3}}, (7, 17, 17)),
        ]:
            with self.subTest(msg="module: {}, kwargs: {}".format(module, kwargs)):
                path = ["pose_similarity"]
                cfg = fill_in_defaults({"pose_similarity": kwargs}, self.default_cfg)

                m: SimilarityModule = module(config=cfg, path=path)

                res = m.forward(torch.ones((7, 17, 2)), torch.ones((7, 17, 2)))
                self.assertEqual(res.shape, torch.Size(out_shape))

    def test_oks(self):
        pass


if __name__ == "__main__":
    unittest.main()
