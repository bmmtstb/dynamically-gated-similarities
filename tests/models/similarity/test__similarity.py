import unittest

from dgs.models import BaseModule
from dgs.models.similarity import get_similarity_module, SimilarityModule, TorchreidSimilarity
from dgs.utils.config import fill_in_defaults
from dgs.utils.exceptions import InvalidParameterException
from helper import get_test_config


class TestSimilarity(unittest.TestCase):

    def test_get_similarity(self):
        test_config = get_test_config()
        for name, mod_class, kwargs in [
            (
                "torchreid",
                TorchreidSimilarity,
                {"model_name": "osnet_x0_25", "similarity": "EuclideanDistance", "embedding_size": 3, "nof_classes": 2},
            ),
        ]:
            with self.subTest(msg="name: {}, mod_class: {}, kwargs: {}".format(name, mod_class, kwargs)):
                module = get_similarity_module(name)
                self.assertEqual(module, mod_class)

                cfg = fill_in_defaults({"sim": kwargs}, default_cfg=test_config)

                module = module(config=cfg, path=["sim"])
                self.assertTrue(isinstance(module, SimilarityModule))
                self.assertTrue(isinstance(module, BaseModule))

        with self.assertRaises(InvalidParameterException) as e:
            _ = get_similarity_module("dummy")
        self.assertTrue("Unknown similarity with name" in str(e.exception), msg=e.exception)


if __name__ == "__main__":
    unittest.main()