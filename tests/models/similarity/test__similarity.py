import unittest
from unittest.mock import patch

from dgs.models.module import BaseModule
from dgs.models.similarity import get_similarity_module, register_similarity_module, SIMILARITIES
from dgs.models.similarity.similarity import SimilarityModule
from dgs.models.similarity.torchreid import TorchreidSimilarity
from dgs.utils.config import fill_in_defaults
from dgs.utils.utils import HidePrint
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
                kwargs["module_name"] = name
                cfg = fill_in_defaults({"sim": kwargs}, default_cfg=test_config)

                with HidePrint():
                    module = module(config=cfg, path=["sim"])

                self.assertTrue(isinstance(module, SimilarityModule))
                self.assertTrue(isinstance(module, BaseModule))

        with self.assertRaises(KeyError) as e:
            _ = get_similarity_module("dummy")
        self.assertTrue("Instance 'dummy' is not defined in" in str(e.exception), msg=e.exception)

    def test_register_similarity(self):
        with patch.dict(SIMILARITIES):
            for name, func, exception in [
                ("dummy", TorchreidSimilarity, False),
                ("dummy", TorchreidSimilarity, KeyError),
                ("new_dummy", TorchreidSimilarity, False),
            ]:
                with self.subTest(msg="name: {}, func: {}, except: {}".format(name, func, exception)):
                    if exception is not False:
                        with self.assertRaises(exception):
                            register_similarity_module(name, func)
                    else:
                        register_similarity_module(name, func)
                        self.assertTrue("dummy" in SIMILARITIES)
        self.assertTrue("dummy" not in SIMILARITIES)
        self.assertTrue("new_dummy" not in SIMILARITIES)


if __name__ == "__main__":
    unittest.main()
