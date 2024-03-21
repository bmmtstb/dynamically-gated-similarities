import unittest
from unittest.mock import patch

from torch.nn import Module as TorchModule
from torch.optim import Adam
from torch.utils.data import DataLoader

from dgs.models.combine import COMBINE_MODULES
from dgs.models.combine.combine import CombineSimilaritiesModule
from dgs.models.dataset import DATASETS
from dgs.models.dataset.posetrack21 import PoseTrack21_BBox
from dgs.models.dgs import DGS_MODULES
from dgs.models.dgs.dgs import DGSModule
from dgs.models.embedding_generator import EMBEDDING_GENERATORS
from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.models.engine import ENGINES
from dgs.models.engine.engine import EngineModule
from dgs.models.loader import module_loader, register_module
from dgs.models.loss import LOSS_FUNCTIONS
from dgs.models.loss.loss import CrossEntropyLoss
from dgs.models.metric import METRICS
from dgs.models.metric.metric import CosineSimilarityMetric
from dgs.models.optimizer import OPTIMIZERS
from dgs.models.similarity import SIMILARITIES
from dgs.models.similarity.similarity import SimilarityModule
from dgs.utils.config import load_config
from dgs.utils.exceptions import InvalidConfigException
from dgs.utils.utils import HidePrint


class TestLoader(unittest.TestCase):

    def test_module_loader(self):
        cfg = load_config("./tests/test_data/configs/test_config.yaml")

        for mod_class, key, class_inst in [
            ("combine", "combine_sims", CombineSimilaritiesModule),
            ("dgs", "dgs", DGSModule),
            ("dataloader", "dataloader", DataLoader),
            ("embedding_generator", "vis_emb_gen", EmbeddingGeneratorModule),
            ("similarity", "box_similarity", SimilarityModule),
            ("similarity", "pose_similarity", SimilarityModule),
        ]:
            with self.subTest(msg="mod_class: {}, key: {}, class_inst: {}".format(mod_class, key, class_inst)):
                with HidePrint():
                    m = module_loader(config=cfg, module_class=mod_class, key=key)
                self.assertTrue(isinstance(m, class_inst))

    def test_module_loader_exceptions(self):
        cfg = load_config("./tests/test_data/configs/test_config.yaml")
        with self.assertRaises(InvalidConfigException) as e:
            _ = module_loader(config=cfg, module_class="dummy", key="invalid")
        self.assertTrue(
            "Module at path '['invalid']' does not contain a module name" in str(e.exception), msg=e.exception
        )

    def test_load_engine(self):
        cfg = load_config("./tests/test_data/configs/test_config.yaml")
        with HidePrint():
            dl = module_loader(config=cfg, module_class="dataloader", key="dataloader")
        kwargs = {"model": TorchModule(), "test_loader": dl, "val_loader": dl}
        with HidePrint():
            m = module_loader(config=cfg, module_class="engine", key="engine", **kwargs)
        self.assertTrue(isinstance(m, EngineModule))

    def test_register_module(self):
        name = "test_module"
        for mod_dict, cls_name, cls_inst in [
            (COMBINE_MODULES, "combine", CombineSimilaritiesModule),
            (DGS_MODULES, "dgs", DGSModule),
            (EMBEDDING_GENERATORS, "embedding_generator", EmbeddingGeneratorModule),
            (ENGINES, "engine", EngineModule),
            (LOSS_FUNCTIONS, "loss", CrossEntropyLoss),
            (METRICS, "metric", CosineSimilarityMetric),
            (OPTIMIZERS, "optimizer", Adam),
            (SIMILARITIES, "similarity", SimilarityModule),
            (DATASETS, "dataset", PoseTrack21_BBox),
        ]:
            with self.subTest(msg="modules: {}, cls_name: {}, cls_inst: {}".format(mod_dict, cls_name, cls_inst)):
                with patch.dict(mod_dict):
                    self.assertFalse(name in mod_dict)
                    register_module(name=name, new_module=cls_inst, inst_class_name=cls_name)
                    self.assertTrue(name in mod_dict)
                # not in original
                self.assertFalse(name in mod_dict)

    def test_register_module_raises_error_on_unknown_cls_name(self):
        with self.assertRaises(ValueError) as e:
            register_module(name="faulty", new_module=CombineSimilaritiesModule, inst_class_name="dummy")
        self.assertTrue("The instance class name 'dummy' could not be found." in str(e.exception), msg=e.exception)


if __name__ == "__main__":
    unittest.main()
