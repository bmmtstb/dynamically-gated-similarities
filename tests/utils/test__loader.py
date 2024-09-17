import unittest
from copy import deepcopy
from unittest.mock import patch

from dgs.models.combine import COMBINE_MODULES
from dgs.models.dataset import DATASETS
from dgs.models.dgs import DGS_MODULES
from dgs.models.embedding_generator import EMBEDDING_GENERATORS
from dgs.models.engine import ENGINES
from dgs.models.loss import LOSS_FUNCTIONS
from dgs.models.metric import METRICS
from dgs.models.optimizer import OPTIMIZERS
from dgs.models.similarity import SIMILARITIES
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.loader import (
    get_instance,
    get_instance_from_name,
    get_registered_class_names,
    get_registered_class_types,
    get_registered_classes,
    register_instance,
)


class TestInstance(object): ...


class CallableInstance(object):
    def __call__(self, *_args, **_kwargs) -> None: ...


class OtherInstance(dict): ...


INSTANCES = {"instance": TestInstance}

TEST_MODULES: dict[str, dict[str, type]] = {
    "combine": COMBINE_MODULES,
    "dgs": DGS_MODULES,
    "embedding_generator": EMBEDDING_GENERATORS,
    "engine": ENGINES,
    "loss": LOSS_FUNCTIONS,
    "metric": METRICS,
    "optimizer": OPTIMIZERS,
    "similarity": SIMILARITIES,
    "dataset": DATASETS,
}


class TestLoader(unittest.TestCase):

    def test_register_instance(self):
        i = {}

        register_instance(name="dummy", instance=TestInstance, instances=i, inst_class=object, call=False)
        self.assertTrue("dummy" in i)

        with self.assertRaises(KeyError) as e:
            register_instance(name="dummy", instance=CallableInstance, instances=i, inst_class=object)
        self.assertTrue("already exists within the registered instances" in str(e.exception), msg=e.exception)

        register_instance(name="callable", instance=CallableInstance, instances=i, inst_class=object)
        self.assertTrue("callable" in i)

        with self.assertRaises(TypeError) as e:
            register_instance(name="not_callable", instance=[], instances=i, inst_class=object)
        self.assertTrue("The given instance is not callable." in str(e.exception), msg=e.exception)

        register_instance(name="dict", instance=OtherInstance, instances=i, inst_class=dict)

        with self.assertRaises(TypeError) as e:
            register_instance(name="other", instance=TestInstance, instances=i, inst_class=dict, call=False)
        self.assertTrue("The given instance is not a valid subclass of type" in str(e.exception), msg=e.exception)

    def test_get_instance_from_name(self):
        instances = deepcopy(INSTANCES)

        i = get_instance_from_name("instance", instances)
        self.assertTrue(isinstance(i, object))
        self.assertEqual(i, TestInstance)

        with self.assertRaises(KeyError) as e:
            _ = get_instance_from_name("dummy", instances)
        self.assertTrue("Instance 'dummy' is not defined in" in str(e.exception), msg=e.exception)

    def test_get_instance(self):
        instances = deepcopy(INSTANCES)

        i = get_instance("instance", instances, inst_class=object)
        self.assertTrue(isinstance(i, object))
        self.assertEqual(i, TestInstance)

        i = get_instance(CallableInstance, instances, inst_class=object)
        self.assertTrue(isinstance(i, object))
        self.assertEqual(i, CallableInstance)

        with self.assertRaises(InvalidParameterException) as e:
            _ = get_instance(TestInstance, instances, inst_class=dict)
        self.assertTrue("is neither string nor a subclass of" in str(e.exception), msg=e.exception)

    def test_get_registered_classes(self):
        for mod_type, mod_dict in TEST_MODULES.items():
            with self.subTest(msg="mod_type: {}, mod_dict: {}".format(mod_type, mod_dict)):
                with patch.dict(mod_dict):
                    classes = get_registered_classes(module_type=mod_type)
                    self.assertTrue(isinstance(classes, dict))
                    self.assertDictEqual(classes, mod_dict)

    def test_get_registered_class_names(self):
        for mod_type, mod_dict in TEST_MODULES.items():
            with self.subTest(msg="mod_type: {}, mod_dict: {}".format(mod_type, mod_dict)):
                with patch.dict(mod_dict):
                    names = get_registered_class_names(module_type=mod_type)
                    self.assertTrue(isinstance(names, set))
                    self.assertSetEqual(names, set(mod_dict.keys()))

    def test_get_registered_class_types(self):
        for mod_type, mod_dict in TEST_MODULES.items():
            with self.subTest(msg="mod_type: {}, mod_dict: {}".format(mod_type, mod_dict)):
                with patch.dict(mod_dict):
                    types = get_registered_class_types(module_type=mod_type)
                    self.assertTrue(isinstance(types, set))
                    self.assertSetEqual(types, set(mod_dict.values()))


if __name__ == "__main__":
    unittest.main()
