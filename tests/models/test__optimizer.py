import unittest
from unittest.mock import patch

import torch
from torch import optim

from dgs.models.optimizer import get_optimizer, OPTIMIZERS, register_optimizer


class TestOptimizer(unittest.TestCase):
    def test_get_optim_function(self):
        for instance, result in [
            ("Adagrad", optim.Adagrad),
            (optim.Adam, optim.Adam),
        ]:
            with self.subTest(msg=f"instance: {instance}, result: {result}"):
                self.assertEqual(get_optimizer(instance), result)

    def test_get_optim_function_exception(self):
        for instance in [None, {}, 1, "", optim.Optimizer([{"params": torch.zeros((8, 32))}], {})]:
            with self.subTest(msg="instance: {}".format(instance)):
                with self.assertRaises(ValueError):
                    get_optimizer(instance)

    def test_register_optim(self):
        with patch.dict(OPTIMIZERS):
            for name, optimizer, exception in [
                ("dummy", optim.Adam, False),
                ("dummy", optim.Adam, ValueError),
                ("new_dummy", optim.Adam([{"params": torch.zeros((8, 32))}]), ValueError),
                ("new_dummy", optim.Adam, False),
            ]:
                with self.subTest(msg="name: {}, optimizer: {}, excpt: {}".format(name, optimizer, exception)):
                    if exception is not False:
                        with self.assertRaises(exception):
                            register_optimizer(name, optimizer)
                    else:
                        register_optimizer(name, optimizer)
                        self.assertTrue("dummy" in OPTIMIZERS)
        self.assertTrue("dummy" not in OPTIMIZERS)
        self.assertTrue("new_dummy" not in OPTIMIZERS)


if __name__ == "__main__":
    unittest.main()
