import unittest
from unittest.mock import patch

from torch import nn

from dgs.models.loss import get_loss_from_name, get_loss_function, LOSS_FUNCTIONS, register_loss_function


class Test(unittest.TestCase):
    def test_get_loss_from_name(self):
        for name, loss_class in LOSS_FUNCTIONS.items():
            with self.subTest(msg=f"name: {name}, loss_class: {loss_class}"):
                loss = get_loss_from_name(name)
                self.assertEqual(loss, loss_class)
                self.assertTrue(issubclass(loss, nn.Module))

    def test_get_loss_from_name_exception(self):
        for name in ["", "undefined", "dummy", "loss"]:
            with self.subTest(msg=f"name: {name}"):
                with self.assertRaises(ValueError):
                    get_loss_from_name(name)

    def test_get_loss_function(self):
        for instance, result in [
            ("L1Loss", nn.L1Loss),
            (nn.MSELoss, nn.MSELoss),
        ]:
            with self.subTest(msg=f"instance: {instance}, result: {result}"):
                self.assertEqual(get_loss_function(instance), result)

    def test_get_loss_function_exception(self):
        for instance in [
            None,
            nn.Module(),
            {},
            1,
        ]:
            with self.subTest(msg="instance: {}".format(instance)):
                with self.assertRaises(ValueError):
                    get_loss_function(instance)

    def test_register_loss_function(self):
        with patch.dict(LOSS_FUNCTIONS):
            for name, func, exception in [
                ("dummy", nn.MSELoss, False),
                ("dummy", nn.MSELoss, ValueError),
                ("new_dummy", nn.MSELoss(), ValueError),
                ("new_dummy", nn.MSELoss, False),
            ]:
                with self.subTest(msg="name: {}, func: {}, excpt: {}".format(name, func, exception)):
                    if exception is not False:
                        with self.assertRaises(exception):
                            register_loss_function(name, func)
                    else:
                        register_loss_function(name, func)
                        self.assertTrue("dummy" in LOSS_FUNCTIONS)
        self.assertTrue("dummy" not in LOSS_FUNCTIONS)
        self.assertTrue("new_dummy" not in LOSS_FUNCTIONS)


if __name__ == "__main__":
    unittest.main()
