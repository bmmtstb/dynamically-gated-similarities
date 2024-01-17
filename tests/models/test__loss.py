import unittest

from torch import nn

from dgs.models.loss import get_loss_from_name, LOSS_FUNCTIONS


class Test(unittest.TestCase):
    def test_get_loss_from_name(self):
        for name, loss_class in LOSS_FUNCTIONS.items():
            with self.subTest(msg=f"name: {name}, loss_class: {loss_class}"):
                loss = get_loss_from_name(name)
                self.assertEqual(loss, loss_class)
                self.assertTrue(isinstance(loss(), nn.Module))

    def test_get_loss_from_name_exception(self):
        for name in ["", "undefined", "dummy", "loss"]:
            with self.subTest(msg=f"name: {name}"):
                with self.assertRaises(ValueError):
                    get_loss_from_name(name)


if __name__ == "__main__":
    unittest.main()
