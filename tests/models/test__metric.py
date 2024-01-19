import unittest
from unittest.mock import patch

from torch import nn

from dgs.models.metric import get_metric, get_metric_from_name, METRICS, register_metric


class TestMetrics(unittest.TestCase):
    def test_get_metric_from_name(self):
        for name, metric_class in METRICS.items():
            with self.subTest(msg=f"name: {name}, metric_class: {metric_class}"):
                metric = get_metric_from_name(name)
                self.assertEqual(metric, metric_class)
                self.assertTrue(issubclass(metric, nn.Module))

    def test_get_metric_from_name_exception(self):
        for name in ["", "undefined", "dummy", "metric"]:
            with self.subTest(msg=f"name: {name}"):
                with self.assertRaises(ValueError):
                    get_metric_from_name(name)

    def test_get_metric(self):
        for instance, result in [
            ("PairwiseDistance", nn.PairwiseDistance),
            (nn.PairwiseDistance, nn.PairwiseDistance),
        ]:
            with self.subTest(msg=f"instance: {instance}, result: {result}"):
                self.assertEqual(get_metric(instance), result)

    def test_get_metric_exception(self):
        for instance in [
            None,
            nn.Module(),
            {},
            1,
        ]:
            with self.subTest(msg="instance: {}".format(instance)):
                with self.assertRaises(ValueError):
                    get_metric(instance)

    def test_register_metric(self):
        with patch.dict(METRICS):
            for name, func, exception in [
                ("dummy", nn.PairwiseDistance, False),
                ("dummy", nn.PairwiseDistance, ValueError),
                ("new_dummy", nn.PairwiseDistance(), ValueError),
                ("new_dummy", nn.PairwiseDistance, False),
            ]:
                with self.subTest(msg="name: {}, func: {}, excpt: {}".format(name, func, exception)):
                    if exception is not False:
                        with self.assertRaises(exception):
                            register_metric(name, func)
                    else:
                        register_metric(name, func)
                        self.assertTrue("dummy" in METRICS)
        self.assertTrue("dummy" not in METRICS)
        self.assertTrue("new_dummy" not in METRICS)


if __name__ == "__main__":
    unittest.main()
