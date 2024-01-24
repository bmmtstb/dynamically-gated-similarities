import math
import unittest
import warnings
from unittest.mock import patch

import torch
from torch import nn

from dgs.models.metric import (
    _validate_metric_inputs,
    compute_cmc,
    CosineDistanceMetric,
    CosineSimilarityMetric,
    EuclideanDistanceMetric,
    EuclideanSquareMetric,
    get_metric,
    get_metric_from_name,
    METRICS,
    register_metric,
)


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
            ("CosineSimilarity", CosineSimilarityMetric),
            ("TorchPairwiseDistance", nn.PairwiseDistance),
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

    def test_compute_cmc(self):
        for distmat, labels, predictions, ranks, results in [
            (
                torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                torch.tensor([1, 4]).int(),
                torch.tensor([2, 3, 4, 0, 1]).int(),
                [1, 2, 3, 4, 5, 6],
                [0.0, 0.5, 0.5, 0.5, 1.0, 1.0],
            ),
        ]:
            with self.subTest(
                msg="ranks: {}, results: {}, distmat: {}, labels: {}, predictions: {}".format(
                    ranks, results, distmat, labels, predictions
                )
            ):
                with warnings.catch_warnings():  # will warn with rank 6, ignore it.
                    warnings.filterwarnings(
                        "ignore",
                        message="Number of gallery samples.*is smaller than the max rank.*Setting rank.",
                        category=UserWarning,
                    )
                    result_dict = {rank: float(result) for rank, result in zip(ranks, results)}
                    self.assertEqual(compute_cmc(distmat, labels, predictions, ranks), result_dict)

    def test_metrics_wrong_input_shape(self):
        for i1, i2, err in [
            (torch.ones((1, 2)), torch.ones((1, 1)), ValueError),
            (torch.ones((1, 1, 2)), torch.ones((1, 2)), ValueError),
            (torch.ones((1, 2)), torch.ones((1, 1, 2)), ValueError),
            (torch.ones(2), torch.ones(2), ValueError),
        ]:
            with self.subTest(msg="i1: {}, i2: {}, err: {}".format(i1, i2, err)):
                with self.assertRaises(err):
                    _validate_metric_inputs(i1, i2)

    def test_euclid_sqr_dist(self):
        for a, b, E in [
            (2, 2, 7),
            (2, 4, 8),
            (5, 3, 6),
        ]:
            with self.subTest(msg="a: {}, b: {}, E: {}".format(a, b, E)):
                f = EuclideanSquareMetric()
                dist = f(torch.ones((a, E)), torch.zeros(b, E))
                dist_inv = f(torch.zeros((b, E)), torch.ones(a, E))
                self.assertEqual(dist.shape, (a, b))
                self.assertTrue(torch.allclose(dist, torch.ones((a, b)) * E))
                self.assertTrue(torch.allclose(dist, dist_inv.T))

    def test_euclid_dist(self):
        for a, b, E in [
            (2, 2, 7),
            (2, 4, 8),
            (5, 3, 6),
        ]:
            with self.subTest(msg="a: {}, b: {}, E: {}".format(a, b, E)):
                f = EuclideanDistanceMetric()
                dist = f(torch.ones((a, E)), torch.zeros(b, E))
                dist_inv = f(torch.zeros((b, E)), torch.ones(a, E))
                self.assertEqual(dist.shape, (a, b))
                self.assertTrue(torch.allclose(dist, torch.ones((a, b)) * math.sqrt(E)))
                self.assertTrue(torch.allclose(dist, dist_inv.T))

    @torch.no_grad()
    def test_cosine_distance(self):
        for t1, t2, res in [
            (torch.ones((2, 4)), torch.ones((3, 4)), torch.zeros((2, 3))),
            (torch.zeros((2, 4)), torch.ones((3, 4)), torch.ones((2, 3))),
            (torch.ones((2, 4)), torch.zeros((3, 4)), torch.ones((2, 3))),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, res: {}".format(t1, t2, res)):
                f = CosineDistanceMetric()
                dist = f(t1.clone(), t2.clone())
                dist_inv = f(t2.clone(), t1.clone())
                self.assertEqual(dist.shape, res.shape)
                self.assertTrue(torch.allclose(dist, res))
                self.assertTrue(torch.allclose(dist, dist_inv.T))

    def test_cosine_similarity(self):
        for t1, t2, res in [
            (torch.ones((2, 4)), torch.ones((3, 4)), torch.ones((2, 3))),
            (torch.zeros((2, 4)), torch.ones((3, 4)), torch.zeros((2, 3))),
            (torch.ones((2, 4)), torch.zeros((3, 4)), torch.zeros((2, 3))),
            (torch.ones((2, 4)), -1 * torch.ones((3, 4)), -1 * torch.ones((2, 3))),
            (-1 * torch.ones((2, 4)), torch.ones((3, 4)), -1 * torch.ones((2, 3))),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, res: {}".format(t1, t2, res)):
                f = CosineSimilarityMetric()
                dist = f(t1.clone(), t2.clone())
                dist_inv = f(t2.clone(), t1.clone())
                self.assertEqual(dist.shape, res.shape)
                self.assertTrue(torch.allclose(dist, res))
                self.assertTrue(torch.allclose(dist, dist_inv.T))


if __name__ == "__main__":
    unittest.main()
