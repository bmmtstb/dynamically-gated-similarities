import math
import unittest
import warnings
from unittest.mock import patch

import torch as t
from torch import nn
from torch.nn.functional import softmax as tf_softmax
from torchvision import tv_tensors as tv_te

from dgs.models.metric import get_metric, METRICS, register_metric
from dgs.models.metric.metric import (
    _validate_metric_inputs,
    compute_accuracy,
    compute_cmc,
    compute_near_k_accuracy,
    CosineDistanceMetric,
    CosineSimilarityMetric,
    EuclideanDistanceMetric,
    EuclideanSquareMetric,
    IOUDistance,
    NegativeSoftmaxEuclideanDistance,
    NegativeSoftmaxEuclideanSquaredDistance,
    PairwiseDistanceMetric,
    TorchreidCosineDistance,
    TorchreidEuclideanSquaredDistance,
)
from helper import test_multiple_devices

# catch cython warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Cython evaluation.*is unavailable", category=UserWarning)
    try:
        # If torchreid is installed using `./dependencies/torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.metrics import accuracy as torchreid_acc
    except ModuleNotFoundError:
        # if torchreid is installed using `pip install torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.reid.metrics import accuracy as torchreid_acc


class TestMetrics(unittest.TestCase):
    def test_get_metric(self):
        for instance, result in [
            ("CosineSimilarity", CosineSimilarityMetric),
            ("TorchPairwiseDistance", nn.PairwiseDistance),
            (nn.PairwiseDistance, nn.PairwiseDistance),
            ("PairwiseDistanceMetric", PairwiseDistanceMetric),
        ]:
            with self.subTest(msg=f"instance: {instance}, result: {result}"):
                self.assertEqual(get_metric(instance), result)

    def test_register_metric(self):
        with patch.dict(METRICS):
            for name, func, exception in [
                ("dummy", nn.PairwiseDistance, False),
                ("dummy", nn.PairwiseDistance, KeyError),
                ("new_dummy", nn.PairwiseDistance(), TypeError),
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

    def test_metrics_wrong_input_shape(self):
        for i1, i2, err in [
            (t.ones((1, 2)), t.ones((1, 1)), ValueError),
            (t.ones((1, 1, 2)), t.ones((1, 2)), ValueError),
            (t.ones((1, 2)), t.ones((1, 1, 2)), ValueError),
            (t.ones(2), t.ones(2), ValueError),
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
                dist = f(t.ones((a, E)), t.zeros(b, E))
                dist_inv = f(t.zeros((b, E)), t.ones(a, E))
                self.assertEqual(dist.shape, (a, b))
                self.assertTrue(t.allclose(dist, t.ones((a, b)) * E))
                self.assertTrue(t.allclose(dist, dist_inv.T))

    def test_euclid_sqr_dist_equals_torchreid(self):
        for a, b, E in [
            (2, 2, 7),
            (2, 4, 8),
            (5, 3, 6),
        ]:
            with self.subTest(msg="a: {}, b: {}, E: {}".format(a, b, E)):
                own_func = EuclideanSquareMetric()
                reid_func = TorchreidEuclideanSquaredDistance()

                i1 = t.rand((a, E))
                i2 = t.rand((b, E))

                own_dist = own_func(i1, i2)
                reid_dist = reid_func(i1, i2)

                self.assertTrue(t.allclose(own_dist, reid_dist))

    def test_euclid_dist(self):
        for a, b, E in [
            (2, 2, 7),
            (2, 4, 8),
            (5, 3, 6),
        ]:
            with self.subTest(msg="a: {}, b: {}, E: {}".format(a, b, E)):
                f = EuclideanDistanceMetric()
                dist = f(t.ones((a, E)), t.zeros(b, E))
                dist_inv = f(t.zeros((b, E)), t.ones(a, E))
                self.assertEqual(dist.shape, (a, b))
                self.assertTrue(t.allclose(dist, t.ones((a, b)) * math.sqrt(E)))
                self.assertTrue(t.allclose(dist, dist_inv.T))

    def test_cosine_distance(self):
        for t1, t2, res in [
            (t.ones((2, 4)), t.ones((3, 4)), t.zeros((2, 3))),
            (t.zeros((2, 4)), t.ones((3, 4)), t.ones((2, 3))),
            (t.ones((2, 4)), t.zeros((3, 4)), t.ones((2, 3))),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, res: {}".format(t1, t2, res)):
                f = CosineDistanceMetric()
                dist = f(t1.clone(), t2.clone())
                dist_inv = f(t2.clone(), t1.clone())
                self.assertEqual(dist.shape, res.shape)
                self.assertTrue(t.allclose(dist, res))
                self.assertTrue(t.allclose(dist, dist_inv.T))

    def test_cosine_dist_equals_torchreid(self):
        for a, b, E in [
            (2, 2, 7),
            (2, 4, 8),
            (5, 3, 6),
        ]:
            with self.subTest(msg="a: {}, b: {}, E: {}".format(a, b, E)):
                own_func = CosineDistanceMetric()
                reid_func = TorchreidCosineDistance()

                i1 = t.rand((a, E))
                i2 = t.rand((b, E))

                own_dist = own_func(i1, i2)
                reid_dist = reid_func(i1, i2)

                self.assertTrue(t.allclose(own_dist, reid_dist))

    def test_cosine_similarity(self):
        for t1, t2, res in [
            (t.ones((2, 4)), t.ones((3, 4)), t.ones((2, 3))),
            (t.zeros((2, 4)), t.ones((3, 4)), t.zeros((2, 3))),
            (t.ones((2, 4)), t.zeros((3, 4)), t.zeros((2, 3))),
            (t.ones((2, 4)), -1 * t.ones((3, 4)), -1 * t.ones((2, 3))),
            (-1 * t.ones((2, 4)), t.ones((3, 4)), -1 * t.ones((2, 3))),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, res: {}".format(t1, t2, res)):
                f = CosineSimilarityMetric()
                dist = f(t1.clone(), t2.clone())
                dist_inv = f(t2.clone(), t1.clone())
                self.assertEqual(dist.shape, res.shape)
                self.assertTrue(t.allclose(dist, res))
                self.assertTrue(t.allclose(dist, dist_inv.T))

    def test_neg_softmax_euclidean(self):
        for t1, t2, res, res_inv in [
            (t.ones((2, 4)), t.ones((3, 4)), 1 / 3 * t.ones((2, 3)), 0.5 * t.ones((3, 2))),
            (t.zeros((2, 4)), t.ones((3, 4)), 1 / 3 * t.ones((2, 3)), 0.5 * t.ones((3, 2))),
            (t.ones((2, 4)), t.zeros((3, 4)), 1 / 3 * t.ones((2, 3)), 0.5 * t.ones((3, 2))),
            (
                t.tensor([[1, 0], [1, 1], [0, 1]]),
                t.ones((1, 2)),
                t.ones((3, 1)),
                tf_softmax(t.tensor([[0, 1, 0]]).float(), dim=-1),
            ),
            (
                t.tensor([[1, 0], [1, 1], [0, 1]]),
                t.ones((4, 2)),
                tf_softmax(t.ones((3, 4)), dim=-1),
                tf_softmax(t.tensor([[0, 1, 0]]).repeat(4, 1).float(), dim=-1),
            ),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, res: {}".format(t1, t2, res)):
                m = NegativeSoftmaxEuclideanDistance()
                dist = m(t1.float(), t2.float())
                dist_inv = m(t2.float(), t1.float())
                self.assertEqual(dist.shape, res.shape)
                self.assertTrue(t.allclose(dist.float(), res.float()), (dist, res))
                self.assertTrue(t.allclose(dist_inv.float(), res_inv.float()), (dist_inv, res_inv))

    def test_neg_softmax_squared_euclidean(self):
        for t1, t2, res, res_inv in [
            (t.ones((2, 4)), t.ones((3, 4)), 1 / 3 * t.ones((2, 3)), 0.5 * t.ones((3, 2))),
            (t.zeros((2, 4)), t.ones((3, 4)), 1 / 3 * t.ones((2, 3)), 0.5 * t.ones((3, 2))),
            (t.ones((2, 4)), t.zeros((3, 4)), 1 / 3 * t.ones((2, 3)), 0.5 * t.ones((3, 2))),
            (
                t.tensor([[1, 0], [1, 1], [0, 1]]),
                t.ones((1, 2)),
                t.ones((3, 1)),
                tf_softmax(t.tensor([[0, 1, 0]]).float(), dim=-1),
            ),
            (
                t.tensor([[1, 0], [1, 1], [0, 1]]),
                t.ones((4, 2)),
                tf_softmax(t.ones((3, 4)), dim=-1),
                tf_softmax(t.tensor([[0, 1, 0]]).repeat(4, 1).float(), dim=-1),
            ),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, res: {}".format(t1, t2, res)):
                m = NegativeSoftmaxEuclideanSquaredDistance()
                sim = m(t1.float(), t2.float())
                sim_inv = m(t2.float(), t1.float())
                self.assertEqual(sim.shape, res.shape)
                self.assertTrue(t.allclose(sim, res), (sim, res))
                self.assertTrue(t.allclose(sim_inv, res_inv), (sim_inv, res_inv))

    def test_pairwise_distance(self):
        m = PairwiseDistanceMetric(p=2, eps=1e-4, keepdim=True)
        t1 = t.ones((4, 10))
        t2 = t.ones((4, 10))
        r = m(t1, t2)
        self.assertTrue(isinstance(r, t.Tensor))
        self.assertEqual(list(r.shape), [4, 1])
        self.assertEqual(m.dist.eps, 1e-4)

    def test_iou_distance(self):
        m = IOUDistance()
        b1 = tv_te.BoundingBoxes([0, 0, 5, 5], canvas_size=(10, 10), format="xyxy")
        b2 = tv_te.BoundingBoxes([1, 1, 5, 5], canvas_size=(10, 10), format="xywh")

        r = m(b1, b2)
        r_inv = m(b2, b1)

        self.assertTrue(isinstance(r, t.Tensor))
        self.assertTrue(isinstance(r_inv, t.Tensor))
        self.assertEqual(list(r.shape), [1, 1])
        self.assertEqual(list(r_inv.shape), [1, 1])
        self.assertTrue(t.allclose(r, t.tensor(1 - (16 / 34))))
        self.assertTrue(t.allclose(r_inv, t.tensor(1 - (16 / 34))))

        with self.assertRaises(TypeError) as e:
            _ = m(t.ones((1, 4)), b2)
        self.assertTrue("input1 should be an instance of tv_tensors.BoundingBoxes" in str(e.exception), msg=e.exception)

        with self.assertRaises(TypeError) as e:
            _ = m(b1, t.ones((1, 4)))
        self.assertTrue("input2 should be an instance of tv_tensors.BoundingBoxes" in str(e.exception), msg=e.exception)


class TestMetricCMC(unittest.TestCase):
    @test_multiple_devices
    def test_compute_cmc(self, device):
        for distmat, query, gallery, ranks, results in [
            (
                t.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                t.tensor([1, 4]).int(),
                t.tensor([2, 3, 4, 0, 1]).long(),
                [1, 2, 3, 4, 5, 6],
                [0.0, 0.5, 0.5, 0.5, 1.0, 1.0],
            ),
            (  # 2d label and no long
                t.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                t.tensor([[1], [4]]).long(),
                t.tensor([[2, 3, 4, 0, 1]]).float(),
                [1, 2, 3, 4, 5, 6],
                [0.0, 0.5, 0.5, 0.5, 1.0, 1.0],
            ),
            (  # gallery contains an ID multiple times
                t.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                t.tensor([2, 0]).long(),
                t.tensor([0, 1, 2, 2, 4]).long(),
                [1, 2, 3, 4, 5, 6],
                [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            ),
            (  # query contains an ID multiple times
                t.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                t.tensor([2, 2]).long(),
                t.tensor([0, 1, 3, 4, 2]).long(),
                [1, 2, 3, 4, 5, 6],
                [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            ),
            (  # query and gallery contain the same ID multiple times
                t.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                t.tensor([2, 2]).long(),
                t.tensor([2, 1, 3, 4, 2]).long(),
                [1, 2, 3, 4, 5, 6],
                [0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            ),
            (  # top rank only
                t.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.3]]),
                t.tensor([1, 2]).long(),
                t.tensor([1, 2, 3, 4, 0]).long(),
                [1],
                [1.0],
            ),
        ]:
            with self.subTest(
                msg="ranks: {}, results: {}, distmat: {}, query: {}, gallery: {}, device: {}".format(
                    ranks, results, distmat, query, gallery, device
                )
            ):
                distmat.to(device=device)
                query.to(device=device)
                gallery.to(device=device)

                with warnings.catch_warnings():  # will warn with rank 6, ignore it.
                    warnings.filterwarnings(
                        "ignore",
                        message="Number of gallery samples.*is smaller than the max rank.*Setting rank.",
                        category=UserWarning,
                    )
                    result_dict = {rank: float(result) for rank, result in zip(ranks, results)}
                    self.assertEqual(compute_cmc(distmat, query, gallery, ranks), result_dict)


class TestMetricAccuracy(unittest.TestCase):

    @test_multiple_devices
    def test_accuracy(self, device):
        topk = [1, 2, 3]
        prediction = t.tensor([[0.1, 0.2, 0.3, 0.4] for _ in range(4)], dtype=t.float32, device=device)
        target = t.tensor([0, 1, 2, 3], device=device).long()
        topk_accuracy = {1: 100 / 4, 2: 200 / 4, 3: 300 / 4}

        self.assertEqual(compute_accuracy(prediction=prediction, target=target, topk=topk), topk_accuracy)

    def test_accuracy_100_on_self(self):
        topk = [1, 2, 3]
        N = 512
        nof_classes = 10
        results = {1: 100.0, 2: 100.0, 3: 100.0}

        prediction = t.rand(size=(N, nof_classes), dtype=t.float32)
        target = t.argmax(prediction, dim=1).long()

        accs = compute_accuracy(prediction=prediction, target=target, topk=topk)
        self.assertEqual(accs, results)

    def test_compare_accuracy_own_vs_torchreid(self):
        N = 512
        nof_classes = 10
        for _ in range(5):
            prediction = t.rand(size=(N, nof_classes), dtype=t.float32)
            target = t.randint(low=1, high=nof_classes, size=(N,), dtype=t.long)
            own_acc = compute_accuracy(prediction=prediction, target=target, topk=None)[1]
            tr_acc = torchreid_acc(output=prediction, target=target, topk=(1,))[0].item()
            self.assertEqual(own_acc, tr_acc)


class TestMetricNearKAccuracy(unittest.TestCase):
    ks = [0, 1, 10, 20, 30]

    @test_multiple_devices
    def test_near_k_accuracy(self, device):
        pred = t.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=t.float32, device=device)
        target = t.tensor([[0.4], [0.4], [0.4], [0.4]], dtype=t.float32, device=device)
        res = {0: 1 / 4, 1: 1 / 4, 10: 2 / 4, 20: 3 / 4, 30: 4 / 4}
        accs = compute_near_k_accuracy(a_pred=pred, a_targ=target, ks=self.ks)
        self.assertDictEqual(accs, res)

    @test_multiple_devices
    def test_near_k_accuracy_with_reshaping(self, device):
        pred = t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32, device=device)
        target = t.tensor([0.4, 0.4, 0.4, 0.4], dtype=t.float32, device=device)
        res = {0: 1 / 4, 1: 1 / 4, 10: 2 / 4, 20: 3 / 4, 30: 4 / 4}
        accs = compute_near_k_accuracy(a_pred=pred, a_targ=target, ks=self.ks)
        self.assertDictEqual(accs, res)

    def test_near_k_accuracy_exceptions(self):
        with self.assertRaises(ValueError) as e:
            _ = compute_near_k_accuracy(a_pred=t.ones(1), a_targ=t.ones(4), ks=self.ks)
        self.assertTrue("alpha_pred and alpha_targ must have the same length" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = compute_near_k_accuracy(a_pred=t.ones(4), a_targ=t.ones(4), ks=[-1])
        self.assertTrue("ks must be positive" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = compute_near_k_accuracy(a_pred=t.ones((4, 2)), a_targ=t.ones((4, 1)), ks=[1])
        self.assertTrue("Alpha pred should be one dimensional" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = compute_near_k_accuracy(a_pred=t.ones((4, 1)), a_targ=t.ones((4, 2)), ks=[1])
        self.assertTrue("Alpha target should be one dimensional" in str(e.exception), msg=e.exception)

        with self.assertRaises(NotImplementedError) as e:
            _ = compute_near_k_accuracy(a_pred=t.ones((4, 1, 1)), a_targ=t.ones(4), ks=[1])


if __name__ == "__main__":
    unittest.main()
