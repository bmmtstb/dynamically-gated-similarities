import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment


class TestDGSEngine(unittest.TestCase):
    def test_matching(self):
        N = 5
        T = 3
        similarity_matrix = np.concatenate([np.random.rand(N, T), np.zeros((N, N))], axis=1)
        rids, cids = linear_sum_assignment(similarity_matrix, maximize=True)  # rids and cids are ndarray of shape [N]
        self.assertTrue(0 <= similarity_matrix[rids, cids].sum() <= N)
        self.assertEqual(len(rids), N)
        self.assertEqual(len(cids), N)
        self.assertEqual(sum(cid >= T for cid in cids), N - T)


if __name__ == "__main__":
    unittest.main()
