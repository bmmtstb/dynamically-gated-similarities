import unittest

from torch import nn

from dgs.utils.nn import fc_linear, set_up_hidden_layer_sizes


class TestFullyConnected(unittest.TestCase):
    def test_set_up_hidden_layer_sizes(self):
        for in_s, out_s, h_s, res in [
            (5, 1, [2, 3], [5, 2, 3, 1]),
            (2, 1, None, [2, 1]),
            (2, 1, [], [2, 1]),
        ]:
            with self.subTest(msg="in_s: {}, out_s: {}, h_s: {}, res: {}".format(in_s, out_s, h_s, res)):
                self.assertEqual(set_up_hidden_layer_sizes(input_size=in_s, output_size=out_s, hidden_sizes=h_s), res)

    def test_fc_linear(self):
        for hl, bias in [
            ([2, 1], True),
            ([2, 3, 1], [True, False]),
        ]:
            with self.subTest(msg="hl: {}, bias: {}".format(hl, bias)):
                out = fc_linear(hidden_layers=hl, bias=bias)
                self.assertTrue(isinstance(out, nn.Sequential))
                self.assertEqual(len(out), len(hl) - 1)

                for i, layer in enumerate(out):
                    self.assertTrue(isinstance(layer, nn.Linear))
                    self.assertEqual(layer.bias is not None, bias if isinstance(bias, bool) else bias[i])
                    self.assertEqual(layer.in_features, hl[i])
                    self.assertEqual(layer.out_features, hl[i + 1])

    def test_fc_linear_exceptions(self):
        for hl, bias, err, err_msg in [
            ([2, 1], 1, NotImplementedError, "Bias should be a boolean or a list of booleans. Got: 1"),
            ([2, 3, 1], [True, False, True], ValueError, "Length of bias 3 should be the same as"),
            ([3, 2, 0], True, ValueError, "Input, hidden or output size is <= 0"),
        ]:
            with self.subTest(msg="hl: {}, bias: {}, err: {}, msg: {}".format(hl, bias, err, err_msg)):
                with self.assertRaises(err) as e:
                    _ = fc_linear(hidden_layers=hl, bias=bias)
                self.assertTrue(err_msg in str(e.exception), msg=e.exception)


if __name__ == "__main__":
    unittest.main()
