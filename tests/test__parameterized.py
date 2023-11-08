import unittest

import torch
from easydict import EasyDict


class DeviceParameterizedTestCase(unittest.TestCase):
    """
    Test Case that can call other test cases parameterized, to allow for testing on cpu and cuda

    Taken from: https://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases
    """

    def __init__(self, methodName='runTest', device: torch.device | str = "cuda") -> None:
        """
        Set up config to contain the device through parameter

        Args:
            methodName: parameter needs to be passed down
            device: device to test on
        """
        super(DeviceParameterizedTestCase, self).__init__(methodName=methodName)
        self.device = device
        self.config = EasyDict({"device": device})

    @staticmethod
    def parametrize(testcase_class, device: torch.device | str = "cuda"):
        """
        Create a suite containing all tests taken from the given
        subclass, passing them the parameter 'param'.
        """
        test_suite = unittest.TestSuite()
        for name in unittest.TestLoader().getTestCaseNames(testcase_class):
            test_suite.addTest(testcase_class(name, device=device))
        return test_suite
