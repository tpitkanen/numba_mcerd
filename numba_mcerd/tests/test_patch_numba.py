import unittest

import numba
import numpy as np

from numba_mcerd import patch_numba


Point = np.dtype([
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float64)
])


PointContainer = np.dtype([
    ("p", Point, 2)
])


@numba.njit
def foo(obj):
    obj["p"]


class MyTestCase(unittest.TestCase):
    def test_patch_nested_array(self):
        point_container = np.zeros(1, PointContainer)[0]

        foo.py_func(point_container)
        self.assertRaisesRegex(
            AttributeError, "'Record' object has no attribute 'bitwidth'", foo, point_container)

        patch_numba.patch_nested_array()

        foo.py_func(point_container)
        foo(point_container)  # No AttributeError
