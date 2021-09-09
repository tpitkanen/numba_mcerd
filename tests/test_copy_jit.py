import unittest

import numpy as np

from numba_mcerd.mcerd import copy_jit
from numba_mcerd.mcerd import objects_dtype as od
from numba_mcerd.mcerd import objects_jit as oj


class MyTestCase(unittest.TestCase):
    def test_copy_point_dtype(self):
        expected_zeros = np.zeros(1, dtype=od.Point)[0]
        expected_123 = np.array((1., 2., 3.), dtype=od.Point)[()]

        a = np.zeros(1, dtype=od.Point)[0]
        b = np.zeros(1, dtype=od.Point)[0]

        a_id = id(a)
        b_id = id(b)

        self.assertEqual(a, b)

        b["x"], b["y"], b["z"] = 1., 2., 3.
        self.assertNotEqual(a, b)

        copy_jit.copy_point(a, b)
        self.assertEqual(a, expected_123)
        self.assertEqual(a, expected_123)
        self.assertEqual(a, b)

        a["x"], a["y"], a["z"] = 0., 0., 0.
        self.assertNotEqual(a, b)
        self.assertEqual(a, expected_zeros)
        self.assertEqual(b, expected_123)

        self.assertEqual(id(a), a_id)
        self.assertEqual(id(b), b_id)

    def test_copy_vector_dtype(self):
        expected_zeros = np.zeros(1, dtype=od.Vector)[0]
        # https://numpy.org/devdocs/reference/arrays.scalars.html#indexing
        expected_123456 = np.array(((1., 2., 3.), 4., 5., 6.), dtype=od.Vector)[()]

        # Clunkier initialization:
        # expected_123456 = np.zeros(1, dtype=od.Vector)[0]
        # expected_123456["p"]["x"], expected_123456["p"]["y"], expected_123456["p"]["z"] = 1., 2., 3.
        # expected_123456["theta"], expected_123456["fii"], expected_123456["v"] = 4., 5., 6.

        a = np.zeros(1, dtype=od.Vector)[0]
        b = np.zeros(1, dtype=od.Vector)[0]

        a_id = id(a)
        b_id = id(b)

        self.assertEqual(a, b)

        b["p"]["x"], b["p"]["y"], b["p"]["z"] = 1., 2., 3.
        b["theta"] = 4.
        b["fii"] = 5.
        b["v"] = 6.
        self.assertNotEqual(a, b)

        copy_jit.copy_vector(a, b)
        self.assertEqual(a, expected_123456)
        self.assertEqual(b, expected_123456)
        self.assertEqual(a, b)

        a["p"]["x"], a["p"]["y"], a["p"]["z"] = 0., 0., 0.
        a["theta"] = 0.
        a["fii"] = 0.
        a["v"] = 0.

        self.assertNotEqual(a, b)
        self.assertEqual(a, expected_zeros)
        self.assertEqual(b, expected_123456)

        self.assertEqual(id(a), a_id)
        self.assertEqual(id(b), b_id)
