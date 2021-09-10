import unittest

import numpy as np

import numba_mcerd.mcerd.objects_dtype as od
from numba_mcerd.mcerd import pre_simulation_jit


class TestSortPresimu(unittest.TestCase):
    def setUp(self) -> None:
        self.presimu = np.array(
            [
                (10.0, 5.0, 5),
                (7.0, 6.5, 2),
                (12.3, 4.2, 3),
                (4.7, 8.0, 0),
                (0.6, 0.2, 1)
            ],
            dtype=od.Presimu)

    def test_swap_presimus(self):
        expected = np.array(
            [
                (10.0, 5.0, 5),
                (4.7, 8.0, 0),
                (12.3, 4.2, 3),
                (7.0, 6.5, 2),
                (0.6, 0.2, 1)
            ],
            dtype=od.Presimu)
        pre_simulation_jit.swap_presimus(self.presimu, 1, 3)
        assert_ndarrays_equal(self.presimu, expected)

    def test_sort_presimu_by_depth_full(self):
        expected = np.array(
            [
                (0.6, 0.2, 1),
                (4.7, 8.0, 0),
                (7.0, 6.5, 2),
                (10.0, 5.0, 5),
                (12.3, 4.2, 3)
            ],
            dtype=od.Presimu)
        pre_simulation_jit.sort_presimu_by_depth(self.presimu)
        assert_ndarrays_equal(self.presimu, expected)

    def test_sort_presimu_by_depth_range(self):
        expected = np.array(
            [
                (10.0, 5.0, 5),
                (4.7, 8.0, 0),
                (7.0, 6.5, 2),
                (12.3, 4.2, 3),
                (0.6, 0.2, 1)
            ],
            dtype=od.Presimu)
        pre_simulation_jit.sort_presimu_by_depth(self.presimu, 1, 4)
        assert_ndarrays_equal(self.presimu, expected)

    def test_sort_presimu_by_angle_full(self):
        expected = np.array(
            [
                (4.7, 8.0, 0),
                (7.0, 6.5, 2),
                (10.0, 5.0, 5),
                (12.3, 4.2, 3),
                (0.6, 0.2, 1)
            ],
            dtype=od.Presimu)
        pre_simulation_jit.sort_presimu_by_angle(self.presimu)
        assert_ndarrays_equal(self.presimu, expected)

    def test_sort_presimu_by_angle_range(self):
        expected = np.array(
            [
                (7.0, 6.5, 2),
                (10.0, 5.0, 5),
                (12.3, 4.2, 3),
                (4.7, 8.0, 0),
                (0.6, 0.2, 1)
            ],
            dtype=od.Presimu)
        pre_simulation_jit.sort_presimu_by_angle(self.presimu, 0, 3)
        assert_ndarrays_equal(self.presimu, expected)


def assert_ndarrays_equal(array1, array2):
    """Assert that one-dimensional ndarrays have equal shapes and elements."""
    if array1.shape != array2.shape:
        raise AssertionError(
            f"Shapes differ: {array1.shape} != {array2.shape}\n{array1=}\n{array2}")

    for a1, a2 in zip(array1, array2):
        if a1 != a2:
            raise AssertionError(
                f"Elements differ: {a1} != {a2}\n{array1=}\n{array2=}")
