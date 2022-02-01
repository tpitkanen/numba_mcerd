import unittest
from typing import Any

import numpy as np
from numba import cuda

from numba_mcerd.mcerd import random_cuda


# TODO: Are CUDA RNG results portable between computers / Numba versions / CUDA versions?
#   If they aren't, these tests will fail.


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.launch_config = 1, 1
        self.rng_states = random_cuda.seed_rnds(1, 1)  # TODO: Multiple blocks?

    def test_rnd(self):
        @cuda.jit()
        def test_func(rng_states: Any, out: np.ndarray):
            thread_id = cuda.grid(1)

            low = 1.0
            high = 4.0

            for i in range(out.shape[0]):
                out[i] = random_cuda.rnd(low, high, rng_states, thread_id)

        expected = np.array([1.3993694510336856, 2.134179071729525])
        results = np.zeros(2, dtype=np.float64)
        test_func[self.launch_config](self.rng_states, results)

        np.testing.assert_array_equal(expected, results)

    def test_gaussian(self):
        @cuda.jit()
        def test_func(rng_states: Any, out: np.ndarray):
            thread_id = cuda.grid(1)

            for i in range(out.shape[0]):
                out[i] = random_cuda.gaussian(rng_states, thread_id)

        expected = np.array([-1.4470637608245427, -1.6603477932661448])
        results = np.zeros(2, dtype=np.float64)
        test_func[self.launch_config](self.rng_states, results)

        for res in results:
            print(res)

        np.testing.assert_array_equal(expected, results)
