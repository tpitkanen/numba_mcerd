import unittest

from numba_mcerd.mcerd import random_vanilla


class TestRandom(unittest.TestCase):
    def setUp(self) -> None:
        random_vanilla.seed_rnd(1)

    def test_determinism(self):
        low = 1
        high = 4
        # Built-in Python RNG
        # self.assertEqual(1.4030927323372038, random.rnd(low, high))
        # self.assertEqual(6.542301210811698, random.rnd(low+3, high+3))

        # Precalculated C RNG (PCG32)
        self.assertEqual(2.998037535464391, random_vanilla.rnd(low, high))
        self.assertEqual(6.485011034877971, random_vanilla.rnd(low+3, high+3))

    def test_invalid_length(self):
        low = 10
        high = 8
        with self.assertRaises(random_vanilla.RndError) as cm:
            random_vanilla.rnd(low, high)
        self.assertEqual("Length negative or zero", str(cm.exception))

    def test_valid_length(self):
        low = 7
        high = 1000
        try:
            random_vanilla.rnd(low, high)
        except random_vanilla.RndError:
            self.fail("Error raised")
