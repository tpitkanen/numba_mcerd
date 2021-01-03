import unittest

import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import potential


class TestPotential(unittest.TestCase):
    def test_make_screening_table(self):
        pot = o.Potential()
        potential.make_screening_table(pot)

        self.assertAlmostEqual(33.333333333333336, pot.d)
        self.assertEqual(3201, pot.n)
        self.assertEqual(3201, len(pot.u))

        for point in pot.u:
            self.assertGreaterEqual(point.x, 0)
            self.assertGreater(point.y, 0)

    def test_get_max_x(self):
        self.assertAlmostEqual(96.0, potential.get_max_x())

    def test_get_npoints(self):
        self.assertEqual(3201, potential.get_npoints(96.0))

    def test_U(self):
        self.assertAlmostEqual(1.000001, potential.U(0))
        self.assertAlmostEqual(0.8927537800626411, potential.U(0.1))
        self.assertAlmostEqual(0.6095592002809335, potential.U(0.5))
        self.assertAlmostEqual(0.41644022808570563, potential.U(1))
