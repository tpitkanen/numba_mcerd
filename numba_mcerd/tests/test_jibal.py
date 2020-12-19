import unittest


from numba_mcerd.mcerd.jibal import Jibal


class TestJibal(unittest.TestCase):
    def test_jibal(self):
        jibal = Jibal()
        jibal.initialize()

        self.assertEqual(119, len(jibal.elements))  # neutron + 118 elements

        He = jibal.get_element(2)
        self.assertEqual("He", He.name)
        self.assertEqual(2, He.Z)
        self.assertAlmostEqual(4.002601026852, He.avg_mass)
        self.assertEqual(8, len(He.isotopes))

        He2 = jibal.get_isotope(2, 1)
        self.assertEqual("He", He2.name)
        self.assertEqual(2, He2.Z)
        self.assertEqual(1, He2.N)
        self.assertEqual(3, He2.A)
        self.assertEqual(3.016029, He2.mass)
        self.assertEqual(0.000002, He2.abundance)

        He3 = jibal.get_isotope(2, 2)
        self.assertEqual("He", He3.name)
        self.assertEqual(2, He3.Z)
        self.assertEqual(2, He3.N)
        self.assertEqual(4, He3.A)
        self.assertEqual(4.002603, He3.mass)
        self.assertEqual(0.999998, He3.abundance)
