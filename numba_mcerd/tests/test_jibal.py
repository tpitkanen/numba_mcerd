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
        # TODO: avg_mass
        self.assertEqual(8, len(He.isotopes))

        He2 = jibal.get_isotope(2, 1)
        self.assertEqual("He", He2.name)
        self.assertEqual(2, He2.Z)
        self.assertEqual(1, He2.N)
        self.assertEqual(3, He2.A)
        self.assertEqual(3.016029, He2.mass)
        # TODO: abundance


