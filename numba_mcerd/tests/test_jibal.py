import unittest


from numba_mcerd.mcerd.jibal import Jibal


class TestJibal(unittest.TestCase):
    def test_jibal(self):
        jibal = Jibal()
        jibal.initialize()

        self.assertEqual(119, len(jibal.elements))  # neutron + 118 elements

        He = jibal.get_element(2)
        self.assertEqual(He, jibal.get_element_by_name("He"))
        self.assertEqual("He", He.name)
        self.assertEqual(2, He.Z)
        self.assertAlmostEqual(4.002601026852, He.avg_mass)
        self.assertEqual(8, len(He.isotopes))

        He3 = jibal.get_isotope_by_neutron_number(2, 1)
        self.assertEqual(He3, jibal.get_isotope_by_mass_number(2, 3))
        self.assertEqual(He3, He.get_isotope_by_mass_number(3))
        self.assertEqual("He", He3.name)
        self.assertEqual(2, He3.Z)
        self.assertEqual(1, He3.N)
        self.assertEqual(3, He3.A)
        self.assertEqual(3.016029, He3.mass)
        self.assertEqual(0.000002, He3.abundance)

        self.assertEqual(He3, jibal.get_isotope_by_neutron_number(2, 1))
        self.assertEqual(He3, jibal.get_isotope_by_mass_number(2, 3))

        He4 = jibal.get_isotope_by_neutron_number(2, 2)
        self.assertEqual("He", He4.name)
        self.assertEqual(2, He4.Z)
        self.assertEqual(2, He4.N)
        self.assertEqual(4, He4.A)
        self.assertEqual(4.002603, He4.mass)
        self.assertEqual(0.999998, He4.abundance)
