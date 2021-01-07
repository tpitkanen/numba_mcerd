import unittest


from numba_mcerd.mcerd.jibal import Jibal, JibalSelectIsotopes


# TODO: Update tests
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
        self.assertEqual("3He", He3.name)
        self.assertEqual(2, He3.Z)
        self.assertEqual(1, He3.N)
        self.assertEqual(3, He3.A)
        self.assertEqual(3.016029, He3.mass)
        self.assertEqual(0.000002, He3.abundance)

        self.assertEqual(He3, jibal.get_isotope_by_neutron_number(2, 1))
        self.assertEqual(He3, jibal.get_isotope_by_mass_number(2, 3))

        He4 = jibal.get_isotope_by_neutron_number(2, 2)
        self.assertEqual("4He", He4.name)
        self.assertEqual(2, He4.Z)
        self.assertEqual(2, He4.N)
        self.assertEqual(4, He4.A)
        self.assertEqual(4.002603, He4.mass)
        self.assertEqual(0.999998, He4.abundance)

    def test_normalize(self):
        jibal = Jibal()
        jibal.initialize()

        Li = jibal.get_element(3)
        self.assertEqual("Li", Li.name)
        self.assertEqual(3, Li.Z)
        self.assertEqual(11, len(Li.isotopes))
        self.assertEqual(11, len(Li.concs))
        self.assertAlmostEqual(6.96746032, Li.avg_mass)

        Li_all = jibal.copy_normalized_element(Li, JibalSelectIsotopes.ALL.value)
        self.assertIsNot(Li, Li_all)
        self.assertEqual("Li", Li_all.name)
        self.assertEqual(3, Li_all.Z)
        self.assertEqual(11, len(Li_all.isotopes))
        self.assertEqual(11, len(Li_all.concs))
        self.assertAlmostEqual(6.96746032, Li_all.avg_mass)

        Li_nat = jibal.copy_normalized_element(Li, JibalSelectIsotopes.NATURAL.value)
        self.assertIsNot(Li, Li_nat)
        self.assertEqual("natLi", Li_nat.name)
        self.assertEqual(3, Li_nat.Z)
        self.assertEqual(2, len(Li_nat.isotopes))
        self.assertEqual(2, len(Li_nat.concs))
        self.assertAlmostEqual(6.96746032, Li_all.avg_mass)

        Li6 = jibal.copy_normalized_element(Li, 6)
        self.assertIsNot(Li, Li6)
        self.assertEqual("6Li", Li6.name)
        self.assertEqual(3, Li6.Z)
        self.assertEqual(1, len(Li6.isotopes))
        self.assertEqual(1, len(Li6.concs))
        self.assertEqual(6.015123, Li6.avg_mass)
