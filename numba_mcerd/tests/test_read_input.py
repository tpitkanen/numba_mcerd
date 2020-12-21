import unittest

from numba_mcerd.mcerd import read_input

# TODO: Make more tests

class TestGetFloat(unittest.TestCase):
    def test_valid_input(self):
        self.assertEqual((10.0, ""), read_input.get_float("10"))
        self.assertEqual((1.0, ""), read_input.get_float("1.0"))
        self.assertEqual((1.0, "b"), read_input.get_float(" 1 b"))
        self.assertEqual((1.0, "b"), read_input.get_float("1 b"))
        self.assertEqual((1.0, "b"), read_input.get_float(" 1b"))
        self.assertEqual((0.1, ""), read_input.get_float(".1"))
        self.assertEqual((1.0, ""), read_input.get_float("1."))
        self.assertEqual((5e10, ""), read_input.get_float("5e10"))
        self.assertEqual((5e10, ""), read_input.get_float("5E10"))
        self.assertEqual((5e-10, ""), read_input.get_float("5e-10"))
        self.assertEqual((-5e-10, ""), read_input.get_float("-5e-10"))
        self.assertEqual((5e-10, ""), read_input.get_float("+5e-10"))
        self.assertEqual((5e+10, ""), read_input.get_float("+5e+10"))
        self.assertEqual((5e+1, ""), read_input.get_float("+5e+01"))
        self.assertEqual((5e+0, ""), read_input.get_float("+5e+0"))
        self.assertEqual((5e+1, "e10"), read_input.get_float("+5e+01e10"))
        self.assertEqual((5e+1, "a"), read_input.get_float("+5e+01a"))
        self.assertEqual((5.0, "."), read_input.get_float("5.."))

    def test_invalid_input(self):
        self.assertRaises(ValueError, read_input.get_float, "A1")
        self.assertRaises(ValueError, read_input.get_float, "")


class TestGetUnitValue(unittest.TestCase):
    def test_good_input(self):
        value, line = read_input.get_unit_value("MeV", 1.0)
        self.assertAlmostEqual(1.602e-13, value)
        self.assertEqual("", line)

        value, line = read_input.get_unit_value("ms asdf", 1.0)
        self.assertAlmostEqual(1.0e-3, value)
        self.assertEqual("asdf", line)

    def test_default_value(self):
        value, line = read_input.get_unit_value("", 1.0)
        self.assertAlmostEqual(1.0, 1.0)
        self.assertEqual("", line)

    def test_bad_input(self):
        self.assertRaises(read_input.ReadInputError, read_input.get_unit_value, "qwerty", 1.0)
        self.assertRaises(read_input.ReadInputError, read_input.get_unit_value, "5 kg", 2)
