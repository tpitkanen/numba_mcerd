import copy
import logging
import re
from array import array
from enum import Enum
from pathlib import Path
from typing import Optional

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


# Constants etc. from read_input.h

MAXUNITSTRING = 20
MAXLEN = 200
ERR_INPUT = 100

C_DEFAULT = 1.0
C_V0 = 2187691.42  # Bohr velocity to m/s


class SettingsLine(Enum):
    """Types of lines in MCERD settings file"""
    I_TYPE = "Type of simulation:"                               # 0
    I_ION = "Beam ion:"                                          # 1
    I_ENERGY = "Beam energy:"                                    # 2
    I_TARGET = "Target description file:"                        # 3
    I_DETECTOR = "Detector description file:"                    # 4
    I_RECOIL = "Recoiling atom:"                                 # 5
    I_RECDIST = "Recoiling material distribution:"               # 6
    I_TANGLE = "Target angle:"                                   # 7
    I_SPOTSIZE = "Beam spot size:"                               # 8
    I_MINANGLE = "Minimum angle of scattering:"                  # 9
    I_MINENERGY = "Minimum energy of ions:"                      # 10
    I_NIONS = "Number of ions:"                                  # 11
    I_NPREIONS = "Number of ions in the presimulation:"          # 12
    I_RECAVE = "Average number of recoils per primary ion:"      # 13
    I_SEED = "Seed number of the random number generator:"       # 14
    I_RECWIDTH = "Recoil angle width (wide or narrow):"          # 15
    I_PREDATA = "Presimulation result file:"                     # 16
    I_MINSCAT = "Minimum main scattering angle:"                 # 17
    I_BDIV = "Beam divergence:"                                  # 18
    I_BPROF = "Beam profile:"                                    # 19
    I_SURFACE = "Surface topography file:"                       # 20
    I_SURFSIZE = "Side length of the surface topography image:"  # 21
    I_NSCALE = "Number of real ions per each scaling ion:"       # 22
    I_TRACKP = "Trackpoint output:"                              # 23
    I_MISSES = "Output misses:"                                  # 24
    I_CASCADES = "Recoil cascades:"                              # 25
    I_NCASCADES = "Number of recoils in a cascade:"              # 26
    I_ADVANCED = "Advanced output:"                              # 27
    I_NOMC = "No multiple scattering:"                           # 28
    # NINPUT = 29  # len(SettingsLine)


# Units with the following structure:
# unit: (value, flag)
# (flag seems to be unused)
units = {
    "rad":      (C_DEFAULT,     0),
    "radians":  (C_DEFAULT,     0),
    s.SYM_DEG:  (c.C_DEG,       0),
    "degree":   (c.C_DEG,       0),
    "deg":      (c.C_DEG,       0),

    "J":        (C_DEFAULT,     0),
    "eV":       (c.C_EV,        0),
    "keV":      (c.C_KEV,       0),
    "MeV":      (c.C_MEV,       0),

    "kg":       (C_DEFAULT,     0),
    "u":        (c.C_U,         0),

    "s":        (1.0,           0),
    "ms":       (1.0e-3,        0),
    s.SYM_US:   (1.0e-6,        0),
    "ns":       (1.0e-9,        0),
    "ps":       (1.0e-12,       0),
    "fs":       (1.0e-15,       0),
    "as":       (1.0e-18,       0),

    "m":        (1.0,           0),
    "km":       (1.0e3,         0),
    "cm":       (1.0e-2,        0),
    "mm":       (1.0e-3,        0),
    s.SYM_UM:   (1.0e-6,        0),
    "nm":       (1.0e-9,        0),
    "pm":       (1.0e-12,       0),
    "fm":       (1.0e-15,       0),
    s.SYM_A:    (1.0e-10,       0),

    "m/s":      (1.0,           0),
    "km/s":     (1000.0,        0),
    "cm/s":     (1.0e-2,        0),
    "v0":       (C_V0,          0),
    "c":        (c.C_C,         0),
    "%c":       (0.01 * c.C_C,  0),

    "N":                    (1.0,                           0),
    "J/m":                  (1.0,                           0),
    "keV/nm":               ((c.C_KEV/c.C_NM),              0),
    s.SYM_KEV_UM:           (1000.0 * (c.C_KEV/c.C_NM),     0),
    "keV/um":               (1000.0 * (c.C_KEV/c.C_NM),     0),
    s.SYM_EV_A:             (100.0 * (c.C_KEV/c.C_NM),      0),
    "eV/A":                 (100.0 * (c.C_KEV/c.C_NM),      0),
    "MeV/mm":               (1000.0 * (c.C_KEV/c.C_NM),     0),
    s.SYM_KEV_UG_CM2:       (1.0,                           1),
    "keV/(ug/cm2)":         (1.0,                           1),
    s.SYM_EVCM2_1E15ATOMS:  (1.0,                           1),
    s.SYM_MEV_MG_CM2:       (1.0,                           1),
    "MeV/(mg/cm2)":         (1.0,                           1),
    s.SYM_EV_UG_CM2:        (1.0,                           1),
    "eV/(ug/cm2)":          (1.0,                           1),

    s.SYM_G_CM3:    (1000.0,     0),
    "g/cm3":        (1000.0,     0),
    s.SYM_KG_M3:    (C_DEFAULT,  0),
    "kg/m3":        (C_DEFAULT,  0),

    "END":          (-1.0,      -1)
}


class Fvalue:
    """Class for containing pre-simulation values"""
    # TODO: Max size is c.N_INPUT
    def __init__(self):
        self.a = array("d")
        self.b = array("d")
        self.c = array("d")


class ReadInputError(Exception):
    """Error while reading input"""


# TODO: ion is actually an array of ions in this and 2 other functions
def read_input(g: o.Global, ion: o.Ion, cur_ion: o.Ion, previous_trackpoint_ion: o.Ion,
               target: o.Target, detector: o.Detector) -> None:
    # FILE *fp, *fout;
    p1 = o.Point()
    p2 = o.Point()
    p3 = o.Point()
    # char buf[MAXLEN], *c, *word;
    lines = None
    number = 0.0
    unit = 0.0
    rec_dist_unit = 0.0
    M = 0.0
    # int i, j, rinput[NINPUT], n;  # rinput unused

    if len(g.master.args) <= 1:
        raise ReadInputError("No command file given")
    fp = Path(g.master.args[1])
    if not fp.exists():
        raise ReadInputError(f"Command file '{fp}' doesn't exist")

    with fp.open("r") as f:
        lines = f.readlines()

    keys = []
    values = []
    for line in lines:
        key, value = line.split(sep=":", maxsplit=1)
        keys.append(key.strip() + ":")
        values.append(value.strip())

    for key, value in zip(keys, values):
        if key == SettingsLine.I_TYPE.value:
            if value == "ERD":
                g.simtype = c.SimType.SIM_ERD
                g.nions = 2  # Incident and recoil
            elif value == "RBS":
                g.simtype = c.SimType.SIM_RBS
                g.nions = 3  # Incident, target and scattered incident
            else:
                raise ValueError(f"No such type for simulation: '{value}'")
            # (Ions are already initialized)
        elif key == SettingsLine.I_ION.value:
            get_ion(g.jibal, value, ion)
            ion.type = c.IonType.PRIMARY
            logging.info(f"Beam ion: {ion.Z=}, M={ion.A / c.C_U}")
        elif key == SettingsLine.I_ENERGY.value:
            raise NotImplementedError
        elif key == SettingsLine.I_TARGET.value:
            raise NotImplementedError
        elif key == SettingsLine.I_DETECTOR.value:
            raise NotImplementedError
        elif key == SettingsLine.I_RECOIL.value:
            raise NotImplementedError
        elif key == SettingsLine.I_RECDIST.value:
            raise NotImplementedError
        elif key == SettingsLine.I_TANGLE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_SPOTSIZE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_MINANGLE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_MINENERGY.value:
            raise NotImplementedError
        elif key == SettingsLine.I_NIONS.value:
            raise NotImplementedError
        elif key == SettingsLine.I_NPREIONS.value:
            raise NotImplementedError
        elif key == SettingsLine.I_RECAVE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_SEED.value:
            raise NotImplementedError
        elif key == SettingsLine.I_RECWIDTH.value:
            raise NotImplementedError
        elif key == SettingsLine.I_PREDATA.value:
            raise NotImplementedError
        elif key == SettingsLine.I_MINSCAT.value:
            raise NotImplementedError
        elif key == SettingsLine.I_BDIV.value:
            raise NotImplementedError
        elif key == SettingsLine.I_BPROF.value:
            raise NotImplementedError
        elif key == SettingsLine.I_SURFACE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_SURFSIZE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_NSCALE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_TRACKP.value:
            raise NotImplementedError
        elif key == SettingsLine.I_MISSES.value:
            raise NotImplementedError
        elif key == SettingsLine.I_CASCADES.value:
            raise NotImplementedError
        elif key == SettingsLine.I_NCASCADES.value:
            raise NotImplementedError
        elif key == SettingsLine.I_ADVANCED.value:
            raise NotImplementedError
        elif key == SettingsLine.I_NOMC.value:
            raise NotImplementedError
        else:
            raise NotImplementedError

    # TODO
    raise NotImplementedError


def read_file():
    raise NotImplementedError


def get_atom():
    raise NotImplementedError


def get_ion(jibal: o.Jibal, line: str, ion: o.Ion) -> None:
    symbol = None
    A = None
    found = False
    n = 0
    number = 0.0
    Amax = -1.0

    # TODO: 'A' was originally 'number' and a double
    # A, symbol = split_element_string(line.strip())
    A, symbol = split_element_string(line)
    A = int(A)

    element = jibal.get_element_by_name(symbol)
    element = copy.deepcopy(element)  # TODO: Replace with real Jibal copying

    if element is None:
        raise ReadInputError(f"Element '{symbol}' not found")

    ion.Z = element.Z

    if A is not None:
        isotope = element.get_isotope_by_mass_number(A)
        ion.A = isotope.mass
        ion.I.c_sum = 1.0
        found = True
    else:
        raise NotImplementedError
        # for isotope in element.isotopes.values():
        #     found = True
        #     ion.I.A.append(isotope.mass)
        #     # TODO: Get the real concentrations
        #     # ion.I.c.append(conc
        #     # ion.I.c_sum += conc
        #     if isotope.abundance > Amax:
        #         Amax = isotope.abundance
        #         ion.I.Am = isotope.mass
        #         ion.A = isotope.mass  # TODO: This seems to be far too big (see log message)
        #     n += 1

    ion.I.n = n
    if n and ion.I.c_sum != 0.0:
        logging.warning(f"Concentrations of element '{symbol}' sum up to {ion.I.c_sum * 100.0}% instead of 100%")
    if not found:
        raise ReadInputError(f"Isotope for '{line}' not found")


def split_element_string(element_string: str) -> (Optional[int], str):
    """Split element string into mass number and symbol.

    Example: "35Cl" -> 35, "Cl"
    """
    # TODO: Add error checking
    i = 0
    for i in range(len(element_string)):
        if not element_string[i].isdigit():
            break
    mass_number = element_string[:i].strip()
    if mass_number:
        mass_number = int(mass_number)
    else:
        mass_number = None

    symbol = element_string[i:].strip()
    return mass_number, symbol


# TODO: This requires converting to float -> remove?
def get_int(line: str) -> (int, str):
    """Extract an integer from the beginning of the line.

    Whitespace around the number is ignored.

    Supported features:
    - scientific notation
    - sign

    Args:
        line:

    Returns:
        Extracted number and the rest of the line
    """
    _, number, rest = re.split(
        r"^\s*("               # whitespace, begin capture group
        r"[+-]?"              # sign
        r"\d+"                # number(s) around comma, or bare integer
        r"(?:[eE][+-]?\d+)?"  # scientific notation
        r")\s*",              # end capture group, whitespace
        line,
        maxsplit=1)
    # Scientific notation is treated as a float in Python, so it has to be converted to float first
    return int(float(number)), rest


def get_float(line: str) -> (float, str):
    """Extract a float from the beginning of the line.

    Whitespace around the number is ignored.

    Supported features:
    - leaving out a zero before or after a comma
    - scientific notation
    - sign

    Examples:
        35Cl
        10.0 Mev
        3.0 5.0 mm (first number)
        1
        1.1
        -1.1
        1e9
        1.5e9
        1.5e+9
        1.5e-9
        -1.5e-9
        .1
        1.

    Args:
        line: Line to extract the number from

    Returns:
        Extracted number and the rest of the line
    """
    _, number, rest = re.split(
        r"^\s*("                                          # whitespace, begin capture group
        r"[+-]?"                                         # sign
        r"(?:(?:\d+\.\d+)|(?:\d+\.)|(?:\.\d+)|(?:\d+))"  # number(s) around comma, or bare integer
        r"(?:[eE][+-]?\d+)?"                             # scientific notation
        r")\s*",                                         # end capture group, whitespace
        line,
        maxsplit=1)
    return float(number), rest


# def get_string(): pass
# def get_word(): pass
# def get_number(): pass
def get_unit_value():
    raise NotImplementedError


# def trim_space(): pass
# def mc_isnumber(): pass
