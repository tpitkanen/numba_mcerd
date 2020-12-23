import logging
import re
from enum import Enum
from pathlib import Path

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


# Constants etc. from read_input.h
from numba_mcerd.mcerd import read_target, read_detector
from numba_mcerd.mcerd.jibal import JibalSelectIsotopes

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
    def __init__(self):
        self.a = []
        self.b = []
        self.c = []


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
    unit_value = 0.0
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
        if not line or line.isspace():
            continue
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
            set_ion(g.jibal, value, ion)
            ion.type = c.IonType.PRIMARY
            logging.info(f"Beam ion: {ion.Z=}, M={ion.A / c.C_U}")
        elif key == SettingsLine.I_ENERGY.value:
            number, value = get_float(value)
            unit_value, value = get_unit_value(value, c.C_MEV)
            g.E0 = number * unit_value
            g.ionemax = 1.01 * g.E0
        elif key == SettingsLine.I_TARGET.value:
            logging.info("Reading target file")
            read_target.read_target_file(value, g, target)
        elif key == SettingsLine.I_DETECTOR.value:
            logging.info("Reading detector file")
            read_detector.read_detector_file(value, g, detector, target)
        elif key == SettingsLine.I_RECOIL.value:
            set_ion(g.jibal, value, cur_ion)
            cur_ion.type = c.IonType.SECONDARY
            logging.info(f"Recoil atom Z={cur_ion.Z}, M={cur_ion.A / c.C_U}")
            logging.info(
                f"Recoil atom isotopes: {[f'{cur_ion.I.A[i]} {cur_ion.I.c[i]}%' for i in range(cur_ion.I.n)]}")
            if cur_ion.I.n > 0:
                logging.info(f"Most abundant isotope: {cur_ion.I.Am / c.C_U}")
        elif key == SettingsLine.I_RECDIST.value:
            filename, line = get_word(value)
            fval, n = read_file(filename, columns=2)
            if n < 2:
                raise ReadInputError("Too few points in the recoiling material distribution")
            if fval.a[0] < 0.0:
                raise ReadInputError("Recoil distribution can not start from a negative depth")
            rec_dist_unit = c.C_NM
            target.recmaxd = fval.a[-1] * rec_dist_unit
            target.effrecd = 0.0
            for i in range(n):
                target.recdist[i].x = fval.a[i] * rec_dist_unit
                target.recdist[i].y = fval.b[i] * rec_dist_unit
                if i > 0 and (target.recdist[-1].y > 0.0 or target.recdist[i].y > 0.0):
                    target.effrecd += target.recdist[i].x - target.recdist[-i].x
            target.nrecdist = n
            # del fval
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


def read_file(filename: str, columns: int) -> (Fvalue, int):
    """Read file into an Fvalue object

    Args:
        filename: file to read
        columns: number of columns in file

    Returns:
        Fvalue object and number of lines read
    """
    fp = Path(filename).open("r")
    fv = Fvalue()

    i = 0
    while line := fp.readline().strip():
        if columns == 1:
            # fv.a[i] = float(line)
            fv.a.append(float(line))
        elif columns == 2:
            a, b = line.split()
            fv.a.append(float(a))
            fv.b.append(float(b))
        elif columns == 3:
            a, b, _c = line.split()
            fv.a.append(float(a))
            fv.b.append(float(b))
            fv.c.append(float(_c))
        else:
            raise ReadInputError(f"Unsupported amount of columns: {columns}")
        i += 1

    fp.close()

    return fv, i


def get_atom():
    raise NotImplementedError


def set_ion(jibal: o.Jibal, line: str, ion: o.Ion) -> None:
    """Set an ion's value to specific isotope or element.

    Whitespace is ignored.

    Args:
        jibal: container for element information
        line: line to extract element information from
        ion: ion to edit
    """
    n = 0
    number = 0.0
    Amax = -1.0

    if line[0].isnumeric():
        A, symbol = get_float(line)
        A = int(A)
    else:
        A = None
        symbol, _ = get_word(line)

    element = jibal.copy_normalized_element(
        jibal.get_element_by_name(symbol), JibalSelectIsotopes.NATURAL.value)

    if element is None:
        raise ReadInputError(f"Element '{symbol}' not found")

    ion.Z = element.Z

    if A is not None:
        # FIXME: add 'mass == isotope.A' check, move this to loop
        isotope = element.get_isotope_by_mass_number(A)
        ion.A = isotope.mass
        ion.I.c_sum = 1.0
        found = True
    else:
        found = True
        for n, isotope in enumerate(element.isotopes):
            concentration = element.concs[n]
            ion.I.A[n] = isotope.mass
            ion.I.c[n] = concentration
            ion.I.c_sum += concentration
            if isotope.abundance > Amax:
                Amax = isotope.abundance
                ion.I.Am = isotope.mass
                ion.A = isotope.mass

    ion.I.n = n
    if n and ion.I.c_sum != 0.0:
        logging.warning(f"Concentrations of element '{symbol}' sum up to {ion.I.c_sum * 100.0}% instead of 100%")
    if not found:
        raise ReadInputError(f"Isotope for '{line}' not found")


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


def get_unit_value(line: str, default: float) -> (float, str):
    """Extract a unit from beginning of the line and get its value.

    Whitespace is ignored.

    Args:
        line: line to extract the unit from
        default: default value for the unit if no unit is found in line

    Returns:
        Unit value and the rest of the line
    """
    pieces = line.lstrip().split(maxsplit=1)
    if len(pieces) == 0:
        return default, ""
    elif len(pieces) == 1:
        unit, line = pieces[0], ""
    else:
        unit, line = pieces

    try:
        unit_value = units[unit][0]
    except KeyError:
        raise ReadInputError(f"Could not find value for unit '{unit}'")

    return unit_value, line


# TODO: Use this in get_float?
def get_word(line: str) -> (str, str):
    """Extract a word from the beginning of the line.

    Whitespace is ignored.

    Args:
        line: Line to extract from

    Returns:
        Extracted word and the rest of the line
    """
    pieces = line.lstrip().split(maxsplit=1)
    if len(pieces) == 0:
        raise ReadInputError("No words remaining")
    elif len(pieces) == 1:
        word, line = pieces[0], ""
    else:
        word, line = pieces

    return word, line