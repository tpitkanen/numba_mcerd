import logging
import math
import re
from enum import Enum
from pathlib import Path

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


# Constants etc. from read_input.h
from numba_mcerd.mcerd import read_target, read_detector, init_detector, random
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
    I_PREDATA_DISABLE = "Presimulation * result file:"           # 29, own addition
    # NINPUT = 30  # len(SettingsLine)


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


def read_input(g: o.Global, primary_ion: o.Ion, secondary_ion: o.Ion, tertiary_ion: o.Ion,
               target: o.Target, detector: o.Detector) -> None:
    """Read settings from the file location specified in g.master.args[1]"""
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

    # TODO: Split these into separate functions to prevent variable reuse bugs,
    #       rename value to line,
    #       make sure that all mandatory settings are defined
    #       mark untested lines
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
            set_ion(g.jibal, value, primary_ion)
            primary_ion.type = c.IonType.PRIMARY
            logging.info(f"Beam ion: {primary_ion.Z=}, M={primary_ion.A / c.C_U}")
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
            set_ion(g.jibal, value, secondary_ion)
            secondary_ion.type = c.IonType.SECONDARY
            logging.info(f"Recoil atom Z={secondary_ion.Z}, M={secondary_ion.A / c.C_U}")
            logging.info(
                f"Recoil atom isotopes: {[f'{secondary_ion.I.A[i]} {secondary_ion.I.c[i]}%' for i in range(secondary_ion.I.n)]}")
            if secondary_ion.I.n > 0:
                logging.info(f"Most abundant isotope: {secondary_ion.I.Am / c.C_U}")
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
                    target.effrecd += target.recdist[i].x - target.recdist[i - 1].x
            target.nrecdist = n
            # del fval
        elif key == SettingsLine.I_TANGLE.value:
            beam_angle, line = get_float(value)
            unit, _ = get_unit_value(line, c.C_DEG)
            g.beamangle = c.C_PI / 2.0 - beam_angle * unit
        elif key == SettingsLine.I_SPOTSIZE.value:
            x, line = get_float(value)
            y, line = get_float(line)
            unit, _ = get_unit_value(line, c.C_MM)
            g.bspot.x = 0.5 * x * unit
            g.bspot.y = 0.5 * y * unit
        elif key == SettingsLine.I_MINANGLE.value:
            min_angle, line = get_float(value)
            unit, _ = get_unit_value(line, c.C_DEG)
            g.minangle = min_angle * unit
        elif key == SettingsLine.I_MINENERGY.value:
            min_angle, line = get_float(value)
            unit, _ = get_unit_value(line, c.C_DEG)
            g.emin = min_angle * unit
        elif key == SettingsLine.I_NIONS.value:
            nsimu, _ = get_float(value)
            g.nsimu = int(nsimu)
        elif key == SettingsLine.I_NPREIONS.value:
            npresimu, _ = get_float(value)
            g.npresimu = int(npresimu)
        elif key == SettingsLine.I_RECAVE.value:
            nrecave, _ = get_float(value)
            g.nrecave = int(nrecave)
        elif key == SettingsLine.I_SEED.value:
            seed, _ = get_float(value)
            seed = int(seed)
            g.seed = seed
            random.seed_rnd(seed)  # numba_mcerd.mcerd.random
        elif key == SettingsLine.I_RECWIDTH.value:
            width, _ = get_word(value)
            if width.lower() == "wide":
                g.recwidth = c.RecWidth.REC_WIDE
            else:
                g.recwidth = c.RecWidth.REC_NARROW
        elif key == SettingsLine.I_PREDATA.value:
            # Note that this is commented out in settings file ('*' in
            # the middle of the key)
            predata_file, _ = get_word(value)
            fval, n = read_file(predata_file, columns=2)
            for i in range(n):
                target.recpar[i].x = fval.a[i] * c.C_DEG / c.C_NM
                target.recpar[i].y = fval.b[i] * c.C_DEG
            # del fval
            g.predata = True
        elif key == SettingsLine.I_MINSCAT.value:
            min_scat, line = get_float(value)
            unit, _ = get_unit_value(line, c.C_DEG)
            g.beamdiv = math.cos(min_scat * unit)
        elif key == SettingsLine.I_BDIV.value:
            beam_div, line = get_float(value)
            unit, _ = get_unit_value(line, c.C_DEG)
            g.beamdiv = math.cos(beam_div * unit)
        elif key == SettingsLine.I_BPROF.value:
            beam_profile, _ = get_word(value)
            if beam_profile.lower() == "flat":
                g.beamprof = c.BeamProf.BEAM_FLAT
            else:
                g.beamprof = c.BeamProf.BEAM_GAUSS
        elif key == SettingsLine.I_SURFACE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_SURFSIZE.value:
            raise NotImplementedError
        elif key == SettingsLine.I_NSCALE.value:
            nscale, _ = get_float(value)
            g.nscale = nscale
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
        elif key == SettingsLine.I_PREDATA_DISABLE.value:
            logging.info("Commented-out presimulation setting skipped")
        else:
            raise NotImplementedError

    p1, p2, p3 = o.Point(), o.Point(), o.Point()

    p2.y = 1.0
    p3.x = math.sin(c.C_PI / 2.0 - g.beamangle)
    p3.z = math.cos(c.C_PI / 2.0 - g.beamangle)

    target.plane = init_detector.get_plane_params(p1, p2, p3)

    if g.recwidth == c.RecWidth.REC_WIDE and g.npresimu > 0:
        g.presimu = 0
        logging.warning("Presimulation not needed with wide recoil angle scheme")

    if g.recwidth == c.RecWidth.REC_WIDE:
        g.predata = False

    if g.predata:
        g.presimu = 0
    elif g.npresimu == 0 and g.recwidth == c.RecWidth.REC_NARROW:
        raise ReadInputError("No precalculated recoiling data in the narrow recoiling scheme")

    if g.npresimu > 0:
        g.cpresimu = 0
        g.simstage = c.SimStage.PRESIMULATION

        g.presimu = [o.Presimu() for _ in range(g.npresimu)]
    else:
        g.simstage = c.SimStage.REALSIMULATION

    target.ntarget = target.nlayers - detector.nfoils

    if detector.type == c.DetectorType.TOF:
        detector.tdet[0] += target.ntarget
        detector.tdet[1] += target.ntarget
        detector.edet[0] += target.ntarget

    i = target.nlayers - 1
    while i > target.ntarget:
        target.layer[i].dlow = 0.0
        target.layer[i].dhigh -= target.layer[i - 1].dhigh
        i -= 1

    g.nsimu += g.npresimu

    M = 4.0 * primary_ion.A * secondary_ion.A / (primary_ion.A + secondary_ion.A) ** 2

    if g.simtype == c.SimType.SIM_ERD:
        g.costhetamax = math.sqrt(g.emin / g.E0 * M)
        g.costhetamin = 1.0
    elif g.simtype == c.SimType.SIM_RBS:
        if primary_ion.A <= secondary_ion.A:
            g.costhetamax = -1.0
        else:
            g.costhetamax = math.sqrt(1.0 - (secondary_ion.A / primary_ion.A) ** 2)


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
        A = round(A)
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
        n += 1  # Offset for compatibility with original code

    ion.I.n = n
    if n and ion.I.c_sum != 1.0:
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
        r"^\s*("                                         # whitespace, begin capture group
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
