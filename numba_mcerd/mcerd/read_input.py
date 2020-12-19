from array import array
from enum import Enum
from pathlib import Path

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


def read_input(g: o.Global, ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
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
            raise NotImplementedError
        elif key == SettingsLine.I_ION.value:
            raise NotImplementedError
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


def get_ion():
    raise NotImplementedError


# def get_string(): pass
# def get_word(): pass
# def get_number(): pass
def get_unit_value():
    raise NotImplementedError


# def trim_space(): pass
# def mc_isnumber(): pass
