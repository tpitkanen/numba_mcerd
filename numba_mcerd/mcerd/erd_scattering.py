from typing import List

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


def erd_scattering(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target,
                   detector: o.Detector) -> c.ScatteringType:
    raise NotImplementedError


def get_isotope(I: o.Isotopes) -> float:
    raise NotImplementedError


def save_ion_history(g: o.Global, ion: o.Ion, recoil: o.Ion, detector: o.Detector,
                     sc_lab: o.Vector, sc_ion: o.Vector, Z: float, A: float) -> None:
    raise NotImplementedError


def get_recoiling_dir(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target,
                      detector: o.Detector) -> o.Vector:
    raise NotImplementedError


# TODO: type for cross
def cross_section(z1: float, m1: float, z2: float, m2: float, E: float, theta: float,
                  cross, table: int, ctype: int) -> float:
    raise NotImplementedError


def Serd(z1: float, m1: float, z2: float, m2: float, t: float, E: float) -> float:
    raise NotImplementedError


def Srbs_mc(z1: float, z2: float, t: float, E: float) -> float:
    raise NotImplementedError


def mc2lab_scatc(mcs: float, tcm: float, t: float) -> float:
    raise NotImplementedError


def cm_angle(lab_angle: float, r: float, a: List[float]) -> float:
    raise NotImplementedError


# TODO: type for cross
def inter_cross_table(E: float, theta: float, cross) -> float:
    raise NotImplementedError
