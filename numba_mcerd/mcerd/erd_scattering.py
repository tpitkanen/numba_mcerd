import copy
import math
from typing import List

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import rotate, random


class ErdScatteringError(Exception):
    """Error in ERD scattering"""


def erd_scattering(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target,
                   detector: o.Detector) -> bool:
    # TODO: What does this function actually check?

    if ion.scale and ion.p.z > c.SCALE_DEPTH:
        return False

    if g.simtype == c.SimType.SIM_RBS:
        raise NotImplementedError

    recoil.w = ion.w
    if ion.wtmp > 0:
        recoil.w *= ion.wtmp

    sc_target = o.Vector()
    sc_ion = o.Vector()
    if g.recwidth == c.RecWidth.REC_NARROW:
        # Recoiling direction in the laboratory coordinate system
        sc_lab = get_recoiling_dir(g, ion, recoil, target, detector)

        # Recoiling direction in the target coordinate system
        sc_target.theta, sc_target.fii = rotate.rotate(
            g.beamangle, 0.0, sc_lab.theta, sc_lab.fii)

        # Recoiling direction in the ion coordinate system
        sc_ion.theta, sc_ion.fii = rotate.rotate(
            ion.theta, ion.fii - c.C_PI, sc_target.theta, sc_target.fii)
    else:
        # Recoiling direction in the ion coordinate system
        sc_ion = get_recoiling_dir(g, ion, recoil, target, detector)

        sc_target.theta, sc_target.fii = rotate.rotate(
            ion.theta, ion.fii, sc_ion.theta, sc_ion.fii)

        sc_lab = o.Vector()
        sc_lab.theta, sc_lab.fii = rotate.rotate(
            g.beamangle, c.C_PI, sc_target.theta, sc_target.fii)

    if recoil.w > 1e7:
        # TODO: Print warning
        pass

    recoil.scale = ion.scale

    if g.simtype == c.SimType.SIM_ERD:
        if sc_ion.theta < c.C_PI / 2.0:  # TODO: Could do * 0.5
            if recoil.I.n > 0:
                recoil.A = get_isotope(recoil.I)
            recoil.E = (ion.E * math.cos(sc_ion.theta)**2 * 4.0
                        * (ion.A * recoil.A) / (ion.A + recoil.A)**2)
            recoil.theta = sc_target.theta
            recoil.fii = sc_target.fii
            recoil.opt.cos_theta = math.cos(recoil.theta)
            recoil.opt.sin_theta = math.sin(recoil.theta)
            recoil.p = copy.copy(ion.p)
            recoil.tlayer = ion.tlayer
            recoil.nsct = 0

            # Rutherford cross sections for scaling ions if False
            table = False if ion.scale else target.table
            recoil.w *= cross_section(ion.Z, ion.A, recoil.Z, recoil.A, ion.E, sc_ion.theta,
                                      target.cross, table, c.SimType.SIM_ERD)
            recoil.time = 0.0
            recoil.type = c.IonType.SECONDARY
            recoil.status = c.IonStatus.NOT_FINISHED
            recoil.lab.p = copy.copy(ion.lab.p)
            recoil.lab.theta = ion.lab.theta
            recoil.lab.fii = ion.lab.fii
            recoil.virtual = False

            save_ion_history(g, ion, recoil, detector, sc_lab, sc_ion, recoil.Z, recoil.A)
            return True
        else:
            recoil.status = ion.status = c.IonStatus.FIN_NO_RECOIL
            return False
    elif g.simtype == c.SimType.SIM_RBS:
        raise NotImplementedError
    else:
        raise ErdScatteringError(f"Unknown simulation type {g.simtype}")


def get_isotope(I: o.Isotopes) -> float:
    assert 0.9999 < I.c_sum < 1.0001
    r = random.rnd(0.0, I.c_sum, c.RndPeriod.RND_OPEN)

    i = 0
    conc = 0.0
    while i < I.n and r > conc:
        conc += I.c[i]
        i += 1

    if i == I.n and r > conc:
        # TODO: print warning
        conc_sum = 0.0
        for i in range(I.n):
            conc_sum += I.c[i]
        raise ErdScatteringError(f"Error in concentrations")  # TODO: More descriptive

    mass = I.A[i - 1]
    return mass


# TODO: Remove unused arguments
def save_ion_history(g: o.Global, ion: o.Ion, recoil: o.Ion, detector: o.Detector,
                     sc_lab: o.Vector, sc_ion: o.Vector, Z: float, A: float) -> None:
    recoil.hist.tar_recoil.p = copy.copy(recoil.p)
    recoil.hist.tar_recoil.theta = recoil.theta
    recoil.hist.tar_recoil.fii = recoil.fii

    recoil.hist.lab_recoil = copy.copy(sc_lab)
    recoil.hist.lab_recoil.p = copy.copy(ion.lab.p)

    recoil.hist.ion_recoil = copy.copy(sc_ion)

    recoil.hist.tar_primary.p = copy.copy(ion.p)
    recoil.hist.tar_primary.theta = ion.theta
    recoil.hist.tar_primary.fii = ion.fii

    recoil.hist.lab_primary = copy.copy(ion.lab)

    recoil.hist.ion_E = ion.E
    recoil.hist.recoil_E = recoil.E

    recoil.hist.layer = recoil.tlayer
    recoil.hist.nsct = ion.nsct
    recoil.hist.time = ion.time

    recoil.hist.w = recoil.w

    recoil.hist.Z = Z  # Atomic number of the target particle
    recoil.hist.A = A  # Mass of the target particle


def get_recoiling_dir(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target,
                      detector: o.Detector) -> o.Vector:
    """Calculate recoiling angle in the laboratory coordinates or in the
     ion coordinates (wide scheme)
     """

    d = o.Vector()
    if g.simstage == c.SimStage.PRESIMULATION:
        theta = 0.0
        fii = 0.0
        d.theta, d.fii = rotate.rotate(detector.angle, 0.0, theta, fii)
    elif g.recwidth == c.RecWidth.REC_NARROW:
        n = ion.tlayer
        thetamax = target.recpar[n].x * ion.p.z + target.recpar[n].y

        cos_thetamax = math.cos(thetamax)

        d_ion = o.Vector()
        d_target = o.Vector()
        i = 0
        d_ion_theta_ok = True
        while d_ion_theta_ok:
            i += 1
            t = random.rnd(cos_thetamax, 1.0, c.RndPeriod.RND_CLOSED)
            theta = math.acos(t)
            fii = random.rnd(0, 2.0 * c.C_PI, c.RndPeriod.RND_OPEN)
            d_target.theta, d_target.fii = rotate.rotate(detector.angle, 0.0, theta, fii)
            d_ion.theta, d_ion.fii = rotate.rotate(
                ion.theta, ion.fii - c.C_PI, d_target.theta, d_target.fii)

            if i % 5 == 0:
                thetamax = min(thetamax * 2.0, c.C_PI * 0.5)
                cos_thetamax = math.cos(thetamax)
                # TODO: Print info
            if i > 100:
                raise ErdScatteringError("Too many calculation for recoiling angle")

            d_ion_theta_ok = math.cos(d_ion.theta) > g.costhetamin

        d.theta, d.fii = rotate.rotate(detector.angle, 0.0, theta, fii)
        recoil.w *= (thetamax / detector.thetamax)**2
    else:
        raise NotImplementedError

    return d


# TODO: type for cross
def cross_section(z1: float, m1: float, z2: float, m2: float, E: float, theta: float,
                  cross, table: bool, ctype: c.SimType) -> float:
    if ctype == c.SimType.SIM_ERD:
        if theta >= c.C_PI * 0.5:
            value = -1.0
        else:
            value = Serd(z1, m1, z2, m2, theta, E)
    else:
        raise NotImplementedError

    if table:
        value *= inter_cross_table(E, theta, cross)

    return value / c.C_BARN


def Serd(z1: float, m1: float, z2: float, m2: float, t: float, E: float) -> float:
    value = ((z1 * z2 * c.C_E**2 / (8.0 * c.C_PI * c.C_EPSILON0 * E))**2
             * (1.0 + m1 / m2)**2
             / math.cos(t)**3)
    return value


def Srbs_mc(z1: float, z2: float, t: float, E: float) -> float:
    raise NotImplementedError


def mc2lab_scatc(mcs: float, tcm: float, t: float) -> float:
    raise NotImplementedError


def cm_angle(lab_angle: float, r: float, a: List[float]) -> float:
    raise NotImplementedError


# TODO: type for cross
def inter_cross_table(E: float, theta: float, cross) -> float:
    raise NotImplementedError
