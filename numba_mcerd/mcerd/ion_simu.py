import math
from typing import List

import numpy as np

from numba_mcerd.mcerd import random, cross_section
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.constants as c


class IonSimulationError(Exception):
    """Error in ion simulation"""


# TODO: Untested
# TODO: Create unit tests
def create_ion(g: o.Global, ion: o.Ion, target: o.Target) -> None:
    """Calculate parameters for ion (primary or secondary)

    About coordinate systems:
    - laboratory coordinates: z-axis is along the beam line
    - target: z-axis is along the target normal
    - ion: z-axis is along the direction of ion movement
    - detector: z-axis is along the detector direction
    """
    # assert g.simtype == c.SimType.SIM_ERD or g.simtype == c.SimType.SIM_RBS
    if g.simtype != c.SimType.SIM_ERD and g.simtype != c.SimType.SIM_RBS:
        raise IonSimulationError("Unsupported simulation type")

    x = random.rnd(-g.bspot.x, g.bspot.x, c.RndPeriod.RND_CLOSED)
    y = random.rnd(-g.bspot.y, g.bspot.y, c.RndPeriod.RND_CLOSED)
    if g.beamangle > 0:
        z = x / math.tan(math.pi / 2.0 - g.beamangle)
    else:
        z = 0.0

    ion.lab.p = o.Point(x, y, z)

    if g.simstage == c.SimStage.REALSIMULATION and g.cion % (g.nscale + 1) == 0:
        ion.scale = True
        ion.lab.p.x = 0.0
        ion.lab.p.y = 0.0
        ion.lab.p.z = 0.0
        ion.effrecd = c.SCALE_DEPTH
        ion.w = g.nscale * c.SCALE_DEPTH / target.effrecd
    else:
        ion.scale = False
        ion.effrecd = target.effrecd
        ion.w = 1.0

    theta = g.beamangle
    fii = 0.0

    ion.tlayer = 0
    ion.time = 0.0
    ion.p.x = 0.0
    # (debug thing here)
    ion.p.y = 0.0
    ion.p.z = 0.001 * c.C_ANGSTROM
    ion.E = g.E0
    ion.nsct = 0
    ion.theta = theta
    ion.fii = fii
    ion.opt.cos_theta = math.cos(theta)
    ion.opt.sin_theta = math.sin(theta)
    ion.type = c.IonType.PRIMARY
    ion.status = c.IonStatus.NOT_FINISHED

    ion.lab.theta = g.beamangle
    ion.lab.fii = math.pi

    # if __debug__:
    #     debug.print_ion_position(g, ion, "L", c.SimStage.ANYSIMULATION)


def move_target(target: o.Target) -> None:
    raise NotImplementedError


def next_scattering(g: o.Global, ion: o.Ion, target: o.Target,
                    scat: List[List[o.Scattering]], snext: o.SNext) -> None:
    """Calculate impact parameter (ion.opt.y) and distance to the next
     scattering point (snext.d)
     """
    if g.nomc:
        snext.d = 100.0 * c.C_NM
        return

    cross = 0.0
    b = np.zeros(shape=c.MAXATOMS, dtype=np.float64)
    layer = target.layer[ion.tlayer]
    for i in range(layer.natoms):
        p = layer.atom[i]
        b[i] = layer.N[i] * cross_section.get_cross(ion, scat[ion.scatindex][p])
        cross += b[i]

    rcross = random.rnd(0.0, cross, c.RndPeriod.RND_OPEN)
    # rcross = 17.554089969710095  # Temporarily value for debugging  # TODO: Remove once RNG is the same as in the C code
    i = 0
    while i < layer.natoms and rcross >= 0.0:
        rcross -= b[i]
        i += 1
    if i < 1 or i > layer.natoms:
        raise IonSimulationError("Scattering atom calculated wrong")
    i -= 1

    ion.opt.y = math.sqrt(-rcross / (math.pi * layer.N[i])) / scat[ion.scatindex][snext.natom].a
    snext.d = -math.log(random.rnd(0.0, 1.0, c.RndPeriod.RND_RIGHT)) / cross


def move_ion(g: o.Global, ion: o.Ion, target: o.Target, snext: o.SNext) -> c.ScatteringType:
    raise NotImplementedError


def inter_sto(stop: o.Target_sto, vel: float, mode: c.IonMode) -> float:
    raise NotImplementedError


def ion_finished(g: o.Global, ion: o.Ion, target: o.Target) -> c.IonStatus:
    raise NotImplementedError


def mc_scattering(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target, detector: o.Detector,
                  scat: o.Scattering, snext: o.SNext) -> bool:
    raise NotImplementedError


def get_angle(scat: o.Scattering, ion: o.Ion) -> float:
    raise NotImplementedError


def ion_rotate(p: o.Ion, cos_theta: float, fii: float) -> None:
    raise NotImplementedError


# TODO: dreclayer is a pointer
def recdist_crossing(g: o.Global, ion: o.Ion, target: o.Target, dist: float, dreclayer: float) -> bool:
    raise NotImplementedError


def recdist_nonzero(g: o.Global, ion: o.Ion, target: o.Target, drec: float) -> bool:
    raise NotImplementedError


def get_reclayer(g: o.Global, target: o.Target, p: o.Point) -> int:
    raise NotImplementedError


