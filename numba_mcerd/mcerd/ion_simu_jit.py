import math
from typing import List, Tuple

import numba as nb
import numpy as np

from numba_mcerd.mcerd import random_jit, cross_section
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.constants as c


class IonSimulationError(Exception):
    """Error in ion simulation"""


@nb.njit()
def create_ion(g: oj.Global, ion: oj.Ion, target: oj.Target) -> None:
    """Calculate parameters for ion (primary or secondary)

    About coordinate systems:
    - laboratory coordinates: z-axis is along the beam line
    - target: z-axis is along the target normal
    - ion: z-axis is along the direction of ion movement
    - detector: z-axis is along the detector direction
    """
    # assert g.simtype == c.SimType.SIM_ERD or g.simtype == c.SimType.SIM_RBS
    if g.simtype != c.SimType.SIM_ERD.value and g.simtype != c.SimType.SIM_RBS.value:
        raise IonSimulationError("Unsupported simulation type")

    x = random_jit.rnd(-g.bspot.x, g.bspot.x, c.RndPeriod.RND_CLOSED.value)
    y = random_jit.rnd(-g.bspot.y, g.bspot.y, c.RndPeriod.RND_CLOSED.value)
    if g.beamangle > 0:
        z = x / math.tan(c.C_PI / 2.0 - g.beamangle)
    else:
        z = 0.0

    # TODO: Is there an easier way to do this?
    ion.lab.p = oj.Point()
    ion.lab.p.x = x
    ion.lab.p.y = y
    ion.lab.p.z = z

    if g.simstage == c.SimStage.REALSIMULATION.value and g.cion % (g.nscale + 1) == 0:
        # TODO: Calculating a random point goes to waste here. Move
        #       random point calculation to the else branch
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
    ion.type = c.IonType.PRIMARY.value
    ion.status = c.IonStatus.NOT_FINISHED.value

    ion.lab.theta = g.beamangle
    ion.lab.fii = c.C_PI

    # if __debug__:
    #     debug.print_ion_position(g, ion, "L", c.SimStage.ANYSIMULATION)


def move_target(target: o.Target) -> None:
    raise NotImplementedError


@nb.njit()
def next_scattering(g: oj.Global, ion: oj.Ion, target: oj.Target,
                    scat: List[List[o.Scattering]], snext: oj.SNext) -> None:
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

    rcross = random_jit.rnd(0.0, cross, c.RndPeriod.RND_OPEN)
    i = 0
    while i < layer.natoms and rcross >= 0.0:
        rcross -= b[i]
        i += 1
    if i < 1 or i > layer.natoms:
        raise IonSimulationError("Scattering atom calculated wrong")
    i -= 1

    snext.natom = layer.atom[i]

    ion.opt.y = math.sqrt(-rcross / (c.C_PI * layer.N[i])) / scat[ion.scatindex][snext.natom].a
    snext.d = -math.log(random_jit.rnd(0.0, 1.0, c.RndPeriod.RND_RIGHT)) / cross


def move_ion(g: o.Global, ion: o.Ion, target: o.Target, snext: o.SNext) -> c.ScatteringType:
    # TODO: Replace copy-paste documentation
    """Move ion to the next point, which is the closest of following
      i) next MC-scattering point, distance already calculated
     ii) next crossing of the target layer, distance calculated
         using surface_crossing -subroutine
    iii) next ERD scattering, in case of a primary ion, random distance
         to the scattering point is calculated here according
         to the simulation mode (wide or narrow), which is accepted only
         if recoil material distribu<tion value is nonzero in that point
     iv) point where the relative electronic stopping power has
         changed more than MAXELOSS

    To make things fast we do not evaluate all these values at the
    beginning but with a stepwise procedure
    (since we are in the innermost loop we have to worry about the speed)
    """
    raise NotImplementedError


def inter_sto(stop: o.Target_sto, vel: float, mode: c.IonMode) -> float:
    """Interpolate the electronic stopping power or straggling value.

    stop.sto or stop.stragg must be equally spaced.

    Args:
        stop: container for calculated stopping values
        vel: current ion velocity to use for interpolation
        mode: whether to calculate stopping or straggling

    Returns:
        Interpolated value
    """
    raise NotImplementedError


def ion_finished(g: o.Global, ion: o.Ion, target: o.Target) -> c.IonStatus:
    """Check finish conditions for ion and decide whether to finish it"""
    raise NotImplementedError


# detector would be used in:
# if g.cascades and E_recoil > 2.0 * g.emin
def mc_scattering(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target, detector: o.Detector,
                  scat: List[List[o.Scattering]], snext: o.SNext) -> bool:
    """Normal elastic scattering from a potential during the slowing down process"""
    raise NotImplementedError


def get_angle(scat: o.Scattering, ion: o.Ion) -> float:
    """Interpolate the cosine of the scattering angle according to the
     ion energy and impact parameter."""
    raise NotImplementedError



def ion_rotate(p: o.Ion, cos_theta: float, fii: float) -> None:
    # TODO: Change this to use the general rotate function. Differences:
    #       - cos_theta is an argument instead of cos, angles are in p
    #       - rz is saved
    #       - p.opt is read and written
    #       - inconsistent end checks
    raise NotImplementedError



def recdist_crossing(g: o.Global, ion: o.Ion, target: o.Target, dist: float) -> Tuple[bool, float]:
    # TODO: Document what dreclayer is. Probably distance to next recoil layer
    """Check if ion is about to cross a layer in the recoil material distribution.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether a layer is crossed, and dreclayer
    """
    raise NotImplementedError


def recdist_nonzero(g: o.Global, ion: o.Ion, target: o.Target, drec: float) -> bool:
    """Check if the recoil material distribution is nonzero at the point
    of the next recoil event.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether material distribution is nonzero
    """
    raise NotImplementedError


def get_reclayer(g: o.Global, target: o.Target, p: o.Point) -> int:
    raise NotImplementedError



