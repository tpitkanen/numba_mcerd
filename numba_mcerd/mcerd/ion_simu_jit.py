import math
from typing import List, Tuple

import numba as nb
import numpy as np

from numba_mcerd.mcerd import random_jit, cross_section_jit, enums
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.mcerd.constants as c


class IonSimulationError(Exception):
    """Error in ion simulation"""


@nb.njit(cache=True)
def create_ion(g: od.Global, ion: od.Ion, target: od.Target) -> None:
    """Calculate parameters for ion (primary or secondary)

    About coordinate systems:
    - laboratory coordinates: z-axis is along the beam line
    - target: z-axis is along the target normal
    - ion: z-axis is along the direction of ion movement
    - detector: z-axis is along the detector direction
    """
    if g.simtype != enums.SimType.ERD.value and g.simtype != enums.SimType.RBS.value:
        raise IonSimulationError("Unsupported simulation type")

    x = random_jit.rnd(-g.bspot.x, g.bspot.x, enums.RndPeriod.CLOSED.value)
    y = random_jit.rnd(-g.bspot.y, g.bspot.y, enums.RndPeriod.CLOSED.value)
    if g.beamangle > 0:
        z = x / math.tan(c.C_PI / 2.0 - g.beamangle)
    else:
        z = 0.0

    ion.lab.p.x = x
    ion.lab.p.y = y
    ion.lab.p.z = z

    if g.simstage == enums.SimStage.REAL.value and g.cion % (g.nscale + 1) == 0:
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
    ion.type = enums.IonType.PRIMARY.value
    ion.status = enums.IonStatus.NOT_FINISHED.value

    ion.lab.theta = g.beamangle
    ion.lab.fii = c.C_PI

    # if __debug__:
    #     debug.print_ion_position(g, ion, "L", enums.SimStage.ANY)


@nb.njit(cache=True)
def move_target(target: od.Target) -> None:
    raise NotImplementedError


# TODO: scat: List[List[od.Scattering]] causes an error:
# TypeError: Parameters to generic types must be types. Got dtype(...)
# TODO: Use numpy.typing for scat array
@nb.njit(cache=True)
def next_scattering(g: od.Global, ion: od.Ion, target: od.Target,
                    scat: np.ndarray, snext: od.SNext) -> None:
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
        b[i] = layer.N[i] * cross_section_jit.get_cross(ion, scat[ion.scatindex][p])
        cross += b[i]

    rcross = random_jit.rnd(0.0, cross, enums.RndPeriod.OPEN)
    i = 0
    while i < layer.natoms and rcross >= 0.0:
        rcross -= b[i]
        i += 1
    if i < 1 or i > layer.natoms:
        raise IonSimulationError("Scattering atom calculated wrong")
    i -= 1

    snext.natom = layer.atom[i]

    ion.opt.y = math.sqrt(-rcross / (c.C_PI * layer.N[i])) / scat[ion.scatindex][snext.natom].a
    snext.d = -math.log(random_jit.rnd(0.0, 1.0, enums.RndPeriod.RIGHT)) / cross


@nb.njit(cache=True)
def move_ion(g: od.Global, ion: od.Ion, target: od.Target, snext: od.SNext) -> int:  # enums.ScatteringType
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
    cross_layer = cross_recdist = sto_dec = False
    sc = enums.ScatteringType.MC_SCATTERING.value

    d = snext.d
    layer = target.layer[ion.tlayer]

    nextp = oj.Point()
    nextp.x = d * ion.opt.sin_theta * math.cos(ion.fii) + ion.p.x
    nextp.y = d * ion.opt.sin_theta * math.sin(ion.fii) + ion.p.y
    nextp.z = d * ion.opt.cos_theta + ion.p.z

    nextz = nextp.z

    if g.rough and ion.tlayer < target.ntarget:
        raise NotImplementedError
    else:
        if nextz < layer.dlow or nextz > layer.dhigh:
            # TODO: cross_layer is probably an enum
            sc = enums.ScatteringType.NO_SCATTERING.value
            if nextz < layer.dlow:
                d = math.fabs((layer.dlow - ion.p.z) / ion.opt.cos_theta)
                cross_layer = -1
            else:
                d = math.fabs((ion.p.z - layer.dhigh) / ion.opt.cos_theta)
                cross_layer = 1

    if ion.type == enums.IonType.PRIMARY.value and 0 < ion.tlayer < target.ntarget:
        cross_recdist, dreclayer = recdist_crossing(g, ion, target, d)
        if cross_recdist:
            cross_layer = False
            d = dreclayer
            sc = enums.ScatteringType.NO_SCATTERING.value
        drec = -math.log(random_jit.rnd(0.0, 1.0, enums.RndPeriod.CLOSED.value))
        if g.recwidth == enums.RecWidth.WIDE.value or g.simstage == enums.SimStage.PRE.value:
            drec *= ion.effrecd / (math.cos(g.beamangle) * g.nrecave)
            ion.wtmp = -1.0
        else:
            n = ion.tlayer
            recang = (target.recpar[n].x * ion.p.z + target.recpar[n].y)**2
            ion.wtmp = target.angave / recang
            # TODO: This may be incorrect
            drec *= (ion.effrecd
                     / (math.cos(g.beamangle) * g.nrecave)
                     / (recang / target.angave))
        if drec < d:
            nz = recdist_nonzero(g, ion, target, drec)
            if nz:  # Recoiling event
                d = drec
                sc = enums.ScatteringType.ERD_SCATTERING.value
                cross_layer = False
                cross_recdist = False

    vel1 = math.sqrt(2.0 * ion.E / ion.A)
    sto1 = inter_sto(layer.sto[ion.scatindex], vel1, enums.IonMode.STOPPING.value)

    dE = sto1 * d

    vel2 = math.sqrt(2.0 * (max(ion.E - dE, 0.0) / ion.A))
    sto2 = inter_sto(layer.sto[ion.scatindex], vel2, enums.IonMode.STOPPING.value)

    while math.fabs(sto1 - sto2) / sto1 > c.MAXELOSS or dE >= ion.E:
        sc = enums.ScatteringType.NO_SCATTERING.value
        cross_layer = False
        sto_dec = True
        d /= 2.0  # TODO: Could use *= 0.5
        dE = sto1 * d
        vel2 = math.sqrt(2.0 * (max(ion.E - dE, 0.0) / ion.A))
        sto2 = inter_sto(layer.sto[ion.scatindex], vel2, enums.IonMode.STOPPING.value)

    stopping = 0.5 * (sto1 + sto2)
    vel = 0.5 * (vel1 + vel2)

    if sc == enums.ScatteringType.NO_SCATTERING.value:
        d += 0.01 * c.C_ANGSTROM  # To make sure that the layer surface is crossed

    if cross_layer:
        ion.tlayer += cross_layer

    # TODO: Copy comments here

    straggling = math.sqrt(inter_sto(layer.sto[ion.type], vel, enums.IonMode.STRAGGLING.value) * d)
    straggling *= random_jit.gaussian()

    eloss = d * stopping
    if math.fabs(straggling) < eloss:
        eloss += straggling
    elif not ion.scale:
        eloss += eloss * straggling / math.fabs(straggling)

    if ion.scale:  # No energy loss for scaling ions
        eloss = 0.0

    ion.E -= eloss
    ion.E = max(ion.E, 0.001 * c.C_EV)

    ion.p.x += d * ion.opt.sin_theta * math.cos(ion.fii)
    ion.p.y += d * ion.opt.sin_theta * math.sin(ion.fii)
    ion.p.z += d * ion.opt.cos_theta

    ion.time += d / vel

    if cross_layer and ion.tlayer > target.ntarget:
        ion.status = enums.IonStatus.FIN_OUT_DET.value

    if cross_layer and ion.type == enums.IonType.SECONDARY.value and ion.tlayer == target.ntarget:
        ion.status = enums.IonStatus.FIN_RECOIL.value

    ion_finished(g, ion, target)

    return sc


# TODO: int instead of enum?
@nb.njit(cache=True)
def inter_sto(stop: od.Target_sto, vel: float, mode: enums.IonMode) -> float:
    """Interpolate the electronic stopping power or straggling value.

    stop.sto or stop.stragg must be equally spaced.

    Args:
        stop: container for calculated stopping values
        vel: current ion velocity to use for interpolation
        mode: whether to calculate stopping or straggling

    Returns:
        Interpolated value
    """
    d = stop.stodiv
    assert stop.n_sto > 0

    if vel >= stop.vel[stop.n_sto - 1]:
        # TODO
        # logging.warning(f"Ion velocity '{vel}' exceeds the maximum velocity of stopping power table")
        if mode == enums.IonMode.STOPPING.value:
            return stop.sto[stop.n_sto - 1]
        return stop.stragg[stop.n_sto - 1]
    if vel <= stop.vel[0]:
        if mode == enums.IonMode.STOPPING.value:
            return stop.sto[0]
        return stop.stragg[0]

    i = int(vel * d)
    assert 0 <= i < stop.n_sto - 1
    vlow = stop.vel[i]
    if mode == enums.IonMode.STOPPING.value:
        slow = stop.sto[i]
        shigh = stop.sto[i + 1]
        assert shigh > 0
    else:
        slow = stop.stragg[i]
        shigh = stop.stragg[i + 1]

    sto = slow + (shigh - slow) * (vel - vlow) * d
    return sto


@nb.njit(cache=True)
def ion_finished(g: od.Global, ion: od.Ion, target: od.Target) -> int:  # enums.IonStatus:
    """Check finish conditions for ion and decide whether to finish it"""
    if ion.E < g.emin:
        ion.status = enums.IonStatus.FIN_STOP.value

    PRIMARY = enums.IonType.PRIMARY.value
    SECONDARY = enums.IonType.SECONDARY.value

    if ion.type == PRIMARY and ion.tlayer >= target.ntarget:
        ion.status = enums.IonStatus.FIN_TRANS.value

    if ion.tlayer < 0:
        if ion.type == PRIMARY:
            ion.status = enums.IonStatus.FIN_BS.value
        else:
            ion.status = enums.IonStatus.FIN_RECOIL.value

    if g.rough:
        raise NotImplementedError
    else:
        if ion.type == PRIMARY and ion.p.z > target.recmaxd:
            ion.status = enums.IonStatus.FIN_MAXDEPTH.value
        if ion.type == PRIMARY and ion.scale and ion.p.z > c.SCALE_DEPTH:
            ion.status = enums.IonStatus.FIN_MAXDEPTH.value

    if ion.type == SECONDARY and ion.tlayer >= target.nlayers:
        ion.status = enums.IonStatus.FIN_DET.value

    # Comment in original: Here we should also check the case that the
    # recoil comes out of the detector layer from sides.

    return ion.status


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


@nb.njit(cache=True)
def recdist_crossing(g: od.Global, ion: od.Ion, target: od.Target, dist: float) -> Tuple[bool, float]:
    # TODO: Document what dreclayer is. Probably distance to next recoil layer
    """Check if ion is about to cross a layer in the recoil material distribution.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether a layer is crossed, and dreclayer
    """
    # n = target.nrecdist  # Unused if not g.rough
    d = -1.0
    i = get_reclayer(g, target, ion.p)

    nextp = oj.Point()
    scross = s1cross = s2cross = False  # s1cross and s2cross unused if not g.rough
    if g.rough:
        raise NotImplementedError
    else:
        nextp.z = dist * ion.opt.cos_theta + ion.p.z
        i1 = i
        i2 = get_reclayer(g, target, nextp)
        if i2 > i1:
            d = (target.recdist[i1].x - ion.p.z) / ion.opt.cos_theta
        elif i1 > i2:
            d = (target.recdist[i1 - 1].x - ion.p.z) / ion.opt.cos_theta
        if i1 != i2:
            scross = True

        if d < 0 and scross:
            pass  # TODO: Print information

    return scross, d


@nb.njit(cache=True)
def recdist_nonzero(g: od.Global, ion: od.Ion, target: od.Target, drec: float) -> bool:
    """Check if the recoil material distribution is nonzero at the point
    of the next recoil event.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether material distribution is nonzero
    """
    nextp = oj.Point()

    if g.rough:
        raise NotImplementedError
    else:
        nextp.z = drec * ion.opt.cos_theta + ion.p.z
        i = get_reclayer(g, target, nextp)

    if i == 0 or i >= target.nrecdist:
        return False
    if target.recdist[i - 1].y > 0 or target.recdist[i].y > 0:
        return True
    return False


@nb.njit(cache=True)
def get_reclayer(g: od.Global, target: od.Target, p: od.Point) -> int:
    z = p.z
    n = target.nrecdist

    if g.rough:
        raise NotImplementedError

    i = 0
    while i < n and z > target.recdist[i].x:
        i += 1

    return i
