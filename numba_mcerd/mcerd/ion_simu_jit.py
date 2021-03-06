import math
from typing import Tuple, Any

import numba as nb
import numpy as np
from numba import cuda, types

from numba_mcerd import logging_jit
from numba_mcerd.mcerd import random_jit, cross_section_jit, enums, random_cuda
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.constants as c


class IonSimulationError(Exception):
    """Error in ion simulation"""


@nb.njit(cache=True, nogil=True)
def create_ion(g: oj.Global, ion: oj.Ion, target: oj.Target) -> None:
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


@nb.njit(cache=True, nogil=True)
def move_target(target: oj.Target) -> None:
    raise NotImplementedError


# TODO: scat: List[List[oj.Scattering]] causes an error:
# TypeError: Parameters to generic types must be types. Got dtype(...)
# TODO: Use numpy.typing for scat array
@nb.njit(cache=True, nogil=True)
def next_scattering(g: oj.Global, ion: oj.Ion, target: oj.Target,
                    scat: np.ndarray, snext: oj.SNext) -> None:
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
        b[i] = layer.N[i] * cross_section_jit.get_cross(ion, scat[ion.scatindex, p])
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

    ion.opt.y = math.sqrt(-rcross / (c.C_PI * layer.N[i])) / scat[ion.scatindex, snext.natom].a
    snext.d = -math.log(random_jit.rnd(0.0, 1.0, enums.RndPeriod.RIGHT)) / cross


# c.MAXATOMS doesn't work in cuda.local.array(c.MAXATOMS, dtype=types.float64) for some reason
c_MAXATOMS = c.MAXATOMS


@cuda.jit()
def next_scattering_cuda(g: oj.Global, ion: oj.Ion, target: oj.Target,
                         scat: np.ndarray, snext: oj.SNext, rng_states: Any) -> None:
    if g.nomc:
        snext.d = 100.0 * c.C_NM
        return

    thread_id = cuda.grid(1)

    cross = 0.0
    # TODO: Array creation is slow in CUDA, is there a way to avoid this?
    b = cuda.local.array(c_MAXATOMS, dtype=types.float64)
    layer = target.layer[ion.tlayer]
    for i in range(layer.natoms):
        p = layer.atom[i]
        b[i] = layer.N[i] * cross_section_jit.get_cross_cuda(ion, scat[ion.scatindex, p])
        cross += b[i]

    rcross = random_cuda.rnd(0.0, cross, rng_states, thread_id)
    i = 0
    while i < layer.natoms and rcross >= 0.0:
        rcross -= b[i]
        i += 1
    # if i < 1 or i > layer.natoms:
    #     raise IonSimulationError("Scattering atom calculated wrong")
    i -= 1

    snext.natom = layer.atom[i]

    ion.opt.y = math.sqrt(-rcross / (c.C_PI * layer.N[i])) / scat[ion.scatindex, snext.natom].a
    snext.d = -math.log(random_cuda.rnd(0.0, 1.0, rng_states, thread_id)) / cross


@nb.njit(cache=True, nogil=True)
def move_ion(g: oj.Global, ion: oj.Ion, target: oj.Target, snext: oj.SNext) -> int:  # enums.ScatteringType
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

    nextx = d * ion.opt.sin_theta * math.cos(ion.fii) + ion.p.x
    nexty = d * ion.opt.sin_theta * math.sin(ion.fii) + ion.p.y
    nextz = d * ion.opt.cos_theta + ion.p.z

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
@nb.njit(cache=True, nogil=True)
def inter_sto(stop: oj.Target_sto, vel: float, mode: enums.IonMode) -> float:
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
        # logging_jit.warning(f"Ion velocity={float(vel)} exceeds the maximum velocity of stopping power table")
        logging_jit.warning(f"Ion velocity exceeds the maximum velocity of stopping power table")
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


@nb.njit(cache=True, nogil=True)
def ion_finished(g: oj.Global, ion: oj.Ion, target: oj.Target) -> int:  # enums.IonStatus:
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
@nb.njit(cache=True, nogil=True)
def mc_scattering(g: oj.Global, ion: oj.Ion, recoil: oj.Ion, target: oj.Target,
                  detector: oj.Detector, scat: np.ndarray, snext: oj.SNext) -> bool:
    """Normal elastic scattering from a potential during the slowing down process"""
    recoils = False

    g.nmc += 1  # TODO: Problematic for multi-threading
    if ion.scale:
        return False

    targetA = target.ele[snext.natom].A  # TODO in original code
    targetZ = target.ele[snext.natom].Z

    # TODO: This could be sliced in the main loop
    s = scat[ion.scatindex, snext.natom]  # TODO: "!!!" in original code, and give a better name

    ion.opt.e = ion.E * s.E2eps
    if ion.E < g.emin:
        return False

    n = ion.A / targetA

    angle = get_angle(s, ion)
    cos_theta_cm = math.cos(angle)
    sin_theta_cm = math.sin(angle)

    theta = math.atan2(sin_theta_cm, cos_theta_cm + n)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    if ion.A > targetA and theta > math.asin(1.0 / n):
        # logging_jit.warning(f"Scattering with M1={float(ion.A)} > M2={float(targetA)} and scattering angle exceeds asin(M2/M1). Physics fails.")
        logging_jit.warning(f"Scattering with M1>M2 and scattering angle exceeds asin(M2/M1). Physics fails.")
        pass

    # This might actually be -0.5*(PI-angle)
    theta_recoil = 0.5 * (c.C_PI - angle)
    if theta_recoil > 0.5 * c.C_PI:
        # logging_jit.warning(f"Unphysically scattered backwards with theta_recoil={float(theta_recoil)}")
        logging_jit.warning(f"Unphysically scattered backwards")
        pass
    cos_theta_recoil = math.cos(theta_recoil)  # cos(-x) = cos(x) so this won't be affected anyway
    # sin_theta_recoil = math.sin(theta_recoil)  # Unused

    Ef = ((math.sqrt(targetA ** 2 - (ion.A * sin_theta) ** 2) + ion.A * cos_theta) / (ion.A + targetA)) ** 2
    E_recoil = (1.0 - Ef) * ion.E
    if g.cascades and E_recoil > 2.0 * g.emin:  # TODO: 2.0 is a workaround for newly created recoil
        # Enough energy to be treated as a "true" recoil, the same way as the original particle
        raise NotImplementedError

    # ifdef SCAT_ANAL
    # ifdef NO_ETRANS
    # ifdef NO_PRI_ANGLES
    # ifdef NO_SEC_ANGLES

    # ifndef NO_RBS_SCATLIMIT
    if (g.simtype == enums.SimType.RBS
            and ion.type == enums.IonType.SECONDARY
            and ion.tlayer < target.ntarget):  # Limited to sample
        raise NotImplementedError

    # All nuclear energy losses are tabulated, including the prime recoil from a cascade (above)
    ion.E_nucl_loss_det += (1.0 - Ef) * ion.E
    ion.E *= Ef
    ion.opt.e = ion.E * s.E2eps

    fii = random_jit.rnd(0.0, 2.0 * c.C_PI, enums.RndPeriod.RIGHT)
    ion_rotate(ion, cos_theta, fii)
    if recoils:
        ion_rotate(recoil, cos_theta_recoil, math.fmod(fii + c.C_PI, 2.0 * c.C_PI))
    ion.nsct += 1
    return recoils


@nb.njit(cache=True, nogil=True)
def get_angle(scat: oj.Scattering, ion: oj.Ion) -> float:
    """Interpolate the cosine of the scattering angle according to the
     ion energy and impact parameter."""
    y = ion.opt.y
    e = ion.opt.e

    i = int(((math.log(e) - scat.logemin) * scat.logediv) + 1)
    j = int(((math.log(y) - scat.logymin) * scat.logydiv) + 1)

    if i > c.EPSNUM - 2:
        # logging_jit.warning(f"Energy i={int(i)} exceeds the maximum of the scattering table energy ({c.EPSNUM - 2})")
        logging_jit.warning(f"Energy exceeds the maximum of the scattering table energy")
        return 0.0
    if i < 1:
        # logging_jit.warning(f"Energy i={int(i)} is below the minimum of the scattering table energy (1)")
        logging_jit.warning(f"Energy is below the minimum of the scattering table energy")
        return 0.0
    if j < 1:
        # logging_jit.warning(f"Impact parameter j={int(j)} is below the minimum of the scattering table value (1)")
        logging_jit.warning(f"Impact parameter is below the minimum of the scattering table value")
        return c.C_PI  # TODO in original: PI or zero?
    if j > c.YNUM - 2:
        # logging_jit.warning(f"Impact parameter j={int(j)} exceeds the maximum of the scattering table value ({c.YNUM - 2})")
        logging_jit.warning(f"Impact parameter exceeds the maximum of the scattering table value")
        return 0.0

    ylow = scat.angle[0, j]
    yhigh = scat.angle[0, j + 1]
    elow = scat.angle[i, 0]
    ehigh = scat.angle[i + 1, 0]

    angle11 = scat.angle[i, j]
    angle12 = scat.angle[i, j + 1]
    angle21 = scat.angle[i + 1, j]
    angle22 = scat.angle[i + 1, j + 1]

    tmp0 = 1.0 / (yhigh - ylow)
    tmp1 = angle11 + (y - ylow) * (angle12 - angle11) * tmp0
    tmp2 = angle21 + (y - ylow) * (angle22 - angle21) * tmp0

    angle = tmp1 + (e - elow) * (tmp2 - tmp1) / (ehigh - elow)
    return angle


@nb.njit(cache=True, nogil=True)
def ion_rotate(p: oj.Ion, cos_theta: float, fii: float) -> None:
    # TODO: Change this to use the general rotate function. Differences:
    #       - cos_theta is an argument instead of cos, angles are in p
    #       - rz is saved
    #       - p.opt is read and written
    #       - inconsistent end checks

    sin_theta = math.sqrt(math.fabs(1.0 - cos_theta * cos_theta))
    cos_fii = math.cos(fii)
    sin_fii = math.sin(fii)

    x = sin_theta * cos_fii
    y = sin_theta * sin_fii
    z = cos_theta

    cosa1 = p.opt.cos_theta
    sina1 = math.sqrt(math.fabs(1.0 - cosa1 * cosa1))

    sina2 = math.sin(p.fii + c.C_PI / 2.0)
    cosa2 = math.cos(p.fii + c.C_PI / 2.0)

    cosa3 = cosa2
    sina3 = -sina2

    rx = (x * (cosa3 * cosa2 - cosa1 * sina2 * sina3)
          + y * (-sina3 * cosa2 - cosa1 * sina2 * cosa3)
          + z * sina1 * sina2)

    ry = (x * (cosa3 * sina2 + cosa1 * cosa2 * sina3)
          + y * (-sina3 * sina2 + cosa1 * cosa2 * cosa3)
          - z * sina1 * cosa2)

    rz = (x * sina1 * sina3
          + y * sina1 * cosa3
          + z * cosa1)

    p.opt.cos_theta = rz
    p.opt.sin_theta = math.sqrt(math.fabs(1.0 - rz**2))

    p.theta = math.acos(rz)
    p.fii = math.atan2(ry, rx)
    if p.fii < 0.0:
        p.fii += 2.0 * c.C_PI


@nb.njit(cache=True, nogil=True)
def recdist_crossing(g: oj.Global, ion: oj.Ion, target: oj.Target, dist: float) -> Tuple[bool, float]:
    # TODO: Document what dreclayer is. Probably distance to next recoil layer
    """Check if ion is about to cross a layer in the recoil material distribution.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether a layer is crossed, and dreclayer
    """
    # n = target.nrecdist  # Unused if not g.rough
    d = -1.0
    i = get_reclayer(g, target, ion.p.z)

    scross = s1cross = s2cross = False  # s1cross and s2cross unused if not g.rough
    if g.rough:
        raise NotImplementedError
    else:
        nextz = dist * ion.opt.cos_theta + ion.p.z
        i1 = i
        i2 = get_reclayer(g, target, nextz)
        if i2 > i1:
            d = (target.recdist[i1].x - ion.p.z) / ion.opt.cos_theta
        elif i1 > i2:
            d = (target.recdist[i1 - 1].x - ion.p.z) / ion.opt.cos_theta
        if i1 != i2:
            scross = True

        if d < 0 and scross:
            pass  # TODO: Print information

    return scross, d


@nb.njit(cache=True, nogil=True)
def recdist_nonzero(g: oj.Global, ion: oj.Ion, target: oj.Target, drec: float) -> bool:
    """Check if the recoil material distribution is nonzero at the point
    of the next recoil event.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether material distribution is nonzero
    """
    if g.rough:
        raise NotImplementedError
    else:
        nextz = drec * ion.opt.cos_theta + ion.p.z
        i = get_reclayer(g, target, nextz)

    if i == 0 or i >= target.nrecdist:
        return False
    if target.recdist[i - 1].y > 0 or target.recdist[i].y > 0:
        return True
    return False


@nb.njit(cache=True, nogil=True)
def get_reclayer(g: oj.Global, target: oj.Target, z: float) -> int:
    n = target.nrecdist

    if g.rough:
        raise NotImplementedError

    i = 0
    while i < n and z > target.recdist[i].x:
        i += 1

    return i
