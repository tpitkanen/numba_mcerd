import logging
import math
from typing import List, Tuple

import numpy as np

from numba_mcerd.mcerd import cross_section, enums
from numba_mcerd.config import rand
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.constants as c


class IonSimulationError(Exception):
    """Error in ion simulation"""


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
    if g.simtype != enums.SimType.SIM_ERD and g.simtype != enums.SimType.SIM_RBS:
        raise IonSimulationError("Unsupported simulation type")

    x = rand.rnd(-g.bspot.x, g.bspot.x, enums.RndPeriod.RND_CLOSED)
    y = rand.rnd(-g.bspot.y, g.bspot.y, enums.RndPeriod.RND_CLOSED)
    if g.beamangle > 0:
        z = x / math.tan(c.C_PI / 2.0 - g.beamangle)
    else:
        z = 0.0

    ion.lab.p = o.Point(x, y, z)

    if g.simstage == enums.SimStage.REALSIMULATION and g.cion % (g.nscale + 1) == 0:
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
    ion.type = enums.IonType.PRIMARY
    ion.status = enums.IonStatus.NOT_FINISHED

    ion.lab.theta = g.beamangle
    ion.lab.fii = c.C_PI

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

    rcross = rand.rnd(0.0, cross, enums.RndPeriod.RND_OPEN)
    i = 0
    while i < layer.natoms and rcross >= 0.0:
        rcross -= b[i]
        i += 1
    if i < 1 or i > layer.natoms:
        raise IonSimulationError("Scattering atom calculated wrong")
    i -= 1

    snext.natom = layer.atom[i]

    ion.opt.y = math.sqrt(-rcross / (c.C_PI * layer.N[i])) / scat[ion.scatindex][snext.natom].a
    snext.d = -math.log(rand.rnd(0.0, 1.0, enums.RndPeriod.RND_RIGHT)) / cross


def move_ion(g: o.Global, ion: o.Ion, target: o.Target, snext: o.SNext) -> enums.ScatteringType:
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
    sc = enums.ScatteringType.MC_SCATTERING

    d = snext.d
    layer = target.layer[ion.tlayer]

    nextp = o.Point()
    nextp.x = d * ion.opt.sin_theta * math.cos(ion.fii) + ion.p.x
    nextp.y = d * ion.opt.sin_theta * math.sin(ion.fii) + ion.p.y
    nextp.z = d * ion.opt.cos_theta + ion.p.z

    nextz = nextp.z

    if g.rough and ion.tlayer < target.ntarget:
        raise NotImplementedError
    else:
        if nextz < layer.dlow or nextz > layer.dhigh:
            # TODO: cross_layer is probably an enum
            sc = enums.ScatteringType.NO_SCATTERING
            if nextz < layer.dlow:
                d = math.fabs((layer.dlow - ion.p.z) / ion.opt.cos_theta)
                cross_layer = -1
            else:
                d = math.fabs((ion.p.z - layer.dhigh) / ion.opt.cos_theta)
                cross_layer = 1

    if ion.type == enums.IonType.PRIMARY and 0 < ion.tlayer < target.ntarget:
        cross_recdist, dreclayer = recdist_crossing(g, ion, target, d)
        if cross_recdist:
            cross_layer = False
            d = dreclayer
            sc = enums.ScatteringType.NO_SCATTERING
        drec = -math.log(rand.rnd(0.0, 1.0, enums.RndPeriod.RND_CLOSED))
        if g.recwidth == enums.RecWidth.REC_WIDE or g.simstage == enums.SimStage.PRESIMULATION:
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
                sc = enums.ScatteringType.ERD_SCATTERING
                cross_layer = False
                cross_recdist = False

    vel1 = math.sqrt(2.0 * ion.E / ion.A)
    sto1 = inter_sto(layer.sto[ion.scatindex], vel1, enums.IonMode.STOPPING)

    dE = sto1 * d

    vel2 = math.sqrt(2.0 * (max(ion.E - dE, 0.0) / ion.A))
    sto2 = inter_sto(layer.sto[ion.scatindex], vel2, enums.IonMode.STOPPING)

    while math.fabs(sto1 - sto2) / sto1 > c.MAXELOSS or dE >= ion.E:
        sc = enums.ScatteringType.NO_SCATTERING
        cross_layer = False
        sto_dec = True
        d /= 2.0  # TODO: Could use *= 0.5
        dE = sto1 * d
        vel2 = math.sqrt(2.0 * (max(ion.E - dE, 0.0) / ion.A))
        sto2 = inter_sto(layer.sto[ion.scatindex], vel2, enums.IonMode.STOPPING)

    stopping = 0.5 * (sto1 + sto2)
    vel = 0.5 * (vel1 + vel2)

    if sc == enums.ScatteringType.NO_SCATTERING:
        d += 0.01 * c.C_ANGSTROM  # To make sure that the layer surface is crossed

    if cross_layer:
        ion.tlayer += cross_layer

    # TODO: Copy comments here

    straggling = math.sqrt(inter_sto(layer.sto[ion.type.value], vel, enums.IonMode.STRAGGLING) * d)
    straggling *= rand.gaussian()

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
        ion.status = enums.IonStatus.FIN_OUT_DET

    if cross_layer and ion.type == enums.IonType.SECONDARY and ion.tlayer == target.ntarget:
        ion.status = enums.IonStatus.FIN_RECOIL

    ion_finished(g, ion, target)

    return sc


def inter_sto(stop: o.Target_sto, vel: float, mode: enums.IonMode) -> float:
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
        logging.warning(f"Ion velocity '{vel}' exceeds the maximum velocity of stopping power table")
        if mode == enums.IonMode.STOPPING:
            return stop.sto[stop.n_sto - 1]
        return stop.stragg[stop.n_sto - 1]
    if vel <= stop.vel[0]:
        if mode == enums.IonMode.STOPPING:
            return stop.sto[0]
        return stop.stragg[0]

    i = int(vel * d)
    assert 0 <= i < stop.n_sto - 1
    vlow = stop.vel[i]
    if mode == enums.IonMode.STOPPING:
        slow = stop.sto[i]
        shigh = stop.sto[i + 1]
        assert shigh > 0
    else:
        slow = stop.stragg[i]
        shigh = stop.stragg[i + 1]

    sto = slow + (shigh - slow) * (vel - vlow) * d
    return sto


def ion_finished(g: o.Global, ion: o.Ion, target: o.Target) -> enums.IonStatus:
    """Check finish conditions for ion and decide whether to finish it"""
    if ion.E < g.emin:
        ion.status = enums.IonStatus.FIN_STOP

    PRIMARY = enums.IonType.PRIMARY
    SECONDARY = enums.IonType.SECONDARY

    if ion.type == PRIMARY and ion.tlayer >= target.ntarget:
        ion.status = enums.IonStatus.FIN_TRANS

    if ion.tlayer < 0:
        if ion.type == PRIMARY:
            ion.status = enums.IonStatus.FIN_BS
        else:
            ion.status = enums.IonStatus.FIN_RECOIL

    if g.rough:
        raise NotImplementedError
    else:
        if ion.type == PRIMARY and ion.p.z > target.recmaxd:
            ion.status = enums.IonStatus.FIN_MAXDEPTH
        if ion.type == PRIMARY and ion.scale and ion.p.z > c.SCALE_DEPTH:
            ion.status = enums.IonStatus.FIN_MAXDEPTH

    if ion.type == SECONDARY and ion.tlayer >= target.nlayers:
        ion.status = enums.IonStatus.FIN_DET

    # Comment in original: Here we should also check the case that the
    # recoil comes out of the detector layer from sides.

    return ion.status


# detector would be used in:
# if g.cascades and E_recoil > 2.0 * g.emin
def mc_scattering(g: o.Global, ion: o.Ion, recoil: o.Ion, target: o.Target, detector: o.Detector,
                  scat: List[List[o.Scattering]], snext: o.SNext) -> bool:
    """Normal elastic scattering from a potential during the slowing down process"""
    recoils = False

    g.nmc += 1  # TODO: Problematic for multi-threading
    if ion.scale:
        return False

    targetA = target.ele[snext.natom].A  # TODO in original code
    targetZ = target.ele[snext.natom].Z

    s = scat[ion.scatindex][snext.natom]  # TODO: "!!!" in original code, and give a better name

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
        # TODO: Better message
        logging.warning("Scattering with M1>M2 and scattering angle exceeds asin(M2/M1). Physics fails.")

    # This might actually be -0.5*(PI-angle)
    theta_recoil = 0.5 * (c.C_PI - angle)
    if theta_recoil > 0.5 * c.C_PI:
        # TODO: Values
        logging.warning("Unphysically scattered backwards.")
    cos_theta_recoil = math.cos(theta_recoil)  # cos(-x) = cos(x) so this won't be affected anyway
    # sin_theta_recoil = math.sin(theta_recoil)  # Unused

    Ef = ((math.sqrt(targetA**2 - (ion.A * sin_theta)**2) + ion.A * cos_theta)
          / (ion.A + targetA))
    E_recoil = (1.0 - Ef) * ion.E
    if g.cascades and E_recoil > 2.0 * g.emin:  # TODO: 2.0 is a workaround for newly created recoil
        # Enough energy to be treated as a "true" recoil, the same way as the original particle
        raise NotImplementedError

    # ifdef SCAT_ANAL
    # ifdef NO_ETRANS
    # ifdef NO_PRI_ANGLES
    # ifdef NO_SEC_ANGLES

    # ifndef NO_RBS_SCATLIMIT
    if (g.simtype == enums.SimType.SIM_RBS
            and ion.type == enums.IonType.SECONDARY
            and ion.tlayer < target.ntarget):  # Limited to sample
        raise NotImplementedError

    # All nuclear energy losses are tabulated, including the prime recoil from a cascade (above)
    ion.E_nucl_loss_det += (1.0 - Ef) * ion.E
    ion.E *= Ef
    ion.opt.e = ion.E * s.E2eps

    fii = rand.rnd(0.0, 2.0 * c.C_PI, enums.RndPeriod.RND_RIGHT)
    ion_rotate(ion, cos_theta, fii)
    if recoils:
        ion_rotate(recoil, cos_theta_recoil, math.fmod(fii + c.C_PI, 2.0 * c.C_PI))
    ion.nsct += 1
    return recoils


def get_angle(scat: o.Scattering, ion: o.Ion) -> float:
    """Interpolate the cosine of the scattering angle according to the
     ion energy and impact parameter."""

    y = ion.opt.y
    e = ion.opt.e

    i = int(((math.log(e) - scat.logemin) * scat.logediv) + 1)
    j = int(((math.log(y) - scat.logymin) * scat.logydiv) + 1)

    if i > c.EPSNUM - 2:
        logging.warning("Energy exceeds the maximum of the scattering table energy")
        return 0.0
    if i < 1:
        logging.warning("Energy is below the minimum of the scattering table energy")
        return 0.0
    if j < 1:
        logging.warning("Empact parameter is below the minimum of the scattering table value")
        # formatted number print
        return c.C_PI  # TODO in original: PI or zero?
    if j > c.YNUM - 2:
        logging.warning("Impact parameter exceeds the maximum of the scattering table value")
        # formatted number print
        return 0.0

    ylow = scat.angle[0][j]
    yhigh = scat.angle[0][j + 1]
    elow = scat.angle[i][0]
    ehigh = scat.angle[i + 1][0]

    angle11 = scat.angle[i][j]
    angle12 = scat.angle[i][j + 1]
    angle21 = scat.angle[i + 1][j]
    angle22 = scat.angle[i + 1][j + 1]

    tmp0 = 1.0 / (yhigh - ylow)
    tmp1 = angle11 + (y - ylow) * (angle12 - angle11) * tmp0
    tmp2 = angle21 + (y - ylow) * (angle22 - angle21) * tmp0

    angle = tmp1 + (e - elow) * (tmp2 - tmp1) / (ehigh - elow)
    return angle


def ion_rotate(p: o.Ion, cos_theta: float, fii: float) -> None:
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


def recdist_crossing(g: o.Global, ion: o.Ion, target: o.Target, dist: float) -> Tuple[bool, float]:
    # TODO: Document what dreclayer is. Probably distance to next recoil layer
    """Check if ion is about to cross a layer in the recoil material distribution.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether a layer is crossed, and dreclayer
    """
    # n = target.nrecdist  # Unused if not g.rough
    d = -1.0
    i = get_reclayer(g, target, ion.p)

    nextp = o.Point()
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


def recdist_nonzero(g: o.Global, ion: o.Ion, target: o.Target, drec: float) -> bool:
    """Check if the recoil material distribution is nonzero at the point
    of the next recoil event.

    Rough and non-rough surfaces are handled separately.

    Returns:
        Whether material distribution is nonzero
    """
    nextp = o.Point()

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


def get_reclayer(g: o.Global, target: o.Target, p: o.Point) -> int:
    z = p.z
    n = target.nrecdist

    if g.rough:
        raise NotImplementedError

    i = 0
    while i < n and z > target.recdist[i].x:
        i += 1

    return i
