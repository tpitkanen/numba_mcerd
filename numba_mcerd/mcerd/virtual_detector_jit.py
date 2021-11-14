import math

import numba as nb
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.mcerd.objects_jit as oj
from numba_mcerd.mcerd import ion_simu_jit, enums, misc_jit, rotate_jit, erd_detector_jit, random_jit


@nb.njit(cache=True)
def hit_virtual_detector(g: oj.Global, ion: oj.Ion, target: oj.Target, det: oj.Detector,
                         p_virt_lab: oj.Point, p_out_tar: oj.Point) -> None:
    """Calculate a random projection point from virtual detector to
    the real detector, and correct the ion energy according to the
    changed kinematics.
    """
    # For kinematic correction we calculate first the angular change we
    # have to make in order to move from the virtual hit point to the
    # real hit point and then apply this to the original recoiling
    # direction to calculate the new initial recoiling direction. Since
    # the original recoiling direction can be almost anything
    # (especially in the wide mode) it is possible that we end up in an
    # unphysical new recoiling direction. Same angles are used for
    # the kinematic correction and the cross section correction.
    #
    # For stopping correction the energy loss and direct path length in
    # the target material is calculated for the initial ion path
    # hitting to the virtual detector and the energy loss is corrected
    # for the change in the path length to the real hit point.

    # Variable naming convention: first part distinguishes between vectors
    # (v) and points (p), second part (after p or v) gives the direction
    # from the recoiling point [rec = recoil direction, virt = direction to the
    # virtual hit point, real = direction to the real (new) hit point].
    # Last part gives the frame of reference [foil = foil coordinate system,
    # pri = primary ion (at the moment of recoiling) coordinate system,
    # tar = target coordinate system, lab = laboratory coordinate system].
    #
    # *p_virt_lab:   hit point in the virtual foil in the lab. coordinates
    # v_real_foil:   new hit position in the real foil coordinates
    # v_recvirt_pri: recoil vector in primary ion coord. at the recoil moment
    # v_rec_tar:     recoil vector in target ion coordinates at the recoil moment
    # v_rec_lab:     recoil vector in lab. coordinates at the recoil moment
    # v_virt_lab:    vector from the recoil point to the virtual hit point
    # v_real_lab:    vector from the recoil point to the new real hit point
    # v_diff_lab:    angle between old and new hit point from recoil point
    # v_recreal_pri: recoil vector to the new real hit point in pri. ion coord.

    # TODO: Should variable initializations be moved back up?

    # Unused
    # v_recvirt_pri = None
    # v_rec_tar = None
    # v_rec_lab = None

    # theta = fii = dx = dy = r = dE1 = dE2 = dw = dist = eloss = m1 = m2 = 0.0
    err = False
    # err1 = err2 = n = 0

    # m1 = m2 = dE1 = dw = 0.0

    v_real_foil = oj.Vector()

    if det.vfoil.type == enums.FoilType.CIRC:
        r = det.foil[0]["size_"][0] * math.sqrt(random_jit.rnd(0.0, 1.0, enums.RndPeriod.CLOSED))
        fii = random_jit.rnd(0.0, 2.0 * c.C_PI, enums.RndPeriod.CLOSED)
        v_real_foil.p.x = r * math.cos(fii)
        v_real_foil.p.y = r * math.sin(fii)
    elif det.vfoil.type == enums.FoilType.RECT:
        dx = det.foil[0]["size_"][0]
        v_real_foil.p.x = random_jit.rnd(-dx, dx, enums.RndPeriod.CLOSED)
        dy = det.foil[0]["size_"][1]
        v_real_foil.p.y = random_jit.rnd(-dy, dy, enums.RndPeriod.CLOSED)
    v_real_foil.p.z = 0.0
    theta = det.angle
    fii = 0.0

    v_recvirt_pri = ion.hist.ion_recoil
    v_rec_tar = ion.hist.tar_recoil
    v_rec_lab = ion.hist.lab_recoil

    # Because of the beam spot effect v_virt_lab and v_real_lab must be
    # given in the coordinates relative to the recoil point. Directions
    # are still in the laboratory coordinates.

    v_virt_lab = oj.Vector()

    # Copying and subtraction combined (compared to original)
    v_virt_lab.p.x = p_virt_lab.x - v_rec_lab.p.x
    v_virt_lab.p.y = p_virt_lab.y - v_rec_lab.p.y
    v_virt_lab.p.z = p_virt_lab.z - v_rec_lab.p.z

    r = math.sqrt(v_virt_lab.p.x**2 + v_virt_lab.p.y**2 + v_virt_lab.p.z**2)
    v_virt_lab.theta = math.acos(v_virt_lab.p.z / r)
    v_virt_lab.fii = math.atan2(v_virt_lab.p.y, v_virt_lab.p.x)

    v_real_lab = oj.Vector()
    v_real_lab.p = misc_jit.coord_transform(
        det.vfoil.center, theta, fii, v_real_foil.p, enums.CoordTransformDirection.FORW)

    v_real_lab.p.x -= v_rec_lab.p.x
    v_real_lab.p.y -= v_rec_lab.p.y
    v_real_lab.p.z -= v_rec_lab.p.z

    r = math.sqrt(v_real_lab.p.x**2 + v_real_lab.p.y**2 + v_real_lab.p.z**2)

    v_real_lab.theta = math.acos(v_real_lab.p.z / r)
    v_real_lab.fii = math.atan2(v_real_lab.p.y, v_real_lab.p.x)

    v_diff_lab = oj.Vector()
    v_diff_lab.theta, v_diff_lab.fii = rotate_jit.rotate(
        v_virt_lab.theta, c.C_PI + v_virt_lab.fii, v_real_lab.theta, v_real_lab.fii)

    v_recreal_pri = oj.Vector()
    v_recreal_pri.theta, v_recreal_pri.fii = rotate_jit.rotate(
        v_recvirt_pri.theta, v_recvirt_pri.fii, v_diff_lab.theta, v_diff_lab.fii)

    if g.simtype == enums.SimType.RBS:
        raise NotImplementedError

    # dE1: relative change in ion energy in the kinematics correction
    # dE2: energy loss correction

    dE1 = 0.0
    if g.simtype == enums.SimType.ERD:
        dE1 = (math.cos(v_recreal_pri.theta) / math.cos(v_recvirt_pri.theta)) **2 - 1.0
    elif g.simtype == enums.SimType.RBS:
        raise NotImplementedError

    # p_rec_tar:   recoiling point in target coordinates
    # v_virt_tar: old (virtual) recoiling direction in target coordinates
    # v_real_tar: new (real) recoiling direction in target coordinates
    #
    # v_rec_tar.p is in the coordinates of the specific ion
    # p_rec_tar is in the coordinates of the target coordinate system

    p_rec_tar = misc_jit.coord_transform(
        v_rec_lab.p, g.beamangle, c.C_PI, v_rec_tar.p, enums.CoordTransformDirection.FORW)

    v_virt_tar = oj.Vector()
    v_virt_tar.p.x = p_out_tar.x - p_rec_tar.x
    v_virt_tar.p.y = p_out_tar.y - p_rec_tar.y
    v_virt_tar.p.z = p_out_tar.z - p_rec_tar.z

    r = math.sqrt(v_virt_tar.p.x**2 + v_virt_tar.p.y**2 + v_virt_tar.p.z**2)

    v_virt_tar.theta = math.acos(v_virt_tar.p.z / r)
    v_virt_tar.fii = math.atan2(v_virt_tar.p.y, v_virt_tar.p.x)

    # Here v_virt_tar direction is in laboratory coordinates. We now calculate
    # the directions in target coordinates for both virtual and real.

    v_real_tar = oj.Vector()
    v_real_tar.theta, v_real_tar.fii = rotate_jit.rotate(
        v_virt_tar.theta, v_virt_tar.fii, v_diff_lab.theta, v_diff_lab.fii)

    v_virt_tar.theta, v_virt_tar.fii = rotate_jit.rotate(
        g.beamangle, 0.0, v_virt_tar.theta, v_virt_tar.fii)

    v_real_tar.theta, v_real_tar.fii = rotate_jit.rotate(
        g.beamangle, 0.0, v_real_tar.theta, v_real_tar.fii)

    eloss = ion.hist.recoil_E - ion.E

    dE2 = get_eloss_corr(ion, target, dE1, v_virt_tar.theta, v_real_tar.theta)

    dE1 *= ion.hist.recoil_E
    dE2 *= -eloss

    # dw is the correction factor cross section when we move from the virtual
    # detector to the real detector

    dw = 0.0
    if g.simtype == enums.SimType.ERD:
        dw = (math.cos(v_recvirt_pri.theta) / math.cos(v_recreal_pri.theta))**3
    elif g.simtype == enums.SimType.RBS:
        raise NotImplementedError

    n = ion.tlayer - target.ntarget  # n = foil number

    ion.w *= dw
    ion.E += dE1 + dE2

    if det.type == enums.DetectorType.GAS:
        raise NotImplementedError

    # We may end up with a negative energy

    if g.ionemax < ion.E < g.emin:
        ion.status = enums.IonStatus.FIN_MISS_DET
        # fprintf(stderr, "Energy would change too much in virtual detector\n")

    if err:
        ion.status = enums.IonStatus.FIN_MISS_DET
        # fprintf(stderr, "Unphysical RBS-scattering in virtual detector\n")

    if ion.status == enums.IonStatus.NOT_FINISHED:
        v_real_lab.p.x += v_rec_lab.p.x
        v_real_lab.p.y += v_rec_lab.p.y
        v_real_lab.p.z += v_rec_lab.p.z

        dist = erd_detector_jit.get_distance(p_out_tar, v_real_lab.p)
        ion.time += dist / math.sqrt(2.0 * ion.E / ion.A)

        # ion.lab.p = v_real_lab.p
        ion.lab.p.x = v_real_lab.p.x
        ion.lab.p.y = v_real_lab.p.y
        ion.lab.p.z = v_real_lab.p.z
        ion.lab.theta = det.angle
        ion.lab.fii = 0.0

        ion.p.x = 0.0
        ion.p.y = 0.0
        ion.p.z = 0.0

        ion.theta, ion.fii = rotate_jit.rotate(det.angle, c.C_PI, v_real_lab.theta, v_real_lab.fii)

        ion.opt.cos_theta = math.cos(ion.theta)
        ion.opt.sin_theta = math.sin(ion.theta)


@nb.njit(cache=True)
def get_eloss_corr(ion: oj.Ion, target: oj.Target, dE1: float, theta1: float, theta2: float) -> float:
    """Calculate the effective (according to the direct trajectory) energy loss
    Ed1 to the virtual detector point using initial recoil energy E, and the
    effective energy loss Ed2 to the real detector point using the kinematically
    corrected recoil energy (E + dE1*E) in order to take into account the
    systematic changes in the energy loss.
    """
    E = ion.hist.recoil_E
    # eloss = ion.hist.recoil_E - ion.E  # Unused

    Ed1 = get_eloss(E, ion, target, math.cos(theta1))
    Ed2 = get_eloss(E + dE1 * E, ion, target, math.cos(theta2))

    if Ed1 > E or Ed2 > (E + dE1 * E):
        return -2.0

    if Ed1 > 0.0:
        Ed2 = Ed2 / Ed1 - 1.0
    else:
        Ed2 = 0.0

    return Ed2


@nb.njit(cache=True)
def get_eloss(E: float, ion: oj.Ion, target: oj.Target, cos_theta: float) -> float:
    """Calculate uncorrected energy loss"""
    cont = True
    surf = False

    nlayer = ion.hist.layer
    layer = target.layer[nlayer]

    recz = ion.hist.tar_recoil.p.z

    Eorig = E

    while cont:
        d = 20.0 * c.C_NM
        nextz = recz + d * cos_theta
        v = math.sqrt(2.0 * E / ion.A)
        next_layer = False
        if nextz < layer.dlow:
            nextz = layer.dlow + 0.1 * c.C_ANGSTROM * cos_theta
            d = (nextz - recz) / cos_theta
            next_layer = True

        sto = ion_simu_jit.inter_sto(layer.sto[ion.scatindex], v, enums.IonMode.STOPPING)
        if E - d * sto < 0.0:
            cont = False
        else:
            v2 = math.sqrt(2.0 * (E - d * sto) / ion.A)
            sto2 = ion_simu_jit.inter_sto(layer.sto[ion.scatindex], v2, enums.IonMode.STOPPING)
            E -= 0.5 * (sto + sto2) * d

        if E < 0.0:
            cont = False

        if next_layer:
            nlayer -= 1
            if nlayer < 0:
                cont = False
                surf = True
            else:
                layer = target.layer[nlayer]

        recz = nextz

    if surf:
        return Eorig - E
    return 1.1 * Eorig


def rbs_cross(m1: float, m2: float, theta: float, err: bool) -> float:
    raise NotImplementedError


def rbs_kin(m1: float, m2: float, theta: float, err: bool) -> float:
    raise NotImplementedError
