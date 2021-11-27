import math
from typing import Tuple

import numba as nb
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.mcerd.objects_jit as oj
from numba_mcerd.mcerd import enums, misc_jit, rotate_jit, virtual_detector_jit


class ErdDetectorError(Exception):
    """Error in ERD detector"""


@nb.njit(cache=True, nogil=True)
def is_in_energy_detector(g: oj.Global, ion: oj.Ion, target: oj.Target, detector: oj.Detector,
                          beyond_detector_ok: bool) -> bool:
    if detector.edet[0] <= target.ntarget:
        return False  # Probably not set or otherwise invalid
    if ion.tlayer == detector.edet[0]:
        return True  # In the first layer of energy detector
    if ion.tlayer >= detector.edet[0] and beyond_detector_ok:
        return True  # Beyond detector
    return False


@nb.njit(cache=True, nogil=True)
def move_to_erd_detector(g: oj.Global, ion: oj.Ion, target: oj.Target, detector: oj.Detector) -> None:
    """Recoil moves from the target to a detector foil or from one
    detector foil to the next
    """
    if ion.tlayer < 0:  # Just recoiled from the target
        ion.tlayer = target.ntarget
        for i in range(2):
            ion.dt[i] = 0.0
        for i in range(detector.nfoils):
            ion.Ed[i] = 0.0
            ion.hit[i].x = 0.0
            ion.hit[i].y = 0.0

    if ion.tlayer >= target.nlayers:
        ion.status = enums.IonStatus.FIN_DET
        return

    n = ion.tlayer - target.ntarget  # n = foil number
    ion.Ed[n] = ion.E

    if n < 0 or n >= detector.nfoils:
        raise ErdDetectorError("Ion in a strange detector foil")

    foil = detector.foil[n]

    pout = misc_jit.coord_transform(
        ion.lab.p, ion.lab.theta, ion.lab.fii, ion.p, enums.CoordTransformDirection.FORW)

    direction = oj.Vector()
    out_theta, out_fii = rotate_jit.rotate(ion.lab.theta, ion.lab.fii, ion.theta, ion.fii)
    direction.theta = out_theta
    direction.fii = out_fii

    ion_line = get_line_params(direction, pout)
    cross = oj.Point()
    is_cross = get_line_plane_cross(ion_line, foil.plane, cross)
    if is_cross:
        # ion.lab.p = cross
        ion.lab.p.x = cross.x
        ion.lab.p.y = cross.y
        ion.lab.p.z = cross.z
        ion.p.x = 0.0
        ion.p.y = 0.0
        ion.p.z = 0.0
        theta = detector.angle
        fii = c.C_PI
        fcross = misc_jit.coord_transform(
            foil.center, theta, fii, cross, enums.CoordTransformDirection.BACK)
        # ion.hit[n] = fcross
        ion.hit[n].x = fcross.x
        ion.hit[n].y = fcross.y
        ion.hit[n].z = fcross.z
    # TODO: is_on_right_side and is_in_foil are needlessly repeated
    if is_cross and is_on_right_side(pout, direction, cross) and is_in_foil(fcross, foil):
        dist = get_distance(pout, cross)
        ion.time += dist / math.sqrt(2.0 * ion.E / ion.A)
        for i in range(2):
            if ion.tlayer == detector.tdet[i]:
                ion.dt[i] = ion.time
        ion.status = enums.IonStatus.NOT_FINISHED
        ion.lab.theta = detector.angle
        ion.lab.fii = 0.0
        ion.theta, ion.fii = rotate_jit.rotate(detector.angle, c.C_PI, out_theta, out_fii)
        ion.opt.cos_theta = math.cos(ion.theta)
        ion.opt.sin_theta = math.sin(ion.theta)
    else:
        if (n == 0 and is_cross and g.virtualdet and is_on_right_side(pout, direction, cross)
                and is_in_foil(fcross, detector.vfoil)):
            ion.status = enums.IonStatus.NOT_FINISHED
            # TODO: Should this return cross and pout instead?
            #       Also check if values are copied
            virtual_detector_jit.hit_virtual_detector(g, ion, target, detector, cross, pout)
            if detector.type == enums.DetectorType.TOF:
                for i in range(2):
                    if ion.tlayer == detector.tdet[i]:
                        ion.dt[i] = ion.time
            ion.virtual = True
        else:
            ion.lab.theta = detector.angle
            ion.lab.fii = 0.0
            if n > 0:
                ion.status = enums.IonStatus.FIN_OUT_DET
            else:
                ion.status = enums.IonStatus.FIN_MISS_DET


@nb.njit(cache=True, nogil=True)
def get_line_params(d: oj.Vector, p: oj.Point) -> oj.Line:
    """Calculate line parameters for a line pointing to the direction d
    and going through point p
    """
    k = oj.Line()

    z = math.cos(d.theta)
    y = math.sin(d.theta) * math.sin(d.fii)
    x = math.sin(d.theta) * math.cos(d.fii)

    if z == 0.0:
        if x == 0.0:
            k.type = enums.LineType.Y_AXIS
            k.a = p.x
            k.b = p.z
        else:
            k.type = enums.LineType.XY_PLANE
            k.a = y / x
            k.b = p.y - k.a * p.x
            k.c = p.z
    else:
        k.type = enums.LineType.GENERAL
        k.a = x / z
        k.b = p.x - k.a * p.z
        k.c = y / z
        k.d = p.y - k.c * p.z

    return k


@nb.njit(cache=True, nogil=True)
def get_line_plane_cross(line: oj.Line, plane: oj.Plane, cross: oj.Point) -> bool:
    # Cross is modified
    p = plane
    q = line
    k = cross
    is_cross = False

    if p.type == enums.PlaneType.GENERAL_PLANE:
        if q.type == enums.LineType.GENERAL:
            if (q.a - p.b - p.a * q.c) != 0.0:
                is_cross = True
                k.z = (p.a * q.b + p.b * q.d + p.c) / (1.0 - p.a * q.a - p.b * q.c)
                k.x = q.a * k.z + q.b
                k.y = q.c * k.z + q.d
            else:
                is_cross = False
        elif q.type == enums.LineType.XY_PLANE:
            is_cross = True
            k.x = -q.b / q.a - (-p.a * q.b - q.a * (-p.c - p.b * q.c)) / (q.a * (p.a - q.a))
            k.y = -(-p.a * q.b - q.a * (-p.c - p.b * q.c)) / (p.a - q.a)
            k.z = q.c
        elif q.type == enums.LineType.Y_AXIS:
            k.x = q.a
            k.y = p.a * q.a + p.b * q.b + p.c
            k.z = q.b
            is_cross = True
    elif p.type == enums.PlaneType.Z_PLANE:
        if q.type == enums.LineType.GENERAL:
            is_cross = True
            k.x = q.c * p.a + q.d
            k.y = q.a * p.a + q.b
            k.z = p.a
        elif q.type == enums.LineType.XY_PLANE:
            is_cross = False
        elif q.type == enums.LineType.Y_AXIS:
            is_cross = False
    elif p.type == enums.PlaneType.X_PLANE:
        if q.type == enums.LineType.GENERAL:
            is_cross = True
            k.x = p.a
            k.z = (p.a - q.d) / (q.c + 1.0e-20)
            k.y = k.z * q.a + q.b
        elif q.type == enums.LineType.XY_PLANE:
            is_cross = True
            k.x = p.a
            k.y = p.a * q.a + q.b
            k.z = q.c
        elif q.type == enums.LineType.Y_AXIS:
            is_cross = False

    return is_cross


@nb.njit(cache=True, nogil=True)
def get_distance(p1: oj.Point, p2: oj.Point) -> float:
    return math.sqrt(
        (p1.x - p2.x)**2
        + (p1.y - p2.y)**2
        + (p1.z - p2.z)**2)


@nb.njit(cache=True, nogil=True)
def is_on_right_side(p1: oj.Point, d: oj.Vector, p2: oj.Point) -> int:
    dp = oj.Point()

    r1 = get_distance(p1, p2)

    dp.x = p1.x + r1 * math.sin(d.theta) * math.cos(d.fii)
    dp.y = p1.y + r1 * math.sin(d.theta) * math.sin(d.fii)
    dp.z = p1.z + r1 * math.cos(d.theta)

    r2 = get_distance(dp, p2)

    return r2 < r1


@nb.njit(cache=True, nogil=True)
def is_in_foil(p: oj.Point, foil: oj.Det_foil) -> bool:
    """(Point p is assumed to be on the plane of circle c)"""
    if foil.type == enums.FoilType.CIRC:
        if foil.virtual:
            return math.sqrt((p.x / foil.size_[0]) ** 2 + (p.y / foil.size_[1]) ** 2) <= 1.0
        return math.sqrt(p.x ** 2 + p.y ** 2) <= foil.size_[0]
    if foil.type == enums.FoilType.RECT:
        return abs(p.x) <= foil.size_[0] and abs(p.y) <= foil.size_[1]
    return False


def get_angle_between_points(p: oj.Point, p1: oj.Point, p2: oj.Point) -> float:
    raise NotImplementedError
