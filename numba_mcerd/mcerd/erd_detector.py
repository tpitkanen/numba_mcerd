import math
from typing import Tuple

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import enums, misc, rotate, virtual_detector


class ErdDetectorError(Exception):
    """Error in ERD detector"""


def is_in_energy_detector(g: o.Global, ion: o.Ion, target: o.Target, detector: o.Detector,
                          beyond_detector_ok: bool) -> bool:
    """Check if ion is if energy energy detector"""
    if detector.edet[0] <= target.ntarget:
        return False  # Probably not set or otherwise invalid
    if ion.tlayer == detector.edet[0]:
        return True  # In the first layer of energy detector
    if ion.tlayer >= detector.edet[0] and beyond_detector_ok:
        return True  # Beyond detector
    return False


def move_to_erd_detector(g: o.Global, ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
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

    pout = misc.coord_transform(
        ion.lab.p, ion.lab.theta, ion.lab.fii, ion.p, enums.CoordTransformDirection.FORW)

    direction = o.Vector()
    out_theta, out_fii = rotate.rotate(ion.lab.theta, ion.lab.fii, ion.theta, ion.fii)
    direction.theta = out_theta
    direction.fii = out_fii

    ion_line = get_line_params(direction, pout)
    is_cross, cross = get_line_plane_cross(ion_line, foil.plane)
    if is_cross:
        ion.lab.p = cross
        ion.p.x = 0.0
        ion.p.y = 0.0
        ion.p.z = 0.0
        theta = detector.angle
        fii = c.C_PI
        fcross = misc.coord_transform(
            foil.center, theta, fii, cross, enums.CoordTransformDirection.BACK)
        ion.hit[n] = fcross

    if is_cross and is_on_right_side(pout, direction, cross) and is_in_foil(fcross, foil):
        dist = get_distance(pout, cross)
        ion.time += dist / math.sqrt(2.0 * ion.E / ion.A)
        for i in range(2):
            if ion.tlayer == detector.tdet[i]:
                ion.dt[i] = ion.time
        ion.status = enums.IonStatus.NOT_FINISHED
        ion.lab.theta = detector.angle
        ion.lab.fii = 0.0
        ion.theta, ion.fii = rotate.rotate(detector.angle, c.C_PI, out_theta, out_fii)
        ion.opt.cos_theta = math.cos(ion.theta)
        ion.opt.sin_theta = math.sin(ion.theta)
    else:
        if (n == 0 and is_cross and g.virtualdet and is_on_right_side(pout, direction, cross)
                and is_in_foil(fcross, detector.vfoil)):
            ion.status = enums.IonStatus.NOT_FINISHED
            # TODO: Should this return cross and pout instead?
            #       Also check if values are copied
            virtual_detector.hit_virtual_detector(g, ion, target, detector, cross, pout)
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


def get_line_params(d: o.Vector, p: o.Point) -> o.Line:
    """Calculate line parameters for a line pointing to the direction d
    and going through point p
    """
    k = o.Line()

    z = math.cos(d.theta)
    y = math.sin(d.theta) * math.sin(d.fii)
    x = math.sin(d.theta) * math.cos(d.fii)

    if z == 0.0:
        if x == 0.0:
            k.type = enums.LineType.L_YAXIS
            k.a = p.x
            k.b = p.z
        else:
            k.type = enums.LineType.L_XYPLANE
            k.a = y / x
            k.b = p.y - k.a * p.x
            k.c = p.z
    else:
        k.type = enums.LineType.L_GENERAL
        k.a = x / z
        k.b = p.x - k.a * p.z
        k.c = y / z
        k.d = p.y - k.c * p.z

    return k


def get_line_plane_cross(line: o.Line, plane: o.Plane) -> Tuple[enums.Cross, o.Point]:
    p = plane
    q = line
    k = o.Point()
    cross_type = enums.Cross.NO_CROSS

    if p.type == enums.PlaneType.GENERAL_PLANE:
        if q.type == enums.LineType.L_GENERAL:
            if (q.a - p.b - p.a * q.c) != 0.0:
                cross_type = enums.Cross.CROSS
                k.z = (p.a * q.b + p.b * q.d + p.c) / (1.0 - p.a * q.a - p.b * q.c)
                k.x = q.a * k.z + q.b
                k.y = q.c * k.z + q.d
            else:
                cross_type = enums.Cross.NO_CROSS
        elif q.type == enums.LineType.L_XYPLANE:
            cross_type = enums.Cross.CROSS
            k.x = -q.b / q.a - (-p.a * q.b - q.a * (-p.c - p.b * q.c)) / (q.a * (p.a - q.a))
            k.y = -(-p.a * q.b - q.a * (-p.c - p.b * q.c)) / (p.a - q.a)
            k.z = q.c
        elif q.type == enums.LineType.L_YAXIS:
            k.x = q.a
            k.y = p.a * q.a + p.b * q.b + p.c
            k.z = q.b
            cross_type = enums.Cross.CROSS
    elif p.type == enums.PlaneType.Z_PLANE:
        if q.type == enums.LineType.L_GENERAL:
            cross_type = enums.Cross.CROSS
            k.x = q.c * p.a + q.d
            k.y = q.a * p.a + q.b
            k.z = p.a
        elif q.type == enums.LineType.L_XYPLANE:
            cross_type = enums.Cross.NO_CROSS
        elif q.type == enums.LineType.L_YAXIS:
            cross_type = enums.Cross.NO_CROSS
    elif p.type == enums.PlaneType.X_PLANE:
        if q.type == enums.LineType.L_GENERAL:
            cross_type = enums.Cross.CROSS
            k.x = p.a
            k.z = (p.a - q.d) / (q.c + 1.0e-20)
            k.y = k.z * q.a + q.b
        elif q.type == enums.LineType.L_XYPLANE:
            cross_type = enums.Cross.CROSS
            k.x = p.a
            k.y = p.a * q.a + q.b
            k.z = q.c
        elif q.type == enums.LineType.L_YAXIS:
            cross_type = enums.Cross.NO_CROSS

    return cross_type, k


def get_distance(p1: o.Point, p2: o.Point) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def is_on_right_side(p1: o.Point, d: o.Vector, p2: o.Point) -> int:
    dp = o.Point()

    r1 = get_distance(p1, p2)

    dp.x = p1.x + r1 * math.sin(d.theta) * math.cos(d.fii)
    dp.y = p1.y + r1 * math.sin(d.theta) * math.sin(d.fii)
    dp.z = p1.z + r1 * math.cos(d.theta)

    r2 = get_distance(dp, p2)

    return r2 < r1


def is_in_foil(p: o.Point, foil: o.Det_foil) -> bool:
    """(Point p is assumed to be on the plane of circle c)"""
    if foil.type == enums.FoilType.CIRC:
        if foil.virtual:
            return math.sqrt((p.x / foil.size[0])**2 + (p.y / foil.size[1])**2) <= 1.0
        return math.sqrt(p.x**2 + p.y**2) <= foil.size[0]
    if foil.type == enums.FoilType.RECT:
        return abs(p.x) <= foil.size[0] and abs(p.y) <= foil.size[1]
    return False


def get_angle_between_points(p: o.Point, p1: o.Point, p2: o.Point) -> float:
    dp1 = o.Point()
    dp2 = o.Point()

    dp1.x = p1.x - p.x
    dp1.y = p1.y - p.y
    dp1.z = p1.z - p.z

    dp2.x = p2.x - p.x
    dp2.y = p2.y - p.y
    dp2.z = p2.z - p.z

    r1 = math.sqrt(dp1.x**2 + dp1.y**2 + dp1.z**2)
    r2 = math.sqrt(dp2.x**2 + dp2.y**2 + dp2.z**2)

    angle = math.acos((dp1.x * dp2.x + dp1.y * dp2.y + dp1.z * dp2.z) / (r1 * r2))

    return angle
