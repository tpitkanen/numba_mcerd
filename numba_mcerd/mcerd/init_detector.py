import copy
import logging
import math

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import misc


def init_detector(g: o.Global, detector: o.Detector) -> None:
    """Initialize detector values"""
    # TODO: This loop may not work like in the original code.
    #       The reason is that C structs are different from Python classes
    for i in range(detector.nfoils):
        p1 = o.Point()
        p2 = o.Point()
        p3 = o.Point()
        foil = detector.foil[i]

        if not (foil.type == c.FoilType.CIRC or foil.type == c.FoilType.RECT):
            logging.warning(f"Unknown foil type {foil.type}")
        else:
            p1.x = math.sin(detector.angle) * foil.dist
            p1.y = 0.0
            p1.z = math.cos(detector.angle) * foil.dist

            p2 = copy.copy(p1)
            p2.y += foil.size[0] if foil.type == c.FoilType.CIRC else foil.size[1]

            p3 = copy.copy(p1)
            p3.x += math.cos(detector.angle) * foil.size[0]
            p3.z -= math.sin(detector.angle) * foil.size[0]

        foil.center = p1
        foil.plane = get_plane_params(p1, p2, p3)
        foil.virtual = False

    if g.virtualdet:
        raise NotImplementedError

    p1 = o.Point()
    p2 = o.Point()
    p3 = o.Point()
    first_foil = detector.foil[0]
    if first_foil.type == c.FoilType.CIRC or first_foil.type == c.FoilType.RECT:
        p1.x = g.bspot.x
        p1.y = g.bspot.y
        p1.z = p1.x / math.tan(c.C_PI / 2.0 - g.beamangle)

        p2.x = first_foil.center.x - math.cos(detector.angle) * first_foil.size[0]
        p2.y = first_foil.size[0] if first_foil.type == c.FoilType.CIRC else first_foil.size[1]
        p2.z = first_foil.center.z + math.sin(detector.angle) * first_foil.size[0]

        p3 = misc.coord_transform(p1, detector.angle, c.C_PI, p2, c.CoordTransformDirection.FORW.value)
        d = math.sqrt(p3.x**2 + p3.y**2 + p3.z**2)
        tmax1 = math.acos(p3.z / d)

        p1.x = -p1.x
        p1.y = -p1.y
        p1.z = -p1.z

        p2.x = first_foil.center.x + math.cos(detector.angle) * first_foil.size[0]
        p2.y = first_foil.size[0] if first_foil.type == c.FoilType.CIRC else first_foil.size[1]
        p2.z = first_foil.center.z + math.sin(detector.angle) * first_foil.size[0]

        p3 = misc.coord_transform(p1, detector.angle, c.C_PI, p2, c.CoordTransformDirection.FORW.value)
        d = math.sqrt(p3.x**2 + p3.y**2 + p3.z**2)
        tmax2 = math.acos(p3.z / d)
        detector.thetamax = max(tmax1, tmax2)

    raise NotImplementedError  # TODO: Untested


# TODO: Create a unit test
def get_plane_params(p1: o.Point, p2: o.Point, p3: o.Point) -> o.Plane:
    """Create a plane from points"""
    plane = o.Plane()

    if p3.x == p1.x or p2.y == p1.y:
        raise ValueError("Geometry not supported")

    plane.type = c.PlaneType.GENERAL_PLANE

    plane.a = (p3.z - p1.z) / (p3.x - p1.x)
    plane.b = (p2.z - p1.z) / (p2.y - p1.y)
    plane.c = p1.z - (plane.a - p1.x) - (plane.b * p1.y)

    return plane
