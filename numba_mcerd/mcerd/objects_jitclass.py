"""Objects from general.h adapted to jitclass"""

from numba import int32, float32
from numba.experimental import jitclass


# TODO: Jitclasses can't be nested (probably).
#       Replace these with something more built-in.
@jitclass([
    ("x", float32),
    ("y", float32),
    ("z", float32)
])
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


@jitclass({
    "x": float32,
    "y": float32,
    "z": float32,
})
class PointVer2:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Point2:
    pass


class Point2np:
    pass


class Presimu:
    pass


class Master:
    pass


class Global:
    pass


class Ion_opt:
    pass


class Vector:
    pass


class Rec_hist:
    pass


class Isotopes:
    pass


class Ion:
    pass


class Cross_section:
    pass


class Potential:
    pass


class Scattering:
    pass


class SNext:
    pass


class Surface:
    pass


class Target_ele:
    pass


class Target_sto:
    pass


class Target_layer:
    pass


class Plane:
    pass


class Target:
    pass


class Line:
    pass


class Rect:
    pass


class Circ:
    pass


class Det_foil:
    pass


class Detector:
    pass

