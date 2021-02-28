"""Objects from general.h adapted to jitclass"""

import numpy as np
from numba import int64, float64
from numba.experimental import jitclass


# TODO: Jitclasses can be nested:
#       https://stackoverflow.com/questions/57640039/numba-jit-and-deferred-types
#       https://stackoverflow.com/questions/38682260/how-to-nest-numba-jitclass
#       https://stackoverflow.com/questions/59215076/how-make-a-python-class-jitclass-compatible-when-it-contains-itself-jitclass-cla

# Tuple-style definition
# @jitclass([
#     ("x", float64),
#     ("y", float64),
#     ("z", float64)
# ])
# class Point: ...

# Dict-style definition
@jitclass({
    "x": float64,
    "y": float64,
    "z": float64,
})
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


@jitclass({
    "x": float64,
    "y": float64
})
class Point2:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# TODO: float64?
@jitclass({
    "n": int64,
    "d": int64,
    "ux": float64[:],
    "uy": float64[:]
})
class Potential:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.ux = np.zeros(n, dtype=np.float64)
        self.uy = np.zeros(n, dtype=np.float64)


# class Point2np:
#     pass
#
#
# class Presimu:
#     pass
#
#
# class Master:
#     pass
#
#
# class Global:
#     pass
#
#
# class Ion_opt:
#     pass
#
#
# class Vector:
#     pass
#
#
# class Rec_hist:
#     pass
#
#
# class Isotopes:
#     pass
#
#
# class Ion:
#     pass
#
#
# class Cross_section:
#     pass
#
#
# class Scattering:
#     pass
#
#
# class SNext:
#     pass
#
#
# class Surface:
#     pass
#
#
# class Target_ele:
#     pass
#
#
# class Target_sto:
#     pass
#
#
# class Target_layer:
#     pass
#
#
# class Plane:
#     pass
#
#
# class Target:
#     pass
#
#
# class Line:
#     pass
#
#
# class Rect:
#     pass
#
#
# class Circ:
#     pass
#
#
# class Det_foil:
#     pass
#
#
# class Detector:
#     pass

