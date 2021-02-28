"""Objects from general.h adapted to jitclass"""

import numpy as np
import numba as nb
from numba import int64, float64
from numba.experimental import jitclass

from numba_mcerd.mcerd import constants


# TODO: Jitclasses can be nested:
#       https://stackoverflow.com/questions/57640039/numba-jit-and-deferred-types
#       https://stackoverflow.com/questions/38682260/how-to-nest-numba-jitclass
#       https://stackoverflow.com/questions/59215076/how-make-a-python-class-jitclass-compatible-when-it-contains-itself-jitclass-cla

# TODO: How to use Enums?
#       https://stackoverflow.com/questions/61155860/can-i-compile-a-python-enum-into-a-numba-jitclass

# TODO: How to create empty lists? Creating with one element and then
#       clearing feels like a workaround.
#       https://stackoverflow.com/questions/58412982/using-lists-in-numba-jitclass

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
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


@jitclass({
    "x": float64,
    "y": float64
})
class Point2:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


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

@jitclass({
    "Z": float64,
    "A": float64
})
class Target_ele:
    """Target element"""
    def __init__(self):
        self.Z = 0.0
        self.A = 0.0


@jitclass({
    "vel": float64[:],
    "sto": float64[:],
    "stragg": float64[:],
    "stodiv": float64,
    "n_sto": int64
})
class Target_sto:
    """Target stopping values"""
    def __init__(self):
        self.vel = np.zeros(constants.MAXSTO, dtype=np.float64)
        self.sto = np.zeros(constants.MAXSTO, dtype=np.float64)
        self.stragg = np.zeros(constants.MAXSTO, dtype=np.float64)
        self.stodiv = 0.0
        self.n_sto = 0


@jitclass({
    "natoms": int64,
    "dlow": float64,
    "dhigh": float64,
    "atom": int64[:],
    "N": float64[:],
    "Ntot": float64,
    # "sto": nb.types.List(Target_sto.class_type.instance_type),
    "type": int64,  # constants.TargetType
    "gas": nb.boolean,
    "stofile_prefix": nb.types.string
})
class Target_layer:
    """Target layer"""
    def __init__(self):
        self.natoms = 0  # Number of different elements in this layer
        self.dlow = 0.0  # lower depth of the target layer
        self.dhigh = 0.0  # higher depth of the target layer
        self.atom = np.zeros(constants.MAXATOMS, dtype=np.int64)  # Array of the indices of the target elements
        self.N = np.zeros(constants.MAXATOMS, dtype=np.float64)  # Array of the atomic densities
        self.Ntot = 0.0  # Total atomic density in the layer
        # self.sto = [Target_sto()]  # Electronic stopping for different ions  # len g.nions
        self.type = -1  # Type of the target layer
        self.gas = False  # Whether the target layer is gas or not
        self.stofile_prefix = ""  # len MAXSTOFILEPREFIXLEN

        # self.sto.clear()  # Needed to clear


@jitclass({
    "a": float64,
    "b": float64,
    "c": float64,
    "type": int64  # constants.PlaneType
})
class Plane:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.type = -1


@jitclass({
    "minN": float64,
    "ele": nb.types.List(Target_ele.class_type.instance_type),
    "layer": nb.types.List(Target_layer.class_type.instance_type),
    "nlayers": int64,
    "ntarget": int64,
    "natoms": int64,
    "recdist": nb.types.List(Point2.class_type.instance_type),
    "nrecdist": int64,
    "effrecd": float64,
    "recmaxd": float64,
    "plane": Plane.class_type.instance_type,
    "efin": float64[:],
    "recpar": nb.types.List(Point2.class_type.instance_type),
    "angave": float64,
    # "surface": Surface.class_type.instance_type,
    "cross": float64[:, :],
    "table": nb.boolean
})
class Target:
    def __init__(self):
        self.minN = 0.0  # Minimum atomic density in the target
        self.ele = [Target_ele() for _ in range(constants.MAXELEMENTS)]  # Array of different elements in the target  # len MAXELEMENTS
        self.layer = [Target_layer() for _ in range(constants.MAXLAYERS)]  # Array of the target layers  # len MAXLAYERS
        self.nlayers = 0  # Total number of target layers
        self.ntarget = 0  # Number of the layers in the target itself
        self.natoms = 0  # Total number of atoms in different target layers
        self.recdist = [Point2() for _ in range(constants.NRECDIST)]  # Recoil material distribution in the target  # len NRECDIST
        self.nrecdist = 0  # Number of points in recoil material distribution
        self.effrecd = 0.0  # Effective thickness of recoil material (nonzero values)
        self.recmaxd = 0.0  # Maximum depth of the recoiling material
        self.plane = Plane()  # Target surface plane
        self.efin = np.zeros(constants.NRECDIST, dtype=np.float64)  # Energies below which the recoil can't reach the surface  # len NRECDIST
        self.recpar = [Point2() for _ in range(constants.MAXLAYERS)]  # Recoiling solid angle fitted parameters  # len MAXLAYERS
        self.angave = 0.0  # Average recoiling half-angle
        # self.surface = Surface()  # Data structure for the rough surface
        # len [NSENE][NSANGLE]:
        self.cross = np.zeros((constants.NSENE, constants.NSANGLE), dtype=np.float64)  # Cross section table for scattering relative to the Rutherford cross sections as a function of ion lab. energy and lab scattering angle
        self.table = False  # True if cross section table is available


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



def main():
    target = Target()
    print(target.layer[0].atom)


if __name__ == '__main__':
    main()

