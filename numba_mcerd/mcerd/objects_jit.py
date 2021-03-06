"""Objects from general.h adapted to jitclass"""

import numba as nb
import numpy as np
from numba import int64, float64, boolean
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


# class Point2np:
#     pass
#
#

@jitclass({
    "depth": float64,
    "angle": float64,
    "layer": int64
})
class Presimu:
    def __init__(self):
        self.depth = 0.0
        self.angle = 0.0
        self.layer = 0


# TODO: Separate Jibal from global


# TODO: Separate this from global
@jitclass({
    # "args": nb.types.List(nb.types.string),
    "fdata": nb.types.string,
    "fpout": nb.types.string,
    "fpdebug": nb.types.string,
    "fpdat": nb.types.string,
    "fperd": nb.types.string,
    "fprange": nb.types.string,
    "fptrack": nb.types.string
    # Not needed:
    # argc: int64
})
class Master:
    def __init__(self):
        # self.args = [""]
        self.fdata = ""
        self.fpout = ""
        self.fpdebug = ""
        self.fpdat = ""
        self.fperd = ""
        self.fprange = ""
        self.fptrack = ""
        # self.argc = 0


@jitclass({
    "mpi": boolean,
    "E0": float64,
    "nions": int64,
    "ncascades": int64,
    "nsimu": int64,
    "emin": float64,
    "ionemax": float64,
    "minangle": float64,
    "seed": int64,
    "cion": int64,
    "simtype": int64,  # enums.SimType
    "beamangle": float64,
    "bspot": Point2.class_type.instance_type,
    "simstage": int64,  # enums.SimStage
    "npresimu": int64,
    "nscale": int64,
    "nrecave": int64,
    "cpresimu": int64,
    "presimu": nb.types.List(Presimu.class_type.instance_type),
    "predata": int64,
    # "master": Master.class_type.instance_type,
    "frecmin": float64,
    "costhetamax": float64,
    "costhetamin": float64,
    "recwidth": int64,  # enums.RecWidth
    "virtualdet": boolean,
    "basename": nb.types.string,
    "finstat": int64[:, :],
    "beamdiv": float64,
    "beamprof": int64,  # enums.BeamProf
    "rough": boolean,
    "nmclarge": int64,
    "nmc": int64,
    "output_trackpoints": boolean,
    "output_misses": boolean,
    "cascades": boolean,
    "advanced_output": boolean,
    # "jibal": Jibal.class_type.instance_type,
    "nomc": boolean,
})
class Global:
    def __init__(self):
        self.mpi = False  # Boolean for parallel simulation
        self.E0 = 0.0  # Energy of the primary ion beam
        self.nions = 0  # Number of different ions in simulation, 1 or 2
        self.ncascades = 0
        self.nsimu = 0  # Total number of the simulated ions
        self.emin = 0.0  # Minimum energy of the simulation
        self.ionemax = 0.0  # Maximum possible ion energy in simulation
        self.minangle = 0.0  # Minimum angle of the scattering
        self.seed = 0  # Seed number of the random number generator  # Positive
        self.cion = 0  # Number of the current ion
        self.simtype = 0  # Type of the simulation  # enums.SimType
        self.beamangle = 0.0  # Angle between target surface normal and beam
        self.bspot = Point2()  # Size of the beam spot
        self.simstage = 0  # Presimulation or real simulation  # enums.SimStage
        self.npresimu = 0  # Number of simulated ions in the presimulation
        self.nscale = 0  # Number of simulated ions per scaling ions
        self.nrecave = 0  # Average number of recoils per primary ion
        self.cpresimu = 0  # Counter of simulated ions the the presimulation
        self.presimu = [Presimu()]  # Data structure for the presimulation  # Variable size
        self.predata = 0  # Presimulation data given in a file
        # self.master = Master()  # Data structure for the MPI-master
        self.frecmin = 0.0
        self.costhetamax = 0.0
        self.costhetamin = 0.0
        self.recwidth = 0  # Recoiling angle width type  # enums.RecWidth
        self.virtualdet = False  # Do we use the virtual detector
        self.basename = ""  # len NFILE
        self.finstat = np.zeros((2, 11), dtype=np.int64)  # len [SECONDARY + 1][NIONSTATUS]
        # TODO: Could pass len_ionstatus as parameter, or wait for len(IntEnum) support.
        #       dtype version supports automatic finstat size.
        # self.finstat = np.zeros((enums.IonType.SECONDARY.value + 1, len_ionstatus), dtype=np.int64)
        self.beamdiv = 0.0  # Angular divergence of the beam, width or FWHM
        self.beamprof = 0  # Beam profile: flat, gaussian, given distribution  # enums.BeamProf
        self.rough = False  # Rough or non-rough sample surface
        self.nmclarge = 0  # Number of rejected (large) MC scatterings (RBS)
        self.nmc = 0  # Number of all MC-scatterings
        self.output_trackpoints = False
        self.output_misses = False
        self.cascades = False
        self.advanced_output = False
        # self.jibal = Jibal()
        self.nomc = False

        self.presimu.clear()


@jitclass({
    "valid": boolean,
    "cos_theta": float64,
    "sin_theta": float64,
    "e": float64,
    "y": float64
})
class Ion_opt:
    def __init__(self):
        self.valid = False  # Validity of opt-variable
        self.cos_theta = 0.0  # Cosinus of laboratory theta-angle [-1,1]
        self.sin_theta = 0.0  # Sinus of laboratory theta-angle [-1,1]
        self.e = 0.0  # Dimensionless energy for the next scattering
        self.y = 0.0  # Dimensionless impact parameter  -"-


@jitclass({
    "p": Point.class_type.instance_type,
    "theta": float64,
    "fii": float64,
    "v": float64
})
class Vector:
    def __init__(self):
        self.p = Point()  # Cartesian coordinates of the object
        self.theta = 0.0  # Direction of the movement of the object
        self.fii = 0.0  # Direction of the movement of the object
        self.v = 0.0  # Object velocity


@jitclass({
    "tar_recoil": Vector.class_type.instance_type,
    "ion_recoil": Vector.class_type.instance_type,
    "lab_recoil": Vector.class_type.instance_type,
    "tar_primary": Vector.class_type.instance_type,
    "lab_primary": Vector.class_type.instance_type,
    "ion_E": float64,
    "recoil_E": float64,
    "layer": int64,
    "nsct": int64,
    "time": float64,
    "w": float64,
    "Z": float64,
    "A": float64
})
class Rec_hist:
    def __init__(self):
        self.tar_recoil = Vector()  # Recoil vector in target coord. at recoil moment
        self.ion_recoil = Vector()  # Recoil vector in prim. ion coord. at recoil moment
        self.lab_recoil = Vector()  # Recoil vector in lab. coord. at recoil moment
        self.tar_primary = Vector()  # Primary ion vector in target coordinates
        self.lab_primary = Vector()  # Primary ion vector in lab coordinates
        self.ion_E = 0.0
        self.recoil_E = 0.0
        self.layer = 0  # Recoiling layer
        self.nsct = 0  # Number of scatterings this far
        self.time = 0.0  # Recoiling time
        self.w = 0.0  # Statistical weight at the moment of the recoiling
        self.Z = 0.0  # Atomic number of the secondary atom
        self.A = 0.0  # Mass of the secondary atom in the scattering


@jitclass({
    "A": float64[:],
    "c": float64[:],
    "c_sum": float64,
    "n": int64,
    "Am": float64
})
class Isotopes:
    def __init__(self):
        self.A = np.zeros(constants.MAXISOTOPES, dtype=np.float64)  # List of the isotope masses in kg  # len MAXISOTOPES
        self.c = np.zeros(constants.MAXISOTOPES, dtype=np.float64)  # Isotope concentrations normalized to 1.0  # len MAXISOTOPES
        self.c_sum = 0.0  # Sum of concentrations, should be 1.0.
        self.n = 0  # Number of different natural isotopes
        self.Am = 0.0  # Mass of the most abundant isotope


@jitclass({
    "Z": float64,
    "A": float64,
    "E": float64,
    "I": Isotopes.class_type.instance_type,
    "p": Point.class_type.instance_type,
    "theta": float64,
    "fii": float64,
    "nsct": int64,
    "status": int64,  # enums.IonStatus
    "opt": Ion_opt.class_type.instance_type,
    "w": float64,
    "wtmp": float64,
    "time": float64,
    "tlayer": int64,
    "lab": Vector.class_type.instance_type,
    "type": int64,  # enums.IonType
    "hist": Rec_hist.class_type.instance_type,
    "dist": float64,
    "virtual": boolean,
    "hit": nb.types.List(Point.class_type.instance_type),
    "Ed": float64[:],
    "dt": float64[:],
    "scale": boolean,
    "effrecd": float64,
    "trackid": int64,
    "scatindex": int64,
    "ion_i": int64,
    "E_nucl_loss_det": float64
})
class Ion:
    def __init__(self):
        self.Z = 0.0  # Atomic number of ion (non-integer for compat.)
        self.A = 0.0  # Mass of ion in the units of kg
        self.E = 0.0  # Energy of the ion
        self.I = Isotopes()  # Data structure for natural isotopes
        self.p = Point()  # Three dimensional position of ion
        self.theta = 0.0  # Laboratory theta-angle of the ion [0,PI]
        self.fii = 0.0  # Laboratory fii-angle of the ion [0,2*PI]
        self.nsct = 0  # Number of scatterings of the ion
        self.status = -1  # enums.IonStatus  # Status value for ion (stopped, recoiled etc.)
        self.opt = Ion_opt()  # Structure for the optimization-variables
        self.w = 0.0  # Statistical weight of the ion
        self.wtmp = 0.0  # Temporary statistical weight of the ion
        self.time = 0.0  # Time since the creation of the ion
        self.tlayer = 0  # Number of the current target layer
        self.lab = Vector()  # Translation and rotation of the current coordinate system in the laboratory coordinate system
        self.type = -1  # enums.IonType  # Primary, secondary etc.
        self.hist = Rec_hist()  # Variables saved at the moment of the recoiling event
        self.dist = 0.0  # Distance to the next ERD-scattering point
        self.virtual = False  # Did we only hit the virtual detector area
        self.hit = [Point() for _ in range(constants.MAXLAYERS)]  # Hit points to the detector layers  # len MAXLAYERS
        self.Ed = np.zeros(constants.MAXLAYERS, dtype=np.float64)  # Ion energy in the detector layers  # len MAXLAYERS
        self.dt = np.zeros(constants.MAXLAYERS, dtype=np.float64)  # Passing times in the detector layers  # len MAXLAYERS
        self.scale = False  # TRUE if we have a scaling ion
        self.effrecd = 0.0  # Parameter for scaling the effective thickness of recoil material
        self.trackid = 0  # originally int64_t
        self.scatindex = 0  # Index of the scattering table for this ion
        self.ion_i = 0  # "i"th recoil in the track
        self.E_nucl_loss_det = 0.0


@jitclass({
    "emin": float64,
    "emax": float64,
    "estep": float64,
    "b": float64[:]
})
class Cross_section:
    def __init__(self):
        self.emin = 0.0
        self.emax = 0.0
        self.estep = 0.0
        self.b = np.zeros(0, dtype=np.float64)  # Initialized later  # TODO: Does this work in jitclass?


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


@jitclass({
    "angle": float64[:, :],
    "cross": Cross_section.class_type.instance_type,
    "logemin": float64,
    "logymin": float64,
    "logediv": float64,
    "logydiv": float64,
    "a": float64,
    "E2eps": float64,
    # "pot": Potential.class_type.instance_type
})
class Scattering:
    def __init__(self):
        self.angle = np.zeros((constants.EPSNUM, constants.YNUM), dtype=np.float64)  # 2D array for scattering angles  # len [EPSNUM][YNUM]
        self.cross = Cross_section()  # Data for maximum impact parameters
        self.logemin = 0.0  # Logarithm for minimum reduced energy
        self.logymin = 0.0  # Logarithm for minimum reduced impact parameter
        self.logediv = 0.0  # Logarithm for difference of reduced energies
        self.logydiv = 0.0  # Logarithm for difference reduced impact parameters
        self.a = 0.0  # Reduced unit for screening length
        self.E2eps = 0.0  # Constant for changing energy unit to reduced energy
        # self.potential = Potential()  # Originally commented out


# TODO: Class for Scatterings (g.nions, MAXELEMENTS)


@jitclass({
    "d": float64,
    "natom": int64,
    "r": boolean
})
class SNext:
    def __init__(self):
        self.d = 0.0  # Distance to the next scattering point
        self.natom = 0  # Number of the scattering atom
        self.r = False  # Boolean for true mc-scattering (instead of eg. interface crossing)


# TODO:
# class Surface:
#     pass


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
    "sto": nb.types.List(Target_sto.class_type.instance_type),
    "type": int64,  # enums.TargetType
    "gas": boolean,
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
        self.sto = [Target_sto()]  # Electronic stopping for different ions  # len g.nions
        # self.sto = nb.typed.List()  # Typed list, doesn't work in ion_simu_jit.move_ion()
        self.type = -1  # Type of the target layer
        self.gas = False  # Whether the target layer is gas or not
        self.stofile_prefix = ""  # len MAXSTOFILEPREFIXLEN

        # self.sto.append(Target_sto())  # Typed list, doesn't work in ion_simu_jit.move_ion()
        self.sto.clear()  # Needed to clear


@jitclass({
    "a": float64,
    "b": float64,
    "c": float64,
    "type": int64  # enums.PlaneType
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
    "table": boolean
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


@jitclass({
    "a": float64,
    "b": float64,
    "c": float64,
    "d": float64,
    "type": int64,
})
class Line:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0
        self.type = -1  # enums.LineType


# Unused
# class Rect:
#     pass
#

# Unused
# class Circ:
#     pass
#


@jitclass({
    "type": int64,
    "virtual": boolean,
    "dist": float64,
    "angle": float64,
    "size_": float64[:],
    "plane": Plane.class_type.instance_type,
    "center": Point.class_type.instance_type
})
class Det_foil:
    def __init__(self):
        self.type = -1  # enums.FoilType  # Rectangular or circular
        self.virtual = False  # Is this foil virtual
        self.dist = 0.0  # Distance from the target center
        self.angle = 0.0  # Angle of the foil
        self.size_ = np.zeros(2, dtype=np.float64)  # Diameter for circular, width and height for rect.  # len 2
        self.plane = Plane()  # Plane of the detector foil
        self.center = Point()  # Point of the center of the foil


@jitclass({
    "type": int64,
    "angle": float64,
    "nfoils": int64,
    "virtual": boolean,
    "vsize": float64[:],
    "tdet": int64[:],
    "edet": int64[:],
    "thetamax": float64,
    "vthetamax": float64,
    "foil": nb.types.List(Det_foil.class_type.instance_type),
    "vfoil": Det_foil.class_type.instance_type
})
class Detector:
    def __init__(self):
        self.type = -1  # enums.DetectorType  # TOF, GAS or FOIL
        self.angle = 0.0  # Detector angle relative to the beam direction
        self.nfoils = 0  # Number of foils in the detector
        self.virtual = False  # Do we have the virtual detector
        self.vsize = np.zeros(2, dtype=np.float64)  # Size of the virtual detector relative to the real  # len 2
        self.tdet = np.zeros(2, dtype=np.int64)  # Layer numbers for the timing detectors  # len 2
        self.edet = np.zeros(constants.MAXLAYERS, dtype=np.int64)  # Layer number(s) for energy detector (layers)  # len MAXLAYERS
        self.thetamax = 0.0  # maximum half-angle of the detector (spot size incl.)
        self.vthetamax = 0.0  # maximum half-angle of the virtual detector
        self.foil = [Det_foil() for _ in range(constants.MAXFOILS)]  # Detector foil data structure  # len MAXFOILS
        self.vfoil = Det_foil()  # Virtual detector data structure


def main():
    # Test initialization of objects

    target = Target()
    print(target.layer[0].atom)

    g = Global()
    print(g)

    ion = Ion()
    print(ion)

    scattering = Scattering()
    print(scattering)

    line = Line()
    print(line)

    detector = Detector()
    print(detector)


if __name__ == '__main__':
    main()

