"""Objects from general.h"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from numba_mcerd.mcerd import constants
from numba_mcerd.mcerd.jibal import Jibal


# TODO: Dataclasses can't be optimized with @njit
#       Replace the most performance-critical with something faster


@dataclass
class Point:  # TODO: should this be Point3?
    """3D point"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Point2:
    """2D point"""
    x: float = 0.0
    y: float = 0.0


@dataclass
class Presimu:
    depth: float = 0.0
    angle: float = 0.0
    layer: int = 0


@dataclass
class Master:
    args: List[str] = None  # Command line arguments
    fdata: Path = None  # fdata[NFILE]
    fpout: Path = None
    fpdebug: Path = None  # Not sure if this should be a Path or even needed
    fpdat: Path = None
    fperd: Path = None
    fprange: Path = None
    fptrack: Path = None
    # Not needed:
    # argc

    def __post_init__(self):
        if self.args is None:
            self.args = []


@dataclass
class Global:
    mpi: bool = False  # Boolean for parallel simulation
    E0: float = 0.0  # Energy of the primary ion beam
    nions: int = 0  # Number of different ions in simulation, 1 or 2
    ncascades: int = 0
    nsimu: int = 0  # Total number of the simulated ions
    emin: float = 0.0  # Minimum energy of the simulation
    ionemax: float = 0.0  # Maximum possible ion energy in simulation
    minangle: float = 0.0  # Minimum angle of the scattering
    seed: int = 0  # Seed number of the random number generator  # Positive
    cion: int = 0  # Number of the current ion
    simtype: constants.SimType = None  # Type of the simulation
    beamangle: float = 0.0  # Angle between target surface normal and beam
    bspot: Point2 = None  # Size of the beam spot
    simstage: constants.SimStage = None  # Presimulation or real simulation
    npresimu: int = 0  # Number of simulated ions in the presimulation
    nscale: int = 0  # Number of simulated ions per scaling ions
    nrecave: int = 0  # Average number of recoils per primary ion
    cpresimu: int = 0  # Counter of simulated ions the the presimulation
    presimu: List[Presimu] = None  # Data structure for the presimulation
    predata: int = 0  # Presimulation data given in a file
    master: Master = None  # Data structure for the MPI-master
    frecmin: float = 0.0
    costhetamax: float = 0.0
    costhetamin: float = 0.0
    recwidth: constants.RecWidth = None  # Recoiling angle width type
    virtualdet: bool = False  # Do we use the virtual detector
    basename: str = None  # len NFILE
    finstat: List[List[int]] = None  # len [SECONDARY + 1][NIONSTATUS]
    beamdiv: int = 0  # Angular divergence of the beam, width or FWHM  # TODO: Should this be int or float?
    beamprof: constants.BeamProf = None  # Beam profile: flat, gaussian, given distribution
    rough: bool = False  # Rough or non-rough sample surface
    nmclarge: int = 0  # Number of rejected (large) MC scatterings (RBS)
    nmc: int = 0  # Number of all MC-scatterings
    output_trackpoints: bool = False
    output_misses: bool = False
    cascades: bool = False
    advanced_output: bool = False
    jibal: Jibal = None
    nomc: bool = False

    def __post_init__(self):
        if self.bspot is None:
            self.bspot = Point2()
        # if self.presimu is None:  # Variable size, too big to initialize beforehand
        #     self.presimu = None
        if self.master is None:
            self.master = Master()
        if self.finstat is None:
            self.finstat = [[0] * len(constants.IonStatus)
                            for _ in range(constants.IonType.SECONDARY.value + 1)]
        if self.jibal is None:
            self.jibal = Jibal()


@dataclass
class Ion_opt:
    valid: bool = False  # Validity of opt-variable
    cos_theta: float = 0.0  # Cosinus of laboratory theta-angle [-1,1]
    sin_theta: float = 0.0  # Sinus of laboratory theta-angle [-1,1]
    e: float = 0.0  # Dimensionless energy for the next scattering
    y: float = 0.0  # Dimensionless impact parameter  -"-


@dataclass
class Vector:
    p: Point = None  # Cartesian coordinates of the object
    theta: float = 0.0  # Direction of the movement of the object
    fii: float = 0.0  # Direction of the movement of the object
    v: float = 0.0  # Object velocity

    def __post_init__(self):
        if self.p is None:
            self.p = Point()


@dataclass
class Rec_hist:
    tar_recoil: Vector = None  # Recoil vector in target coord. at recoil moment
    ion_recoil: Vector = None  # Recoil vector in prim. ion coord. at recoil moment
    lab_recoil: Vector = None  # Recoil vector in lab. coord. at recoil moment
    tar_primary: Vector = None  # Primary ion vector in target coordinates
    lab_primary: Vector = None  # Primary ion vector in lab coordinates
    ion_E: float = 0.0
    recoil_E: float = 0.0
    layer: int = 0  # Recoiling layer
    nsct: int = 0  # Number of scatterings this far
    time: float = 0.0  # Recoiling time
    w: float = 0.0  # Statistical weight at the moment of the recoiling
    Z: float = 0.0  # Atomic number of the secondary atom
    A: float = 0.0  # Mass of the secondary atom in the scattering

    def __post_init__(self):
        if self.tar_recoil is None:
            self.tar_recoil = Vector()
        if self.ion_recoil is None:
            self.ion_recoil = Vector()
        if self.lab_recoil is None:
            self.lab_recoil = Vector()
        if self.tar_primary is None:
            self.tar_primary = Vector()
        if self.lab_primary is None:
            self.lab_primary = Vector()


@dataclass
class Isotopes:
    A: List[float] = None  # List of the isotope masses in kg  # len MAXISOTOPES
    c: List[float] = None  # Isotope concentrations normalized to 1.0  # len MAXISOTOPES
    c_sum: float = 0.0  # Sum of concentrations, should be 1.0.
    n: int = 0  # Number of different natural isotopes
    Am: float = 0.0  # Mass of the most abundant isotope

    def __post_init__(self):
        if self.A is None:
            self.A = [0.0] * constants.MAXISOTOPES
        if self.c is None:
            self.c = [0.0] * constants.MAXISOTOPES


@dataclass
class Ion:
    Z: float = 0.0  # Atomic number of ion (non-integer for compat.)
    A: float = 0.0  # Mass of ion in the units of kg
    E: float = 0.0  # Energy of the ion
    I: Isotopes = None  # Data structure for natural isotopes
    p: Point = None  # Three dimensional position of ion
    theta: float = 0.0  # Laboratory theta-angle of the ion [0,PI]
    fii: float = 0.0  # Laboratory fii-angle of the ion [0,2*PI]
    nsct: int = 0  # Number of scatterings of the ion
    status: constants.IonStatus = None  # Status value for ion (stopped, recoiled etc.)
    opt: Ion_opt = None  # Structure for the optimization-variables
    w: float = 0.0  # Statistical weight of the ion
    wtmp: float = 0.0  # Temporary statistical weight of the ion
    time: float = 0.0  # Time since the creation of the ion
    tlayer: int = 0  # Number of the current target layer
    lab: Vector = None  # Translation and rotation of the current coordinate system in the laboratory coordinate system
    type: constants.IonType = None  # Primary, secondary etc.
    hist: Rec_hist = None  # Variables saved at the moment of the recoiling event
    dist: float = 0.0  # Distance to the next ERD-scattering point
    virtual: bool = False  # Did we only hit the virtual detector area
    hit: List[Point] = None  # Hit points to the detector layers  # len MAXLAYERS
    Ed: List[float] = None  # Ion energy in the detector layers  # len MAXLAYERS
    dt: List[float] = None   # Passing times in the detector layers  # len MAXLAYERS
    scale: bool = False  # TRUE if we have a scaling ion
    effrecd: float = 0.0  # Parameter for scaling the effective thickness of recoil material
    trackid: int = 0  # originally int64_t
    scatindex: int = 0  # Index of the scattering table for this ion
    ion_i: int = 0  # "i"th recoil in the track
    E_nucl_loss_det: float = 0.0

    def __post_init__(self):
        if self.I is None:
            self.I = Isotopes()
        if self.p is None:
            self.p = Point()
        if self.opt is None:
            self.opt = Ion_opt()
        if self.lab is None:
            self.lab = Vector()
        if self.hist is None:
            self.hist = Rec_hist()
        if self.hit is None:
            self.hit = [Point() for _ in range(constants.MAXLAYERS)]
        if self.Ed is None:
            self.Ed = [0.0] * constants.MAXLAYERS
        if self.dt is None:
            self.dt = [0.0] * constants.MAXLAYERS


@dataclass
class Cross_section:
    emin: float = 0.0
    emax: float = 0.0
    estep: float = 0.0
    b: List[float] = None  # Initialized later


@dataclass
class Potential:
    n: int = 0
    d: int = 0
    u: List[Point2] = None

    def __post_init__(self):
        if self.u is None:
            self.u = []


@dataclass
class Scattering:
    angle: List[List[float]] = None  # 2D array for scattering angles  # len [EPSNUM][YNUM]
    cross: Cross_section = None  # Data for maximum impact parameters
    logemin: float = 0.0  # Logarithm for minimum reduced energy
    logymin: float = 0.0  # Logarithm for minimum reduced impact parameter
    logediv: float = 0.0  # Logarithm for difference of reduced energies
    logydiv: float = 0.0  # Logarithm for difference reduced impact parameters
    a: float = 0.0  # Reduced unit for screening length
    E2eps: float = 0.0  # Constant for changing energy unit to reduced energy
    # Potential *pot;  # Originally commented out

    def __post_init__(self):
        if self.angle is None:
            self.angle = [[0.0] * constants.YNUM for _ in range(constants.EPSNUM)]
        if self.cross is None:
            self.cross = Cross_section()


@dataclass
class SNext:
    d: float = 0.0  # Distance to the next scattering point
    natom: int = 0  # Number of the scattering atom
    r: bool = False  # Boolean for true mc-scattering (instead of eg. interface crossing)


@dataclass
class Surface:
    z: List[List[float]] = None  # Depth data leveled below zero  # len [NSURFDATA][NSURFDATA]
    nsize: int = 0  # Number of data points per side
    size: float = 0.0  # Physical length of the side
    depth: float = 0.0  # Maximum depth in the data
    step: float = 0.0  # size/(nsize - 1)
    origin: Point = None  # (Random) origin of the surface x-y -data
    sin_r: float = 0.0  # sinus of the rotation (for speed)
    cos_r: float = 0.0  # cosinus of the rotation (for speed)
    move: float = 0.0

    def __post_init__(self):
        if self.z is None:
            self.z = [[0.0] * constants.NSURFDATA for _ in range(constants.NSURFDATA)]
        if self.origin is None:
            self.origin = Point()


@dataclass
class Target_ele:
    """Target element"""
    Z: float = 0.0
    A: float = 0.0


@dataclass
class Target_sto:
    """Target stopping values"""
    vel: List[float] = None  # len MAXSTO
    sto: List[float] = None  # len MAXSTO
    stragg: List[float] = None  # len MAXSTO
    stodiv: float = 0.0
    n_sto: int = 0

    def __post_init__(self):
        if self.vel is None:
            self.vel = [0.0] * constants.MAXSTO
        if self.sto is None:
            self.sto = [0.0] * constants.MAXSTO
        if self.stragg is None:
            self.stragg = [0.0] * constants.MAXSTO


@dataclass
class Target_layer:
    """Target layer"""
    natoms: int = 0  # Number of different elements in this layer
    dlow: float = 0.0  # lower depth of the target layer
    dhigh: float = 0.0  # higher depth of the target layer
    atom: List[int] = None  # Array of the indices of the target elements  # len MAXATOMS
    N: List[float] = None  # Array of the atomic densities  # len MAXATOMS
    Ntot: float = 0.0  # Total atomic density in the layer
    sto: List[Target_sto] = None  # Electronic stopping for different ions  # len g.nions
    type: constants.TargetType = None  # Type of the target layer
    gas: bool = False  # Whether the target layer is gas or not
    stofile_prefix: str = None  # len MAXSTOFILEPREFIXLEN

    def __post_init__(self):
        if self.atom is None:
            self.atom = [0] * constants.MAXATOMS
        if self.N is None:
            self.N = [0.0] * constants.MAXATOMS

@dataclass
class Plane:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    type: constants.PlaneType = None


@dataclass
class Target:
    minN: float = 0.0  # Minimum atomic density in the target
    ele: List[Target_ele] = None  # Array of different elements in the target  # len MAXELEMENTS
    layer: List[Target_layer] = None  # Array of the target layers  # len MAXLAYERS
    nlayers: int = 0  # Total number of target layers
    ntarget: int = 0  # Number of the layers in the target itself
    natoms: int = 0  # Total number of atoms in different target layers
    recdist: List[Point2] = None  # Recoil material distribution in the target  # len NRECDIST
    nrecdist: int = 0  # Number of points in recoil material distribution
    effrecd: float = 0.0  # Effective thickness of recoil material (nonzero values)
    recmaxd: float = 0.0  # Maximum depth of the recoiling material
    plane: Plane = None  # Target surface plane
    efin: List[float] = None  # Energies below which the recoil can't reach the surface  # len NRECDIST
    recpar: List[Point2] = None  # Recoiling solid angle fitted parameters  # len MAXLAYERS
    angave: float = 0.0  # Average recoiling half-angle
    surface: Surface = None  # Data structure for the rough surface
    # len [NSENE][NSANGLE]:
    cross: List[List[float]] = None  # Cross section table for scattering relative to the Rutherford cross sections as a function of ion lab. energy and lab scattering angle
    table: bool = False  # True if cross section table is available

    def __post_init__(self):
        if self.ele is None:
            self.ele = [Target_ele() for _ in range(constants.MAXELEMENTS)]
        if self.layer is None:
            self.layer = [Target_layer() for _ in range(constants.MAXLAYERS)]
        if self.recdist is None:
            self.recdist = [Point2() for _ in range(constants.NRECDIST)]
        if self.plane is None:
            self.plane = Plane()
        if self.efin is None:
            self.efin = [0.0] * constants.NRECDIST
        if self.recpar is None:
            self.recpar = [Point2() for _ in range(constants.MAXLAYERS)]
        if self.surface is None:
            self.surface = Surface()
        if self.cross is None:
            self.cross = [[0.0] * constants.NSANGLE for _ in range(constants.NSENE)]


@dataclass
class Line:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    type: constants.LineType = None


@dataclass
class Rect:
    p: List[Point]  # len 4


@dataclass
class Circ:
    p: Point = None
    r: float = 0.0

    def __post_init__(self):
        self.p = Point()


@dataclass
class Det_foil:
    type: constants.FoilType = None  # Rectangular or circular
    virtual: bool = False  # Is this foil virtual
    dist: float = 0.0  # Distance from the target center
    angle: float = 0.0  # Angle of the foil
    size: List[float] = None  # Diameter for circular, width and height for rect.  # len 2
    plane: Plane = None  # Plane of the detector foil
    center: Point = None  # Point of the center of the foil

    def __post_init__(self):
        if self.size is None:
            self.size = [0.0] * 2


@dataclass
class Detector:
    type: constants.DetectorType = None  # TOF, GAS or FOIL
    angle: float = 0.0  # Detector angle relative to the beam direction
    nfoils: int = 0  # Number of foils in the detector
    virtual: bool = False  # Do we have the virtual detector
    vsize: List[float] = None  # Size of the virtual detector relative to the real  # len 2
    tdet: List[int] = None  # Layer numbers for the timing detectors  # len 2
    edet: List[int] = None  # Layer number(s) for energy detector (layers)  # len MAXLAYERS
    thetamax: float = 0.0  # maximum half-angle of the detector (spot size incl.)
    vthetamax: float = 0.0  # maximum half-angle of the virtual detector
    foil: List[Det_foil] = None  # Detector foil data structure  # len MAXFOILS
    vfoil: Det_foil = None  # Virtual detector data structure

    def __post_init__(self):
        if self.vsize is None:
            self.vsize = [0.0] * 2
        if self.tdet is None:
            self.tdet = [0] * 2
        if self.edet is None:
            self.edet = [0] * constants.MAXLAYERS
        if self.foil is None:
            self.foil = [Det_foil() for _ in range(constants.MAXFOILS)]
        if self.vfoil is None:
            self.vfoil = Det_foil()
