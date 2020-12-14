"""Objects from general.h"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import constants

@dataclass
class Point:  # TODO: should this be Point3?
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Point2:
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
    data_file: Path = None  # fdata[NFILE]
    erd_out: Path = None  # ERD output
    data_out: Path = None
    range_out: Path = None
    track_out: Path = None
    # Not needed:
    # argc
    # fpout?
    # fpdebug


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
    presimu: Presimu = None  # Data structure for the presimulation
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
    # jibal: jibal  # TODO
    nomc: bool = False

    def __post_init__(self):
        self.bspot = Point2()
        self.presimu = Presimu()
        self.master = Master()
        # self.jibal = Jibal()


@dataclass
class Ion_opt:
    valid: bool  # Boolean for the validity of these opt-variables
    cos_theta: float  # Cosinus of laboratory theta-angle [-1,1]
    sin_theta: float  # Sinus of laboratory theta-angle [-1,1]
    e: float  # Dimensionless energy for the next scattering
    y: float  # Dimensionless impact parameter  -"-


@dataclass
class Vector:
    p: Point  # Cartesian coordinates of the object
    theta: float  # Direction of the movement of the object
    fii: float  # Direction of the movement of the object
    v: float  # Object velocity


@dataclass
class Rec_hist:
    tar_recoil: Vector  # Recoil vector in target coord. at recoil moment
    ion_recoil: Vector  # Recoil vector in prim. ion coord. at recoil moment
    lab_recoil: Vector  # Recoil vector in lab. coord. at recoil moment
    tar_primary: Vector  # Primary ion vector in target coordinates
    lab_primary: Vector  # Primary ion vector in lab coordinates
    ion_E: float
    recoil_E: float
    layer: int  # Recoiling layer
    nsct: int  # Number of scatterings this far
    time: float  # Recoiling time
    w: float  # Statistical weight at the moment of the recoiling
    Z: float  # Atomic number of the secondary atom
    A: float  # Mass of the secondary atom in the scattering


@dataclass
class Isotopes:
    A: List[float]  # List of the isotope masses in kg  # len MAXISOTOPES
    c: List[float]  # Isotope concentrations normalized to 1.0  # len MAXISOTOPES
    c_sum: float  # Sum of concentrations, should be 1.0.
    n: int  # Number of different natural isotopes
    Am: float  # Mass of the most abundant isotope


@dataclass
class Ion:
    Z: float  # Atomic number of ion (non-integer for compat.)
    A: float  # Mass of ion in the units of kg
    E: float  # Energy of the ion
    I: Isotopes  # Data structure for natural isotopes
    p: Point  # Three dimensional position of ion
    theta: float  # Laboratory theta-angle of the ion [0,PI]
    fii: float  # Laboratory fii-angle of the ion [0,2*PI]
    nsct: int  # Number of scatterings of the ion
    status: constants.IonStatus  # Status value for ion (stopped, recoiled etc.)
    opt: Ion_opt  # Structure for the optimization-variables
    w: float  # Statistical weight of the ion
    wtmp: float  # Temporary statistical weight of the ion
    time: float  # Time since the creation of the ion
    tlayer: int  # Number of the current target layer
    lab: Vector  # Translation and rotation of the current coordinate system in the laboratory coordinate system
    type: constants.IonType  # Primary, secondary etc.
    hist: Rec_hist  # Variables saved at the moment of the recoiling event
    dist: float  # Distance to the next ERD-scattering point
    virtual: bool  # Did we only hit the virtual detector area
    hit: List[Point]  # Hit points to the detector layers
    Ed: List[float]  # Ion energy in the detector layers
    dt: List[float]  # Passing times in the detector layers
    scale: bool  # TRUE if we have a scaling ion
    effrecd: float  # Parameter for scaling the effective thickness of recoil material
    trackid: int  # originally int64_t
    scatindex: int  # Index of the scattering table for this ion
    ion_i: int  # "i"th recoil in the track
    E_nucl_loss_det: float


@dataclass
class Cross_section:
    emin: float
    emax: float
    estep: float
    b: float  # TODO: This was a pointer


@dataclass
class Potential:
    n: int
    d: int
    u: Point2


@dataclass
class Scattering:
    angle: List[List[float]]  # 2D array for scattering angles  # len [EPSNUM][YNUM]
    cross: Cross_section  # Data for maximum impact parameters
    logemin: float  # Logarithm for minimum reduced energy
    logymin: float  # Logarithm for minimum reduced impact parameter
    logediv: float  # Logarithm for difference of reduced energies
    logydiv: float  # Logarithm for difference reduced impact parameters
    a: float  # Reduced unit for screening length
    E2eps: float  # Constant for changing energy unit to reduced energy
    # Potential *pot;  # Originally commented out


@dataclass
class SNext:
    d: float  # Distance to the next scattering point
    natom: int  # Number of the scattering atom
    r: bool  # Boolean for true mc-scattering (instead of eg. interface crossing)


@dataclass
class Surface:
    z: List[List[float]]  # Depth data leveled below zero  # len [NSURFDATA][NSURFDATA]
    nsize: int  # Number of data points per side
    size: float  # Physical length of the side
    depth: float  # Maximum depth in the data
    step: float  # size/(nsize - 1)
    origin: Point  # (Random) origin of the surface x-y -data
    sin_r: float  # sinus of the rotation (for speed)
    cos_r: float  # cosinus of the rotation (for speed)
    move: float


@dataclass
class Target_ele:
    """Target element"""
    Z: float
    A: float


@dataclass
class Target_sto:
    """Target stopping values"""
    vel: List[float]  # len MAXSTO
    sto: List[float]  # len MAXSTO
    stragg: List[float]  # len MAXSTO
    stodiv: float
    n_sto: int


@dataclass
class Target_layer:
    """Target layer"""
    natoms: int  # Number of different elements in this layer
    dlow: float  # lower depth of the target layer
    dhigh: float  # higher depth of the target layer
    atom: List[int]  # Array of the indices of the target elements  # len MAXATOMS
    N: List[float]  # Array of the atomic densities  # len MAXATOMS
    Ntot: float  # Total atomic density in the layer
    sto: Target_sto  # Electronic stopping for different ions
    type: constants.TargetType  # Type of the target layer
    gas: bool  # Whether the target layer is gas or not
    stofile_prefix: str  # len MAXSTOFILEPREFIXLEN


@dataclass
class Plane:
    a: float
    b: float
    c: float
    type: constants.PlaneType


@dataclass
class Target:
    minN: float  # Minimum atomic density in the target
    ele: List[Target_ele]  # Array of different elements in the target  # len MAXELEMENTS
    layer: List[Target_layer]  # Array of the target layers  # len MAXLAYERS
    nlayers: int  # Total number of target layers
    ntarget: int  # Number of the layers in the target itself
    natoms: int  # Total number of atoms in different target layers
    recdist: Point2  # Recoil material distribution in the target  # len NRECDIST
    nrecdist: int  # Number of points in recoil material distribution
    effrecd: float  # Effective thickness of recoil material (nonzero values)
    recmaxd: float  # Maximum depth of the recoiling material
    plane: Plane  # Target surface plane
    efin: List[float]  # Energies below which the recoil can't reach the surface  # len NRECDIST
    recpar: List[Point2]  # Recoiling solid angle fitted parameters  # len MAXLAYERS
    angave: float  # Average recoiling half-angle
    surface: Surface  # Data structure for the rough surface
    # len [NSENE][NSANGLE]:
    cross: List[List[float]]  # Cross section table for scattering relative to the Rutherford cross sections as a function of ion lab. energy and lab scattering angle
    table: bool  # True if cross section table is available


@dataclass
class Line:
    a: float
    b: float
    c: float
    d: float
    type: constants.LineType


@dataclass
class Rect:
    p: List[Point]  # len 4


@dataclass
class Circ:
    p: Point
    r: float


@dataclass
class Det_foil:
    type: constants.FoilType  # Rectangular or circular
    virtual: bool  # Is this foil virtual
    dist: float  # Distance from the target center
    angle: float  # Angle of the foil
    size: List[float]  # Diameter for circular, width and height for rect.  # len 2
    plane: Plane  # Plane of the detector foil
    center: Point  # Point of the center of the foil


@dataclass
class Detector:
    type: constants.DetectorType  # TOF, GAS or FOIL
    angle: float  # Detector angle relative to the beam direction
    nfoils: int  # Number of foils in the detector
    virtual: bool  # Do we have the virtual detector
    vsize: List[float]  # Size of the virtual detector relative to the real  # len 2
    tdet: List[int]  # Layer numbers for the timing detectors  # len 2
    edet: List[int]  # Layer number(s) for energy detector (layers)  # len MAXLAYERS
    thetamax: float  # maximum half-angle of the detector (spot size incl.)
    vthetamax: float  # maximum half-angle of the virtual detector
    foil: List[Det_foil]  # Detector foil data structure  # len MAXFOILS
    vfoil: Det_foil  # Virtual detector data structure







