"""Constants from general.h"""

from enum import Enum


################################################################################
# Lines 16 .. 143 (roughly)
################################################################################


class RndPeriod(Enum):
    """Random number generator value ranges"""
    RND_CLOSED = 0  # [a, b] (both closed)
    RND_OPEN = 1    # ]a, b[ (both open)
    RND_LEFT = 2    # ]a, b] (left open)
    RND_RIGHT = 3   # [a, b[ (right open)
    # TODO: Is this really used for settings the random seed?
    #       random.rnd doesn't seem like it.
    RND_SEED = 4  # Set the seed number for random number generator
    # RND_CONT = 1  # Do not reset random number generator

# SMALLNUM = 1e-10


class IonStatus(Enum):
    """Ion status"""
    NOT_FINISHED = 0    # Ion continues
    FIN_STOP = 1        # Ion stopped
    FIN_TRANS = 2       # Ion transmitted through the sample
    FIN_BS = 3          # Ion backscattered from the sample
    FIN_RECOIL = 4      # Recoil came out of the sample
    FIN_NO_RECOIL = 5   # Could not recoil atom (because of the kinematics)
    FIN_MISS_DET = 6    # Recoil didn't hit one of the detector aperture
    FIN_OUT_DET = 7     # Recoil came out in the middle of a detector layers
    FIN_DET = 8         # Recoil came out from the last detector layer
    FIN_MAXDEPTH = 9    # Primary ion reached maximum recoil depth
    FIN_RECTRANS = 10   # Recoil was transmitted through the sample
    # NIONSTATUS = 11   # Number of different ion statuses  # len(IonStatus)


# Constants which may effect the physics of simulation:
MAXANGLE = 177  # Minimum of maximum scattering angles


# Maximums for arrays etc. Probably not needed
#
# EPSNUM = 50  # Number of energies in scattering angle table
# YNUM = 100  # Number of impact factors in scattering angle table
# EPSIMP = 200  # Number of points in maximum impact factor array
# N_RAND_DIST = 500  # Number of points in random distribution
# N_INPUT = 20000  # Maximum number of lines in input file
# LINE = 1024  # Maximum length of line in input files
# PATH = ""  # Path for input files
# NFILE = 512  # Maximum name length of the input file
# MAXSTO = 500  # Maximum number of values in stopping power array
# MAXELOSS = 0.01  # Maximum relative change in electonic energy loss
# MAXENEFF = 2000  # Maximum number of point in the energy eff. curve
# NSURFDATA = 520  # Maximum width of the surface topography image
#
# NRECDIST = 500
#
# MAXLAYERS = 50  # Number of different target layers
# MAXATOMS = 20  # Number of different atoms in the target layer
# MAXELEMENTS = 20  # Maximum total number of different atoms in the target
# NANGDIST = 200  # Number of points in reaction angular distribution
#
# MAXISOTOPES = 20


class IonType(Enum):
    """Ion type"""
    PRIMARY = 0
    SECONDARY = 1
    TARGET_ATOM = 2
    GAS_RECOIL = 3

# TODO:
# FORW = 1
# BACK = 2

# TODO:
# STOPPING = 0
# STRAGGLING = 1


class ScatteringType(Enum):
    """Element(?) scattering type"""
    NO_SCATTERING = 0
    MC_SCATTERING = 1
    ERD_SCATTERING = 2


class SimType(Enum):
    """Simulation type"""
    SIM_ERD = 1
    SIM_RANGE = 2  # TODO: What is this?
    SIM_RBS = 3


class SimStage(Enum):
    """Simulation stage"""
    ANYSIMULATION = 0
    PRESIMULATION = 1
    REALSIMULATION = 2
    # SCALESIMULATION = 3  # This is actually not used


class RecWidth(Enum):
    """Recoiling angle width"""
    REC_NARROW = 1  # Recoiling to a narrow solid angle
    REC_WIDE = 0  # Recoiling to a wide solid angle


C_NM = 1.0e-9  # From rough_surface.c
SCALE_DEPTH = (10.0*C_NM)  # All scaling particles are recoiled below this

PRESIMU_LEVEL = 0.01  # Ratio of ions for defining recoil solid angle


class BeamProf(Enum):
    """Beam profile"""
    BEAM_NONE = 0  # No beam divergence
    BEAM_GAUSS = 1  # Beam profile gaussian
    BEAM_FLAT = 2  # Beam profile flat
    BEAM_DIST = 3  # Beam profile a given distribution


class TargetType(Enum):
    TARGET_FILM = 0
    TARGET_BULK = 1
    TARGET_ABSORBER = 2


# # Booleans
# TRUE = 1
# FALSE = 0


# TODO: What is this? Used in erd_detector
class PlaneType(Enum):
    GENERAL_PLANE = 0
    X_PLANE = 1
    Z_PLANE = 2


# TODO: What is this? Used in erd_detector
class LineType(Enum):
    L_GENERAL = 0
    L_XYPLANE = 1
    L_YAXIS = 2


# TODO: Is this needed? Used in erd_detector
class Cross(Enum):
    """Whether line crosses plane or not"""
    CROSS = True
    NO_CROSS = False


NSENE = 2000  # Maximum number of energies in the cross section table
NSANGLE = 200  # Maximum number of angle in the cross section table


C_KEV = 1.602176634e-16  # from jibal_units.h

TRACKPOINT_INTERVAL = (10.0*C_KEV)
TRACKPOINT_LIMIT_SMALL = (50.0*C_KEV)
TRACKPOINT_INTERVAL_SMALL = (1.0*C_KEV)

MAXSTOFILEPREFIXLEN = 200


################################################################################
# Lines 436 .. 443
################################################################################


class DetectorType(Enum):
    """Detector type"""
    DET_TOF = 1
    DET_GAS = 2
    DET_FOIL = 3


MAXFOILS = 10  # Maximum number of foils in detector


class FoilType(Enum):
    """Foil type/shape"""
    FOIL_CIRC = 0
    FOIL_RECT = 1
