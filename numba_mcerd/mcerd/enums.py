"""Enums from general.h"""

from enum import IntEnum


class RndPeriod(IntEnum):
    """Random number generator value ranges"""
    CLOSED = 0  # [a, b] (both closed)  # Originally RND_CLOSED
    OPEN = 1    # ]a, b[ (both open)    # Originally RND_OPEN
    LEFT = 2    # ]a, b] (left open)    # Originally RND_LEFT
    RIGHT = 3   # [a, b[ (right open)   # Originally RND_RIGHT
    SEED = 4  # Set the seed number for random number generator  # Not really needed  # Originally RND_SEED
    # CONT = 1  # Do not reset random number generator  # Not needed  # Originally RND_CONT


class IonStatus(IntEnum):
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


class IonType(IntEnum):
    """Ion type"""
    PRIMARY = 0
    SECONDARY = 1
    TARGET_ATOM = 2
    GAS_RECOIL = 3


class CoordTransformDirection(IntEnum):
    """Direction for coordinate transform"""
    FORW = 1  # Forwards
    BACK = 2  # Backwards


class IonMode(IntEnum):
    """Ion mode (?)"""
    STOPPING = 0
    STRAGGLING = 1


class ScatteringType(IntEnum):
    """Element(?) scattering type"""
    NO_SCATTERING = 0
    MC_SCATTERING = 1
    ERD_SCATTERING = 2


class SimType(IntEnum):
    """Simulation type"""
    ERD = 1    # Originally SIM_ERD
    RANGE = 2  # Originally SIM_RANGE
    RBS = 3    # Originally SIM_RBS


class SimStage(IntEnum):
    """Simulation stage"""
    ANY = 0   # Originally ANYSIMULATION
    PRE = 1   # Originally PRESIMULATION
    REAL = 2  # Originally REALSIMULATION
    # SCALE = 3  # This is actually not used  # Originally SCALESIMULATION


class RecWidth(IntEnum):
    """Recoiling angle width"""
    NARROW = 1  # Recoiling to a narrow solid angle  # Originally REC_NARROW
    WIDE = 0  # Recoiling to a wide solid angle      # Originally REC_WIDE


class BeamProf(IntEnum):
    """Beam profile"""
    NONE = 0  # No beam divergence                 # Originally BEAM_NONE
    GAUSS = 1  # Beam profile gaussian             # Originally BEAM_GAUSS
    FLAT = 2  # Beam profile flat                  # Originally BEAM_FLAT
    DIST = 3  # Beam profile a given distribution  # Originally BEAM_DIST


class TargetType(IntEnum):
    FILM = 0      # Originally TARGET_FILM
    BULK = 1      # Originally TARGET_BULK
    ABSORBER = 2  # Originally TARGET_ABSORBER


class PlaneType(IntEnum):
    GENERAL_PLANE = 0
    X_PLANE = 1
    Z_PLANE = 2


class LineType(IntEnum):
    GENERAL = 0   # Originally L_GENERAL
    XY_PLANE = 1  # Originally L_XYPLANE
    Y_AXIS = 2    # Originally L_YAXIS


class Cross(IntEnum):
    """Whether line crosses plane or not"""
    CROSS = True
    NO_CROSS = False


class DetectorType(IntEnum):
    """Detector type"""
    TOF = 1     # Originally DET_TOF
    GAS = 2     # Originally DET_GAS
    FOIL = 3    # Originally DET_FOIL


class FoilType(IntEnum):
    """Foil type/shape"""
    CIRC = 0   # Originally FOIL_CIRC
    RECT = 1   # Originally FOIL_RECT
