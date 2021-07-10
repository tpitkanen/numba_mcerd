"""Enums from general.h"""

from enum import IntEnum


class RndPeriod(IntEnum):
    """Random number generator value ranges"""
    RND_CLOSED = 0  # [a, b] (both closed)
    RND_OPEN = 1    # ]a, b[ (both open)
    RND_LEFT = 2    # ]a, b] (left open)
    RND_RIGHT = 3   # [a, b[ (right open)
    RND_SEED = 4  # Set the seed number for random number generator  # Not really needed
    # RND_CONT = 1  # Do not reset random number generator  # Not needed


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
    SIM_ERD = 1
    SIM_RANGE = 2  # TODO: What is this?
    SIM_RBS = 3


class SimStage(IntEnum):
    """Simulation stage"""
    ANYSIMULATION = 0
    PRESIMULATION = 1
    REALSIMULATION = 2
    # SCALESIMULATION = 3  # This is actually not used


class RecWidth(IntEnum):
    """Recoiling angle width"""
    REC_NARROW = 1  # Recoiling to a narrow solid angle
    REC_WIDE = 0  # Recoiling to a wide solid angle


class BeamProf(IntEnum):
    """Beam profile"""
    BEAM_NONE = 0  # No beam divergence
    BEAM_GAUSS = 1  # Beam profile gaussian
    BEAM_FLAT = 2  # Beam profile flat
    BEAM_DIST = 3  # Beam profile a given distribution


class TargetType(IntEnum):
    TARGET_FILM = 0
    TARGET_BULK = 1
    TARGET_ABSORBER = 2


class PlaneType(IntEnum):
    GENERAL_PLANE = 0
    X_PLANE = 1
    Z_PLANE = 2


class LineType(IntEnum):
    L_GENERAL = 0
    L_XYPLANE = 1
    L_YAXIS = 2


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
