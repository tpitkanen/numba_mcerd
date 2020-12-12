from enum import Enum

# (RNG stuff)

# SMALLNUM = 1e-10

# RNG type


class IonStatus(Enum):
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
    NIONSTATUS = 11     # Number of different ion statuses


# Constants which may effect the physics of simulation:
MAXANGLE = 177  # Minimum of maximum scattering angles

# TODO: lines 52 .. 109


class TargetType(Enum):
    TARGET_FILM = 0
    TARGET_BULK = 1
    TARGET_ABSORBER = 2

# TODO: lines 117 .. 144

# TODO: Lines 436 .. 443
