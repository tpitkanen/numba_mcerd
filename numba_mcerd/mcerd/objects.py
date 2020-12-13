"""Objects from general.h"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Point:  # TODO: should be Point3
    x: float
    y: float
    z: float


@dataclass
class Point2:
    x: float
    y: float


@dataclass
class Presimu:
    depth: float = 0.0
    angle: float = 0.0
    layer: int = 0.0


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


# @dataclass
# class Global:
#     mpi: bool  # Boolean for parallel simulation
#     E0: float  # Energy of the primary ion beam
#     nions: int  # Number of different ions in simulation, 1 or 2
#     ncascades: int
#     nsimu: int  # Total number of the simulated ions
#     emin: float  # Minimum energy of the simulation
#     ionemax: float  # Maximum possible ion energy in simulation
#     minangle: float  # Minimum angle of the scattering
#     seed: int  # Seed number of the random number generator
#     cion: int  # Number of the current ion  # TODO: Is this needed?
#     simtype: int_or_bool  # Type of the simulation
#     beamangle: float  # Angle between target surface normal and beam
#     bspot: Point2  # Size of the beam spot
#     simstage: int_or_bool  # Presimulation or real simulation
#     npresimu: int_or_bool  # Number of simulated ions the  presimulation
#     nscale: int_or_bool  # Number of simulated ions per scaling ions
#     nrecave: int_or_bool  # Average number of recoils per primary ion
#     cpresimu: int_or_bool  # Counter of simulated ions the the presimulation
#     *presimu: Presimu  # Data structure for the presimulation
#     predata: int_or_bool  # Presimulation data given in a file
#     master: Master  # Data structure for the MPI-master
#     frecmin: float
#     costhetamax: float
#     costhetamin: float
#     recwidth: int_or_bool  # Recoiling angle width type
#     virtualdet: int_or_bool  # Do we use the virtual detector
#     basename[NFILE]: str
#     finstat[SECONDARY + 1][NIONSTATUS]: int_or_bool
#     beamdiv: int_or_bool  # Angular divergence of the beam, width or FWHM
#     beamprof: int_or_bool  # Beam profile: flat, gaussian, given distribution
#     rough: int_or_bool  # Rough or non-rough sample surface
#     nmclarge: int_or_bool  # Number of rejected (large) MC scatterings (RBS)
#     nmc: int_or_bool  # Number of all MC-scatterings
#     output_trackpoints: int_or_bool  # Bool TRUE/FALSE
#     output_misses: int_or_bool  # Bool TRUE/FALSE
#     cascades: int_or_bool  # Bool TRUE/FALSE
#     advanced_output: int_or_bool  # Bool TRUE/FALSE
#     *jibal: jibal
#     nomc: int_or_bool  # Bool TRUE/FALSE


@dataclass
class Ion_opt:
    pass  # TODO


@dataclass
class Vector:
    pass  # TODO


@dataclass
class Rec_hist:
    pass  # TODO


@dataclass
class Isotopes:
    pass  # TODO


@dataclass
class Ion:
    pass  # TODO


@dataclass
class Cross_section:
    pass  # TODO


@dataclass
class Potential:
    pass  # TODO


@dataclass
class Scattering:
    pass  # TODO


@dataclass
class SNext:
    pass  # TODO


@dataclass
class Surface:
    pass  # TODO


@dataclass
class Target_ele:
    """Target element"""
    pass  # TODO


@dataclass
class Target_sto:
    pass  # TODO


@dataclass
class Target_layer:
    pass  # TODO


@dataclass
class Plane:
    pass  # TODO


@dataclass
class Target:
    pass  # TODO


@dataclass
class Line:
    pass  # TODO


@dataclass
class Rect:
    pass  # TODO


@dataclass
class Circ:
    pass  # TODO


@dataclass
class Def_foil:
    pass  # TODO


@dataclass
class Detector:
    pass  # TODO







