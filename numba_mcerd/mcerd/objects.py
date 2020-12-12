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


@dataclass
class Global:
    pass  # TODO


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







