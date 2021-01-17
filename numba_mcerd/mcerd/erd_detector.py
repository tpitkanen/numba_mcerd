import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


def is_in_energy_detector(g: o.Global, ion: o.Ion, target: o.Target, detector: o.Detector, beyond_detector_ok: bool) -> bool:
    raise NotImplementedError


def move_to_erd_detector(g: o.Global, ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


def get_line_params(d: o.Vector, p: o.Point) -> o.Line:
    raise NotImplementedError


def get_line_plane_cross(line: o.Line, plane: o.Plane, cross: o.Point) -> int:
    raise NotImplementedError


def get_distance(p1: o.Point, p2: o.Point) -> float:
    raise NotImplementedError


def is_on_right_side(p1: o.Point, d: o.Vector, p2: o.Point) -> int:
    raise NotImplementedError


def is_in_foil(p: o.Point, foil: o.Det_foil) -> bool:
    raise NotImplementedError


def get_angle_between_points(p: o.Point, p1: o.Point, p2: o.Point) -> float:
    raise NotImplementedError
