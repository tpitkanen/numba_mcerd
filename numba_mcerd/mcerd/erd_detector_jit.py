import math
from typing import Tuple

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_jit as oj
from numba_mcerd.mcerd import enums, misc, rotate, virtual_detector


class ErdDetectorError(Exception):
    """Error in ERD detector"""


def is_in_energy_detector(g: oj.Global, ion: oj.Ion, target: oj.Target, detector: oj.Detector,
                          beyond_detector_ok: bool) -> bool:
    raise NotImplementedError


def move_to_erd_detector(g: oj.Global, ion: oj.Ion, target: oj.Target, detector: oj.Detector) -> None:
    """Recoil moves from the target to a detector foil or from one
    detector foil to the next
    """
    raise NotImplementedError


def get_line_params(d: oj.Vector, p: oj.Point) -> oj.Line:
    """Calculate line parameters for a line pointing to the direction d
    and going through point p
    """
    raise NotImplementedError


def get_line_plane_cross(line: oj.Line, plane: oj.Plane) -> Tuple[enums.Cross, oj.Point]:
    raise NotImplementedError


def get_distance(p1: oj.Point, p2: oj.Point) -> float:
    raise NotImplementedError


def is_on_right_side(p1: oj.Point, d: oj.Vector, p2: oj.Point) -> int:
    raise NotImplementedError


def is_in_foil(p: oj.Point, foil: oj.Det_foil) -> bool:
    """(Point p is assumed to be on the plane of circle c)"""
    raise NotImplementedError


def get_angle_between_points(p: oj.Point, p1: oj.Point, p2: oj.Point) -> float:
    raise NotImplementedError
