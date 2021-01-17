from typing import Tuple

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


NPRESIMU = 500


def finish_presimulation(g: o.Global, detector: o.Detector, recoil: o.Ion) -> None:
    raise NotImplementedError


def analyze_presimulation(g: o.Global, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


def presimu_comp_angle(a: o.Presimu, b: o.Presimu) -> int:
    raise NotImplementedError


def presimu_comp_depth(a: o.Presimu, b: o.Presimu) -> int:
    raise NotImplementedError


def fit_linear(x: float, y: float, n: float, a: float, b: float) \
        -> Tuple[float, float, float, float, float]:
    raise NotImplementedError



