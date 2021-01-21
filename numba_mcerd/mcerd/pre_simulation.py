from typing import Tuple

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import rotate

NPRESIMU = 500


def finish_presimulation(g: o.Global, detector: o.Detector, recoil: o.Ion) -> None:
    """In the presimulation stage we save the recoil angle relative to the
    detector when the recoil comes out of the target. Also the recoil
    depth and layer are saved for later use.
    """
    # Recoil direction in the laboratory coordinate system
    theta_lab, fii_lab = rotate.rotate(recoil.lab.theta, recoil.lab.fii, recoil.theta, recoil.fii)

    # Recoil direction in the detector coordinate system
    theta, fii = rotate.rotate(detector.angle, c.C_PI, theta_lab, fii_lab)

    g.presimu[g.cpresimu].depth = recoil.hist.tar_recoil.p.z
    g.presimu[g.cpresimu].angle = theta
    g.presimu[g.cpresimu].layer = recoil.hist.layer
    g.cpresimu += 1


def analyze_presimulation(g: o.Global, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


def presimu_comp_angle(a: o.Presimu, b: o.Presimu) -> int:
    raise NotImplementedError


def presimu_comp_depth(a: o.Presimu, b: o.Presimu) -> int:
    raise NotImplementedError


def fit_linear(x: float, y: float, n: float, a: float, b: float) \
        -> Tuple[float, float, float, float, float]:
    raise NotImplementedError



