import math
from typing import Tuple

import numpy as np
import numba

import numba_mcerd.mcerd.objects_jit as oj


NPOINTS = 50
POTMIN = 1.0e-10
POTERR = 1.0e-3
MAXPOINTS = 30000
XMAX = 1000.0


@numba.njit()
def make_screening_table() -> oj.Potential:
    """Create and return a Potential object.

    The function can't be cached or parallelized because it returns an object.
    """
    xmax = get_max_x()
    n = get_npoints(xmax)
    xstep = xmax / (n - 1)
    d = round(1 / xstep)

    pot = oj.Potential(n, d)

    x = 0
    for i in range(pot.n):
        pot.ux[i] = x
        pot.uy[i] = U(x)
        x += xstep

    return pot


@numba.njit(cache=True)
def make_screening_table_cached() -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Create and return the pieces needed for a Potential object"""
    xmax = get_max_x()
    n = get_npoints(xmax)
    xstep = xmax / (n - 1)
    d = round(1 / xstep)

    ux = np.zeros(n, dtype=np.float32)
    uy = np.zeros(n, dtype=np.float32)

    x = 0
    for i in range(n):
        ux[i] = x
        uy[i] = U(x)
        x += xstep

    return n, d, ux, uy


@numba.njit(cache=True)
def get_max_x() -> float:
    x = 1.0
    while U(x) > POTMIN and x < XMAX:
        x *= 2

    x0 = x / 2
    while U(x) < POTMIN:
        x = 0.5 * (x0 + x)

    return x


@numba.njit(cache=True)
def get_npoints(xmax: float) -> int:
    maxdiff = 0.0
    x = 0.0
    xlow = 0.0
    xhigh = xmax
    xstep = xmax / NPOINTS

    for _ in range(NPOINTS):
        pred1 = 0.5 * (U(x) + U(x + xstep))
        accur1 = U(x + xstep / 2)
        err1 = abs(1.0 - pred1 / accur1)
        if err1 > maxdiff:
            xlow = x
            xhigh = x + xstep
            maxdiff = err1

    errors_above_threshold = True
    while errors_above_threshold:
        xstep = (xhigh - xlow) / 2
        pred1 = 0.5 * (U(xlow) + U(xlow + xstep))
        pred2 = 0.5 * (U(xhigh) + U(xhigh - xstep))
        accur1 = U(xlow + xstep / 2)
        accur2 = U(xhigh - xstep / 2)
        err1 = abs(1.0 - pred1 / accur1)
        err2 = abs(1.0 - pred2 / accur2)
        if err1 > err2:
            xhigh = xlow + xstep
        else:
            xlow = xhigh - xstep
        xstep /= 2

        errors_above_threshold = not (err1 < POTERR and err2 < POTERR)

    npoints = xmax / (xhigh - xlow) + 1
    return round(npoints)


@numba.njit(cache=True)
def U(x: float) -> float:
    """Universal screening potential function"""
    return (0.18175 * math.exp(-3.1998 * x)
            + 0.50986 * math.exp(-0.94229 * x)
            + 0.28022 * math.exp(-0.4029 * x)
            + 0.028171 * math.exp(-0.20162 * x))
