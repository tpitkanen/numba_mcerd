import math
from typing import Tuple

import numba as nb
from numba.experimental import jitclass

from numba_mcerd import logging_jit
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_jit as oj


EPS = 1e-5
MAXSTEP = 100
DEPS = 1e-7
DISTEPS = 1e-6


@jitclass({
    "x0": nb.float64,
    "i_y2": nb.float64,
    "i_ey2": nb.float64,
    "tmp2": nb.float64,
    "e": nb.float64,
    "y": nb.float64
})
class Opt:
    def __init__(self):
        self.x0 = 0.0
        self.i_y2 = 0.0
        self.i_ey2 = 0.0
        self.tmp2 = 0.0
        self.e = 0.0
        self.y = 0.0


@nb.njit(cache=True)
def scattering_angle(pot: oj.Potential, ion_opt_e, ion_opt_y) -> float:
    """Get scattering angle for ion's specific optimization state (ion.opt)"""
    opt = Opt()

    opt.e = ion_opt_e
    opt.y = ion_opt_y
    opt.i_y2 = 1.0 / opt.y**2
    opt.i_ey2 = 1.0 / (opt.e * opt.y**2)

    opt.x0 = mindist(pot, opt)
    opt.tmp2 = opt.x0**2 / (opt.y**2 * opt.e)

    theta = c.C_PI - 4.0 * simpson(0.0 + DEPS, 1.0 - DEPS, pot, opt)

    if theta < 0.0:
        theta = abs(theta)
    if theta > c.C_PI:
        theta -= c.C_PI

    return theta


@nb.njit(cache=True)
def Ut(pot: oj.Potential, x: float) -> float:
    if x < 0:
        return pot.u[0].y
    if x > pot.u[pot.n - 2].x:
        return pot.u[pot.n - 2].y

    # Unused
    # xstep = pot.u[1].x - pot.u[0].x

    i = int(x * pot.d)

    xlow = pot.u[i].x
    ylow = pot.u[i].y
    yhigh = pot.u[i + 1].y

    value = ylow + (yhigh - ylow) * (x - xlow) * pot.d
    return value


# TODO: Check that this still produces the same results as before
@nb.njit(cache=True)
def Angint(u: float, pot: oj.Potential, opt: Opt) -> float:
    u2 = u**2

    tmp0 = opt.x0 / (1.0 - u2)

    tmp1 = Ut(pot, opt.x0) / opt.x0 - Ut(pot, tmp0) / tmp0

    # tmp1 in pieces, to demonstrate strange performance issues
    # tmp1 = Ut(pot, opt.x0) / opt.x0
    # ut_out = Ut(pot, tmp0)
    # ut_out /= tmp0  # Commenting out this part causes performance issues
    # tmp1 -= ut_out

    tmp2 = opt.tmp2 / u2
    tmp3 = 2.0 - u2 + tmp2 * tmp1

    # Needed if fastmath=True, though the flag doesn't improve performance
    # if tmp3 < EPS:  # EPS is probably too big
    #     tmp3 = EPS

    value = 1.0 / math.sqrt(tmp3)

    return value


@nb.njit(cache=True)
def trapezoid(a: float, b: float, value: float, nextn: int, pot: oj.Potential,
              n: int, opt: Opt) -> Tuple[float, int]:
    if n == 1:
        value = 0.5 * (b - a) * (Angint(a, pot, opt) + Angint(b, pot, opt))
        nextn = 1
    else:
        h = (b - a) / nextn
        x = a + 0.5 * h
        sum_ = 0.0
        for i in range(1, nextn + 1):
            sum_ += Angint(x, pot, opt)
            x += h
        value = 0.5 * (value + (b - a) * sum_ / nextn)
        nextn *= 2

    return value, nextn


@nb.njit(cache=True)
def simpson(a: float, b: float, pot: oj.Potential, stmp: Opt) -> float:
    old_s = old_st = -1e30
    st = 0.0  # Also value in trapezoid
    nextn = 1

    for i in range(1, MAXSTEP + 1):
        st, nextn = trapezoid(a, b, st, nextn, pot, i, stmp)
        s = (4.0 * st - old_st) / 3.0

        tmp = abs((s - old_s) / old_s)
        if i > 2 and tmp < EPS:
            break
        old_s = s
        old_st = st

    if i > MAXSTEP:
        logging_jit.warning("Maximum number of integration points reached")

    return s


@nb.njit(cache=True)
def mindist(pot: oj.Potential, opt: Opt) -> float:
    x1 = (1 + math.sqrt(1 + 4 * opt.y**2 * opt.e**2)) / (2 * opt.e)

    while Psi(x1, pot, opt) < 0.0:
        x1 *= 2

    x2 = x1
    diff = 1e6
    diff_ok = True
    while diff_ok:
        x1 = x2
        # C code doesn't fail on floating point zero divisions, but they
        # still occur
        try:
            x2 = x1 - DEPS / (Psi(x1 + DEPS, pot, opt) / Psi(x1, pot, opt) - 1)
        except:  # Numba doesn't support specific types (e.g. ZeroDivisionError)
            # logging_jit.error(f"Division by zero, x1={float(x1)} x2={float(x2)}")
            logging_jit.error(f"Division by zero in mindist")
            break
        diffold = diff
        diff = abs(x2 - x1)

        diff_ok = DISTEPS < diff < diffold

    return x2


@nb.njit(cache=True)
def Psi(x: float, pot: oj.Potential, opt: Opt) -> float:
    value = x**2 * opt.i_y2 - x * Ut(pot, x) * opt.i_ey2 - 1.0
    return value
