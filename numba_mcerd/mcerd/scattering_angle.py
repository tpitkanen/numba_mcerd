import logging
import math
from dataclasses import dataclass
from typing import Tuple

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


EPS = 1e-5
MAXSTEP = 100
DEPS = 1e-7
DISTEPS = 1e-6


@dataclass
class Opt:
    x0: float = 0.0
    i_y2: float = 0.0
    i_ey2: float = 0.0
    tmp2: float = 0.0
    e: float = 0.0
    y: float = 0.0


def scattering_angle(pot: o.Potential, ion: o.Ion) -> float:
    opt = Opt()

    opt.e = ion.opt.e
    opt.y = ion.opt.y
    opt.i_y2 = 1 / opt.y**2
    opt.i_ey2 = 1 / (opt.e * opt.y**2)

    opt.x0 = mindist(pot, opt)
    opt.tmp2 = opt.x0**2 / (opt.y**2 * opt.e)

    theta = c.C_PI - 4 * simpson(0.0 + DEPS, 1.0 - DEPS, pot, opt)

    if theta < 0:
        theta = abs(theta)
    if theta > c.C_PI:
        theta -= c.C_PI

    return theta


def Angint(u: float, pot: o.Potential, opt: Opt) -> float:
    u2 = u**2
    tmp0 = opt.x0 / (1 - u2)
    tmp1 = Ut(pot, opt.x0) / opt.x0 - Ut(pot, tmp0) / tmp0
    tmp2 = opt.tmp2 / u2
    tmp3 = 2 - u2 + tmp2 * tmp1

    value = 1.0 / math.sqrt(tmp3)

    return value


def Ut(pot: o.Potential, x: float) -> float:
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


def trapezoid(a: float, b: float, value: float, nextn: int, pot: o.Potential,
              n: int, opt: Opt) -> Tuple[float, int]:
    # TODO: value and next_n are static in the original code

    nextn = 1
    h = x = sum_ = 0

    if n == 1:
        value = 0.5 * (b - a) * (Angint(a, pot, opt) + Angint(b, pot, opt))
        nextn = 1
    else:
        h = (b - a) / nextn
        x = a + 0.5 * h
        for i in range(1, nextn + 1):
            sum_ += Angint(x, pot, opt)
            x += h
        value = 0.5 * (value + (b - a) * sum_ / nextn)
        nextn *= 2

    return value, nextn


def simpson(a: float, b: float, pot: o.Potential, stmp: Opt) -> float:
    old_s = old_st = -1e30
    st = 0  # Also value in trapezoid
    nextn = 1

    for i in range(1, MAXSTEP + 1):
        st, nextn = trapezoid(a, b, st, nextn, pot, i, stmp)
        s = (4 * st - old_st) / 3

        tmp = abs((s - old_s) / old_s)
        if i > 2 and tmp < EPS:
            break
        old_s = s
        old_st = st

    if i > MAXSTEP:
        logging.warning("Maximum number of integration points reached")

    return s


def mindist(pot: o.Potential, opt: Opt) -> float:
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
        except ZeroDivisionError:
            logging.warning(f"Division by zero, {x1=} {x2=}")
            break
        diffold = diff
        diff = abs(x2 - x1)

        diff_ok = DISTEPS < diff < diffold

    return x2


def Psi(x: float, pot: o.Potential, opt: Opt) -> float:
    value = x**2 * opt.i_y2 - x * Ut(pot, x) * opt.i_ey2 - 1
    return value
