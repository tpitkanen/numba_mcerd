import logging
import math

import numba as nb
from numba import cuda

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj
# import numba_mcerd.mcerd.symbols as s

from numba_mcerd.mcerd import scattering_angle_jit


# Doesn't need JIT
def calc_cross_sections(g: o.Global, scat: o.Scattering, pot: oj.Potential) -> None:
    """Calculate cross sections for scat.cross"""
    scat.cross.emin = math.log(0.99 * g.emin * scat.E2eps)
    scat.cross.emax = math.log(1.01 * g.ionemax * scat.E2eps)

    scat.cross.estep = (scat.cross.emax - scat.cross.emin) / (c.EPSIMP - 1)

    e = scat.cross.emin

    scat.cross.b = [0.0] * c.EPSIMP
    for i in range(c.EPSIMP):
        scat.cross.b[i] = calc_cross(g.minangle, math.exp(e), scat, pot)
        e += scat.cross.estep


DISTEPS = 5.0e-3
MAXSTEPS = 50


# Doesn't need JIT
def calc_cross(angle: float, e: float, scat: o.Scattering, pot: oj.Potential) -> float:
    # scat is unused because it has been commented out in the original code

    step = 0

    yold = 10.0
    ylogold = math.log(yold)
    ynew = 2.0
    ylognew = math.log(ynew)

    # ion = o.Ion()
    # ion.opt.e = e
    # ion.opt.y = yold

    aold = scattering_angle_jit.scattering_angle(pot, e, yold)
    # ion.opt.y = ynew
    anew = scattering_angle_jit.scattering_angle(pot, e, ynew)

    alogold = math.log(aold)
    alognew = math.log(anew)

    diff_ok = True
    while diff_ok:
        step += 1

        try:
            y = math.exp(
                ylognew
                + (ylogold - ylognew)
                * (math.log(angle) - alognew)
                / (alogold - alognew)
            )
        except ZeroDivisionError:
            logging.warning(f"Division by zero, {alogold=} {alognew=}")
            break
        if y < 0.0:
            y = ynew / 2.0
        # ion.opt.y = y

        # yold = ynew  # Unused
        ylogold = ylognew
        ynew = y
        ylognew = math.log(y)
        # aold = anew  # Unused
        alogold = alognew
        anew = scattering_angle_jit.scattering_angle(pot, e, y)
        alognew = math.log(anew)

        diff = abs(anew - angle) / angle
        diff_ok = diff > DISTEPS and step < MAXSTEPS

    if diff > DISTEPS:
        logging.warning("convergence stopped")  # Originally included "get_impact:"

    return ynew


@nb.njit(cache=True, nogil=True)
def get_cross(ion: oj.Ion, scat: oj.Scattering) -> float:
    """Interpolate the cross section (maximum impact parameter for current ion energy)"""
    e = math.log(ion.E * scat.E2eps)

    i = int((e - scat.cross.emin) / scat.cross.estep)

    b = (scat.cross.b[i]
         + (scat.cross.b[i + 1] - scat.cross.b[i])
         * (e - (i * scat.cross.estep + scat.cross.emin))
         / scat.cross.estep)

    b *= scat.a
    b = c.C_PI * b ** 2

    if not 0 < b < 1e-15:
        # TODO: Print a warning
        raise NotImplementedError

    return b


# TODO: only pass ion.E?
@cuda.jit(device=True)
def get_cross_cuda(ion: oj.Ion, scat: oj.Scattering) -> float:
    e = math.log(ion.E * scat.E2eps)

    i = int((e - scat.cross.emin) / scat.cross.estep)

    b = (scat.cross.b[i]
         + (scat.cross.b[i + 1] - scat.cross.b[i])
         * (e - (i * scat.cross.estep + scat.cross.emin))
         / scat.cross.estep)

    b *= scat.a
    b = c.C_PI * b ** 2

    # if not 0 < b < 1e-15:
    #     raise NotImplementedError

    return b
