import math

import numba
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj
# import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import scattering_angle_jit


def scattering_table(g: o.Global, ion: o.Ion, target: o.Target, scat: o.Scattering,
                     pot: oj.Potential, natom: int) -> None:
    """Create a lookup table for scattering (energies?)"""
    targetZ = target.ele[natom].Z
    targetA = target.ele[natom].A

    scat.a = 0.8854 * c.C_BOHR_RADIUS / (ion.Z ** 0.23 + targetZ ** 0.23)
    scat.E2eps = 4.0 * c.C_PI * c.C_EPSILON0 * scat.a * targetA /\
                 ((ion.A + targetA) * ion.Z * targetZ * c.C_E**2)

    emin = math.log(g.emin * scat.E2eps)
    emax = math.log(g.ionemax * scat.E2eps)

    ymin = math.log(1.0 / (2.0 * math.exp(emax) * math.tan(0.5 * c.C_PI * c.MAXANGLE / 180.0)))
    ymax = math.log(1.0 / (2.0 * scat.a * target.minN**(1.0/3.0)))

    estep = (emax - emin) / (c.EPSNUM - 2)
    ystep = (ymax - ymin) / (c.YNUM - 2)

    scat.logemin = emin
    scat.logymin = ymin

    scat.logediv = 1.0 / estep
    scat.logydiv = 1.0 / ystep

    with g.master.fpout.open("a") as f:
        f.write(f"a, E2eps {scat.a} {scat.E2eps}\n")
        f.write(f"emin, emax: {emin} {emax}\n")
        f.write(f"ymin, ymax: {ymin} {ymax}\n")
        f.write(f"estep, ystep: {estep} {ystep}\n")

    scat_matrix = np.array(scat.angle, dtype=np.float64)  # Numba seems to do float64 instead of float32
    opt_e, opt_y = main_math(scat_matrix, pot, emin, estep, ymin, ystep)
    scat.angle = scat_matrix
    ion.opt.e = opt_e
    ion.opt.y = opt_y


# Can't be parallelized automatically. A manual option would be to split
# scat_matrix into pieces, and then calculate exp_e and exp_y like so:
#   exp_e = math.exp(emin + (i-1) * estep)
#   exp_y = math.exp(ymin + (j-1) * ystep)
# scat_matrix[i][0] and scat_matrix[0][j] need to be done separately
@numba.njit(cache=True)
def main_math(scat_matrix, pot, emin, estep, ymin, ystep):
    exp_e = exp_y = 0.0

    for i in range(1, scat_matrix.shape[0]):
        exp_e = math.exp(emin)
        scat_matrix[i][0] = exp_e
        y = ymin
        for j in range(1, scat_matrix.shape[1]):
            # print(i, j)  # Good for debugging performance
            exp_y = math.exp(y)
            scat_matrix[i][j] = scattering_angle_jit.scattering_angle(pot, exp_e, exp_y)
            y += ystep
        emin += estep

    y = ymin
    for j in range(1, scat_matrix.shape[1]):
        scat_matrix[0][j] = math.exp(y)
        y += ystep

    return exp_e, exp_y


@numba.njit(parallel=True)
def main_math_parallelized(scat_matrix, pot, emin, estep, ymin, ystep):
    """Can be parallelized, but doesn't offer any speedup. Incomplete, don't use."""
    exp_e = exp_y = 0.0

    for i in numba.prange(1, scat_matrix.shape[0]):
        # exp_e = math.exp(emin + (i - 1) * estep)  # Warning for this
        # scat_matrix[i][0] = exp_e
        for j in numba.prange(1, scat_matrix.shape[1]):
            # exp_y = math.exp(ymin + (j - 1) * ystep)  # Warning for this
            scat_matrix[i][j] = scattering_angle_jit.scattering_angle(
                pot,
                math.exp(emin + (i - 1) * estep),
                math.exp(ymin + (j - 1) * ystep))

    # for j in numba.prange(1, scat_matrix.shape[1]):
    #     y = ymin + (j - 1) * ystep
    #     scat_matrix[0][j] = math.exp(y)

    return exp_e, exp_y
