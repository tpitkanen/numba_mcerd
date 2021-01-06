import math

import numba
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
# import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import scattering_angle_jit


# TODO: jit-ify this


def scattering_table(g: o.Global, ion: o.Ion, target: o.Target, scat: o.Scattering,
                     pot: o.Potential, natom: int) -> None:
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

    scat.logemin = 1.0 / estep
    scat.logydiv = 1.0 / ystep

    e = emin

    text = f"""a, E2eps {scat.a} {scat.E2eps}
emin, emax: {emin} {emax}
ymin, ymax: {ymin} {ymax}
estep, ystep: {estep} {ystep}
"""
    g.master.fpout.write_text(text)

    # scat_matrix = np.zeros((len()), dtype=np.float32)

    # scat_matrix1 = np.array(scat.angle[0:25], dtype=np.float32)
    # scat_matrix2 = np.array(scat.angle[25:50], dtype=np.float32)
    #
    # main_math(scat_matrix1, pot, e, estep, ymin, ystep)
    # main_math(scat_matrix2, pot, e, estep, ymin, ystep)

    # scat_matrix = np.array(scat.angle[0:1], dtype=np.float32)
    # main_math(scat_matrix, pot, e, estep, ymin, ystep)

    scat_matrix = np.array(scat.angle, dtype=np.float32)
    opt_e, opt_y = main_math(scat_matrix, pot, e, estep, ymin, ystep)
    ion.opt.e = opt_e
    ion.opt.y = opt_y


# @numba.njit(cache=True, parallel=True)
def main_math(scat_matrix, pot, e, estep, ymin, ystep):
    exp_e = exp_y = 0.0

    for i in range(1, scat_matrix.shape[0]):
        exp_e = math.exp(e)
        scat_matrix[i][0] = exp_e
        # ion_opt_e = exp_e
        y = ymin
        for j in range(1, scat_matrix.shape[1]):
            print(i, j)
            # ion_opt_y = math.exp(y)
            exp_y = math.exp(y)
            # scat_matrix[i][j] = scattering_angle_jit.scattering_angle(pot, ion_opt_e, ion_opt_y)
            scat_matrix[i][j] = scattering_angle_jit.scattering_angle(pot, exp_e, exp_y)
            y += ystep
        e += estep

    y = ymin
    for j in range(1, scat_matrix.shape[1]):
        scat_matrix[0][j] = math.exp(y)
        y += ystep

    return exp_e, exp_y
