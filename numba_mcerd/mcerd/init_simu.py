import math

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import scattering_angle


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

    ymin = math.log(1 / (2 * math.exp(emax) * math.tan(0.5 * c.C_PI * c.MAXANGLE / 180)))
    ymax = math.log(1 / (2 * scat.a * target.minN**(1/3)))

    estep = (emax - emin) / (c.EPSNUM - 2)
    ystep = (ymax - ymin) / (c.YNUM - 2)

    scat.logemin = emin
    scat.logymin = ymin

    scat.logediv = 1 / estep
    scat.logydiv = 1 / ystep

    e = emin

    text = f"""a, E2eps {scat.a} {scat.E2eps}
emin, emax: {emin} {emax}
ymin, ymax: {ymin} {ymax}
estep, ystep: {estep} {ystep}
"""
    with g.master.fpout.open("a") as f:
        f.write(text)

    for i in range(1, c.EPSNUM):
        exp_e = math.exp(e)
        scat.angle[i][0] = exp_e
        ion.opt.e = exp_e
        y = ymin
        for j in range(1, c.YNUM):
            ion.opt.y = math.exp(y)
            # FIXME: Angles should be slightly under pi in the beginning
            scat.angle[i][j] = scattering_angle.scattering_angle(pot, ion)
            y += ystep
        e += estep

    y = ymin
    for j in range(1, c.YNUM):
        scat.angle[0][j] = math.exp(y)
        y += ystep
