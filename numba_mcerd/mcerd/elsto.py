import math

import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd import config

NSTO = 500


class ElstoError(Exception):
    """Error in elsto"""


# class GstoStoppingType(Enum):
#     NONE = 0
#     NUCL = 1
#     ELE = 2
#     TOT = 3
#     STRAGG = 4


# Storage for (temporary) precalculated energies.
# Remove this once the real GSTO is done
const_gsto = {
    "sto": None,
    "stragg": None
}


def calc_stopping_and_straggling_const(g: o.Global, ion: o.Ion, target: o.Target, nlayer: int, gsto_index: int) -> None:
    """Look up precalculated stopping and straggling energies from files in GSTO_ROOT"""
    if const_gsto["sto"] is None:
        const_gsto["sto"] = np.loadtxt(config.CONST_STO_PATH)
    if const_gsto["stragg"] is None:
        const_gsto["stragg"] = np.loadtxt(config.CONST_STRAGG_PATH)

    layer = target.layer[nlayer]
    sto = layer.sto[ion.scatindex]

    minv = 0.0
    maxv = 1.1 * math.sqrt(2.0 * g.ionemax / ion.A)

    # z1 = round(ion.Z)  # Unused
    vstep = (maxv - minv) / NSTO

    sto.stodiv = 1.0 / vstep
    sto.n_sto = NSTO

    sto.sto = const_gsto["sto"][gsto_index]
    sto.stragg = const_gsto["stragg"][gsto_index]

    # Outer loop is not needed for vel
    for j in range(sto.n_sto):
        sto.vel[j] = j * vstep


# TODO: Implement proper GSTO and finish this
def calc_stopping_and_straggling(g: o.Global, ion: o.Ion, target: o.Target, nlayer: int) -> None:
    """Calculate stopping and straggling energies for atoms in the current layer"""
    # nion = ion.scatindex
    layer = target.layer[nlayer]
    sto = layer.sto[ion.scatindex]

    minv = 0.0
    maxv = 1.1 * math.sqrt(2.0 * g.ionemax / ion.A)

    # There's a comment in the original source code that the divisor
    # could/should be NSTO-1, but it's like this for compatibility with stodiv
    vstep = (maxv - minv) / NSTO
    z1 = round(ion.Z)

    # Probably not needed
    for i in range(c.MAXSTO):
        sto.sto[i] = 0.0

    sto.stodiv = 1.0 / vstep
    sto.n_sto = NSTO

    for i in range(layer.natoms):
        p = layer.atom[i]
        z2 = int(target.ele[p].Z)
        # if not jibal_jsto_auto_assign(g.jibal.gsto, z1, z2) ...

    # if not jibal_gsto_load_all(g.jibal.gsto) ...

    for i in range(layer.natoms):
        p = layer.atom[i]
        z2 = int(target.ele[p].Z)
        for j in range(sto.n_sto):
            v = j * vstep
            sto.vel[j] = v
            # em = jibal.energy_per_mass(v)
            # stop = jibal.jibal_gsto_get_em(...)
            # stragg = jibal.jibal_gsto_get_em(...)
            # if (j > 0 and stop == 0.0) or not math.isfinite(stop): ...
            # if (j > 0 and stragg == 0.0) or not math.isfinite(stragg): ...
            # sto.sto[j] += stop
            # sto.stragg[j] += stragg

    raise NotImplementedError
