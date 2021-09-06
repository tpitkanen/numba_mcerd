import numba as nb

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_jit as oj
from numba_mcerd.mcerd import enums


# TODO: I/O not supported in Numba
# @nb.njit(cache=True)
def finish_ion(g: oj.Global, ion: oj.Ion) -> None:
    """Output information about stopped or transmitted ions to g.master.fprange"""
    # TODO: Problematic for multi-threading, and possibly slow
    if ion.status == enums.IonStatus.FIN_STOP:
        with g.master.fprange.open("a") as f:
            f.write(f"R {ion.p.z / c.C_NM:10.3f}\n")
    if ion.status == enums.IonStatus.FIN_TRANS:
        with g.master.fprange.open("a") as f:
            f.write(f"T {ion.E / c.C_MEV:12.6f}\n")
