import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import enums


def finish_ion(g: o.Global, ion: o.Ion) -> None:
    """Output information about stopped or transmitted ions to g.master.fprange"""
    # TODO: Problematic for multi-threading, and possibly slow
    if ion.status == enums.IonStatus.FIN_STOP:
        with g.master.fprange.open("a") as f:
            f.write(f"R {ion.p.z / c.C_NM:10.3f}\n")
    if ion.status == enums.IonStatus.FIN_TRANS:
        with g.master.fprange.open("a") as f:
            f.write(f"T {ion.E / c.C_MEV:12.6f}\n")
