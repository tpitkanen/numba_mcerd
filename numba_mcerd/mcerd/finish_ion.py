import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


def finish_ion(g: o.Global, ion: o.Ion) -> None:
    """Output information about stopped or transmitted ions to g.master.fprange"""
    # TODO: Problematic for multi-threading, and possibly slow
    if ion.status == c.IonStatus.FIN_STOP:
        g.master.fprange.write_text(f"R {ion.p.z / c.C_NM:10.3f}\n")
    if ion.status == c.IonStatus.FIN_TRANS:
        g.master.fprange.write_text(f"T {ion.E / c.C_MEV:12.6f}\n")
