import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import enums


def finalize(g: o.Global) -> None:
    """Output statistics of ion finishes to g.master.fpout"""
    dat_lines = ["Statistics of ion finishes: "]

    for i in range(enums.IonType.SECONDARY.value + 1):
        for j in range(len(enums.IonStatus)):
            dat_lines.append(f"{i:2d} {j:2d}: {g.finstat[i][j]:5d}")

    if g.simtype == enums.SimType.SIM_RBS:
        raise NotImplementedError

    g.master.fpdat.write_text("\n".join(dat_lines))
