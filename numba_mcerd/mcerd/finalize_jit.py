from pathlib import Path

import numba as nb

import numba_mcerd.mcerd.objects_jit as oj
from numba_mcerd.mcerd import enums


# TODO: I/O not supported in Numba
# @nb.njit(cache=True)
def finalize(g: oj.Global, master: oj.Master) -> None:
    """Output statistics of ion finishes to g.master.fpout"""
    dat_lines = ["Statistics of ion finishes: \n"]

    for i in range(enums.IonType.SECONDARY.value + 1):
        for j in range(len(enums.IonStatus)):
            dat_lines.append(f"{i:2d} {j:2d}: {g.finstat[i][j]:5d}\n")

    if g.simtype == enums.SimType.RBS:
        raise NotImplementedError

    with Path(master.fpdat).open("a") as f:
        f.writelines(dat_lines)
