from typing import List

import numba as nb
import numpy as np

import numba_mcerd.mcerd.objects_jit as oj
# import numba_mcerd.mcerd.constants as c
# from numba_mcerd import logging_jit


# Called once in preprocessing
def init_params(g: oj.Global, target: oj.Target, argv: List[str]) -> None:
    raise NotImplementedError


# Called once in preprocessing and after pre-simulation
@nb.njit(cache=True, nogil=True)
def init_recoiling_angle(target: oj.Target) -> None:
    """Calculate target.angave (average recoiling half-angle)"""
    angle = np.zeros(shape=(50, 2), dtype=np.float64)
    depth = np.zeros(shape=(50, 2), dtype=np.float64)
    layer = np.zeros(shape=50, dtype=np.int64)
    i = 1
    j = 0
    k = 0
    finished = False
    delay = 0

    rlow = target.recdist[i - 1].x
    rhigh = target.recdist[i].x
    tlow = target.layer[j].dlow
    thigh = target.layer[j].dhigh

    while not (tlow <= rlow <= thigh):
        j += 1
        delay = 1
        tlow = target.layer[j].dlow
        thigh = target.layer[j].dhigh

    high = tlow

    while not finished:
        low = high

        recd_nonzero = target.recdist[i - 1].y > 0 or target.recdist[i].y > 0

        if rhigh < thigh:
            high = rhigh
            rlow = rhigh  # Unused
            i += 1
            if i < target.nrecdist:
                rhigh = target.recdist[i].x
            else:
                finished = True
        else:
            high = thigh
            tlow = thigh  # Unused
            j += 1
            delay = 1
            if j < target.ntarget:
                thigh = target.layer[j].dhigh
            else:
                finished = True

        if not finished and recd_nonzero:
            n = j - delay
            depth[k][0] = low
            depth[k][1] = high
            layer[k] = n
            angle[k][0] = low * target.recpar[n].x + target.recpar[n].y
            angle[k][1] = high * target.recpar[n].x + target.recpar[n].y
            k += 1

        delay = 0

    if j >= target.ntarget and recd_nonzero:
        n = j - delay
        depth[k][0] = low
        depth[k][1] = high
        layer[k] = n
        angle[k][0] = low * target.recpar[n].x + target.recpar[n].y
        angle[k][1] = high * target.recpar[n].x + target.recpar[n].y
        k += 1
        delay = 0

    angave = 0.0
    lensum = 0.0
    for i in range(k):
        for j in range(2):
            # Formatting: f"{layer[i]:3} {depth[i][j] / c.C_NM:10.3} {angle[i][j] / c.C_DEG:10.3}"
            # logging_jit.info(f"{int(layer[i])} {float(depth[i][j] / c.C_NM)} {float(angle[i][j] / c.C_DEG)}")
            pass

        n = layer[i]
        a = target.recpar[n].x
        b = target.recpar[n].y
        x0 = depth[i][0]
        x1 = depth[i][1]
        angave += ((1.0 / 3.0) * a**2 * (x0**2 + x0 * x1**3) + a * b * (x0 + x1) + b**2) * (x1 - x0)
        lensum += x1 - x0

    target.angave = angave / lensum

    # Formatting: f"angave={target.angave / c.C_DEG:10.7}"
    # logging_jit.info(f"angave={target.angave / c.C_DEG}")


# Called once in preprocessing
def init_io(g: oj.Global, ion: oj.Ion, target: oj.Target) -> None:
    raise NotImplementedError
