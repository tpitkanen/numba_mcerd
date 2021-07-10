import logging
import math
from pathlib import Path
from typing import List

import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import enums


# Called once in preprocessing
def init_params(g: o.Global, target: o.Target, argv: List[str]) -> None:
    """Initialize general parameters"""
    g.master.args = argv
    g.ionemax = -1.0
    g.emin = 10 * c.C_KEV
    g.E0 = -1.0
    g.nsimu = 10
    g.predata = False

    g.beamprof = enums.BeamProf.NONE
    g.beamdiv = 0.0

    g.nions = 2
    g.ncascades = 10

    g.nmclarge = 0
    g.nmc = 0

    g.costhetamin = math.cos(1 * c.C_DEG)

    target.natoms = 0
    target.nlayers = 0
    # target.natoms = 0  # TODO: This is repeated twice in the original, why?

    # Not needed, values are already zero
    # for i in range(len(g.finstat)):
    #     for j in range(len(g.finstat[i])):
    #         g.finstat[i][j] = 0

    g.rough = False
    g.output_misses = False
    g.output_trackpoints = False
    g.cascades = False
    g.advanced_output = False
    g.nomc = False


# Called once in preprocessing and once for last pre-simulation ion
def init_recoiling_angle(target: o.Target) -> None:
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
            logging.info(f"{layer[i]:3} {depth[i][j] / c.C_NM:10.3} {angle[i][j] / c.C_DEG:10.3}")

        n = layer[i]
        a = target.recpar[n].x
        b = target.recpar[n].y
        x0 = depth[i][0]
        x1 = depth[i][1]
        angave += ((1.0 / 3.0) * a**2 * (x0**2 + x0 * x1**3) + a * b * (x0 + x1) + b**2) * (x1 - x0)
        lensum += x1 - x0

    target.angave = angave / lensum

    logging.info(f"angave {target.angave / c.C_DEG:10.7}")


# Called once in preprocessing
def init_io(g: o.Global, ion: o.Ion, target: o.Target) -> None:
    """Initialize paths for I/O"""
    # TODO: Set these to data/out/
    #       Check that these paths can be opened
    g.basename = f"{g.master.args[1]}.{g.seed}"

    g.master.fpout = Path(g.basename + ".out")
    g.master.fpdat = Path(g.basename + ".dat")
    # g.master.fpdebug = Path(g.basename + ".debug")  # Skipped
    g.master.fperd = Path(g.basename + ".erd")
    g.master.fprange = Path(g.basename + ".range")
    g.master.fptrack = Path(g.basename + ".track")

    # Clear previous files
    g.master.fpout.write_text("")
    g.master.fpdat.write_text("")
    # g.master.fpdebug.write_text("")  # Skipped
    g.master.fperd.write_text("")
    g.master.fprange.write_text("")
    g.master.fptrack.write_text("")
