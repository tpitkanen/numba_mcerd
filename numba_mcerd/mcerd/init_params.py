import math
from pathlib import Path
from typing import List

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


# Called once in preprocessing
def init_params(g: o.Global, target: o.Target, argv: List[str]) -> None:
    """Initialize general parameters"""
    g.master.args = argv
    g.ionemax = -1.0
    g.emin = 10 * c.C_KEV
    g.E0 = -1.0
    g.nsimu = 10
    g.predata = False

    g.beamprof = c.BeamProf.BEAM_NONE
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
def init_recoiling_angle(g: o.Global, ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


# Called once in preprocessing
def init_io(g: o.Global, ion: o.Ion, target: o.Target) -> None:
    """Initialize paths for I/O"""
    # TODO: Set these to data/out/
    #       Check that these paths can be opened
    g.basename = f"{g.master.args[1]}.{g.seed}"

    g.master.fpout = Path(g.basename + ".out")
    g.master.fpdat = Path(g.basename + ".dat")
    # g.master.fpdebug  # Skipped
    g.master.fperd = Path(g.basename + ".erd")
    g.master.fprange = Path(g.basename + ".range")
    g.master.fptrack = Path(g.basename + ".track")
