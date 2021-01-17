import logging

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


def output_erd(g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


def output_data(g: o.Global) -> None:
    """Output how many ions have been calculated"""
    if g.simstage == c.SimStage.PRESIMULATION:
        nion = g.cion
        nmaxion = g.npresimu
    else:
        nion = g.cion - g.npresimu
        nmaxion = g.nsimu - g.npresimu

    if nmaxion > 100 and nion % (nmaxion // 100) == 0:
        logging.info(f"Calculated {nion} of {nmaxion} ions {100.0 * nion / nmaxion:.0}%")


def output_trackpoint(
        g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector, label: str) -> None:
    raise NotImplementedError
