from typing import List

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


# Probably not needed
def next_ion():
    raise NotImplementedError


# Probably not needed
def prev_ion():
    raise NotImplementedError


# Maybe needed
def copy_ions():
    raise NotImplementedError


def cascades_create_additional_ions(
        g: o.Global, detector: o.Detector, target: o.Target, ions: List[o.Ion]) -> None:
    if not g.cascades:
        return
    raise NotImplementedError("Additional ion creation case not implemented")

