from typing import List, Optional

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


# These functions are only needed for simulations with cascades.
# In other cases, they can be replaced with cur_ion = ions[PRIMARY] etc.


# Untested
# def next_ion(g: o.Global, cur_ion: o.Ion, ions_moving: o.Ion):
def next_ion(g: o.Global, cur_ion_index: int) -> Optional[int]:
    """Get the next ion's index. Original code returns a pointer instead."""
    next_ion_index = cur_ion_index + 1
    if g.simtype == c.SimType.SIM_RBS and next_ion_index == c.IonType.TARGET_ATOM.value:
        # In RBS ions_moving[TARGET_ATOM] has a special role
        next_ion_index += 1
    if next_ion_index >= g.ncascades:
        # This is used to limit recoil cascades
        return None  # TODO: Raise instead?
    return next_ion_index


# Untested
def prev_ion(g: o.Global, cur_ion_index: int) -> Optional[int]:
    """Get the previous ion's index. Original code returns a pointer instead."""
    prev_ion_index = cur_ion_index - 1
    if g.simtype == c.SimType.SIM_RBS and prev_ion_index == c.IonType.TARGET_ATOM.value:
        prev_ion_index -= 1
    if prev_ion_index <= 0:
        return None
    return prev_ion_index


def copy_ions():
    raise NotImplementedError


def cascades_create_additional_ions(
        g: o.Global, detector: o.Detector, target: o.Target, ions: List[o.Ion]) -> None:
    if not g.cascades:
        return
    raise NotImplementedError("Additional ion creation case not implemented")

