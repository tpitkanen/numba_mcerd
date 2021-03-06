import numba as nb
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.objects_dtype as od
from numba_mcerd import list_conversion as lc
from numba_mcerd.mcerd import enums


# TODO: Create another buffer type that has formats for each row.
#  FIN_STOP is 10.3f, FIN_TRANS is 12.6f


def create_range_buffer(g: oj.Global, additional_multiplier: float = 1.0) -> od.Buffer:
    """Create a buffer for range output

    Args:
        g: global settings. `nsimu` affects the buffer size
        additional_multiplier: additional multiplier for buffer size (for
            use with multithreading).

    Returns:
        A buffer object
    """
    t = lc.TypeInt
    width = 2

    # Output occurs at most once per simulated ion
    length = int(g.nsimu * additional_multiplier)

    dt = od.get_buffer_dtype(length, width)
    range_buf = np.zeros(1, dtype=dt)[0]
    range_buf["types"] = np.array([t.STR, t.FLOAT])
    range_buf["formats"] = np.array(["", "12.6f"], dtype="U5")

    return range_buf


@nb.njit(cache=True, nogil=True)
def finish_ion(g: oj.Global, ion: oj.Ion, range_buf: od.Buffer) -> None:
    """Output information about stopped or transmitted ions to g.master.fprange"""
    if ion.status == enums.IonStatus.FIN_STOP:
        range_buf["col_i"] = 0

        lc.set_buf(range_buf, ord("R"))
        lc.set_buf(range_buf, ion.p.z / c.C_NM)  # 10.3f

        range_buf["row_i"] += 1
    elif ion.status == enums.IonStatus.FIN_TRANS:
        range_buf["col_i"] = 0

        lc.set_buf(range_buf, ord("T"))
        lc.set_buf(range_buf, ion.E / c.C_MEV)  # 12.6f

        range_buf["row_i"] += 1
