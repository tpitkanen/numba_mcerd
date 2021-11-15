import logging

import numba as nb
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.list_conversion as lc
from numba_mcerd import logging_jit
from numba_mcerd.mcerd import erd_detector_jit, enums


def create_erd_buffer(g: oj.Global) -> od.Buffer:
    """Create a buffer for ERD output based on `g` settings"""
    t = lc.TypeInt
    if g.advanced_output and g.output_trackpoints:
        raise NotImplementedError
    elif g.advanced_output:
        raise NotImplementedError
    elif g.output_trackpoints:
        raise NotImplementedError
    else:
        width = 11
        types = np.array(
            [t.STR, t.STR, t.STR, t.FLOAT, t.INT, t.FLOAT, t.FLOAT, t.FLOAT, t.FLOAT, t.FLOAT, t.FLOAT])
        formats = np.array(["", "", "", "8.4f", "3d", "6.2f", "10.4f", "14.7e", "10.3f", "7.2f", "7.2f"], dtype="U5")

    # TODO: Figure out a good way to determine length
    length = int((g.npresimu + g.nsimu) * 0.2)

    dt = od.get_buffer_dtype(length, width)
    erd_buf = np.zeros(1, dtype=dt)[0]
    erd_buf["types"] = types
    erd_buf["formats"] = formats

    return erd_buf


@nb.njit(cache=True)
def _output_tof(g: oj.Global, master: oj.Master, cur_ion: oj.Ion, target: oj.Target,
                detector: oj.Detector, buf: od.Buffer) -> None:
    # https://stackoverflow.com/questions/3167494/how-often-does-python-flush-to-a-file

    buf["col_i"] = 0
    n = cur_ion.tlayer - target.ntarget  # Foil number  # TODO: Rename

    # TODO: These could be combined into one np.float64 (3*1 bytes)
    lc.set_buf(buf, ord("S") if cur_ion.scale else ord("R"))
    lc.set_buf(buf, ord("V") if cur_ion.virtual else ord("R"))
    lc.set_buf(buf, ord("R") if g.simtype == enums.SimType.ERD else ord("S"))

    if g.output_trackpoints:
        lc.set_buf(buf, cur_ion.trackid)  # 12d
    if g.advanced_output:
        lc.set_buf(buf, cur_ion.status)
        lc.set_buf(buf, n)
    z = cur_ion.hist.tar_recoil.p.z

    if g.rough:
        raise NotImplementedError

    lc.set_buf(buf, cur_ion.E / c.C_MEV)  # 8.4f

    lc.set_buf(buf, int(cur_ion.hist.Z + 0.5))  # 3d
    lc.set_buf(buf, cur_ion.hist.A / c.C_U)  # 6.2f

    lc.set_buf(buf, z / c.C_NM)  # 10.4f
    lc.set_buf(buf, cur_ion.w)  # 14.7e

    if g.advanced_output:
        lc.set_buf(buf, cur_ion.hist.ion_E / c.C_KEV)  # 8.4f
        lc.set_buf(buf, cur_ion.hist.ion_recoil.theta / c.C_DEG)  # 7.3f
        lc.set_buf(buf, cur_ion.hist.w)  # 10.4e
        lc.set_buf(buf, cur_ion.dt[0] / c.C_NS)  # 7.3f
        lc.set_buf(buf, cur_ion.dt[1] / c.C_NS)  # 7.3f
    else:
        lc.set_buf(buf, (cur_ion.dt[1] - cur_ion.dt[0]) / c.C_NS)  # 10.3f  # ToF

    if not g.advanced_output:
        lc.set_buf(buf, cur_ion.hit[0].x / c.C_MM)  # 7.2f
        lc.set_buf(buf, cur_ion.hit[0].y / c.C_MM)  # 7.2f
    else:
        # x, y coordinates of hits
        for j in (0,  # First aperture
                  detector.tdet[0] - target.ntarget,  # T1
                  detector.tdet[1] - target.ntarget,  # T2
                  detector.edet[0] - target.ntarget,  # Energy detector
                  n):
            lc.set_buf(buf, cur_ion.hit[j].x / c.C_MM)  # 6.2f
            lc.set_buf(buf, cur_ion.hit[j].y / c.C_MM)  # 6.2f

    if g.advanced_output:  # Energy losses in layers
        for i in range(target.nlayers - target.ntarget - 1):
            lc.set_buf(buf, (cur_ion.Ed[i] - cur_ion.Ed[i + 1]) / c.C_KEV)  # 7.4f
            if (cur_ion.Ed[i] - cur_ion.Ed[i + 1]) / c.C_MEV < 0.0 and i:
                # logging.warning(f"Ion (trackid={cur_ion.trackid}, ion_i={cur_ion.ion_i}) gains energy between layers i={i} and i+1={i+1}")
                logging_jit.warning("Ion gains energy between layers")
        if cur_ion.tlayer == target.nlayers:
            lc.set_buf(buf, (cur_ion.Ed[target.nlayers - target.ntarget - 1] - cur_ion.E) / c.C_MEV)  # 7.4f
        else:
            lc.set_buf(buf, 0.0)  # 7.4f

    buf["row_i"] += 1

    # Original code generates a trailing space if g.advanced_output is False


# TODO: I/O not supported in Numba
@nb.njit(cache=True)
def _output_gas(g: oj.Global, master: oj.Master, cur_ion: oj.Ion, target: oj.Target,
                detector: oj.Detector) -> None:
    raise NotImplementedError


@nb.njit(cache=True)
def output_erd(g: oj.Global, master: oj.Master, cur_ion: oj.Ion, target: oj.Target,
               detector: oj.Detector, erd_buf: od.Buffer) -> None:
    """Output ERD information to master.fperd"""
    if detector.type == enums.DetectorType.TOF:
        # Actually output if: a secondary particle (not the primary beam) is either:
        # 1. In the last layer
        # 2. In the energy detector or beyond (if we are outputting trackpoints and energy detector layer is given)
        # 3. The particle missed (exited detector, hit an aperture), but output misses is on.
        if cur_ion.type == enums.IonType.SECONDARY and (
                cur_ion.tlayer == target.nlayers or
                erd_detector_jit.is_in_energy_detector(g, cur_ion, target, detector, True) or
                cur_ion.status == enums.IonStatus.FIN_OUT_DET and g.output_misses):
            _output_tof(g, master, cur_ion, target, detector, erd_buf)
        return
    elif detector.type == enums.DetectorType.GAS:
        _output_gas(g, master, cur_ion, target, detector)
    # Other types are ignored, namely DET_FOIL


def output_data(g: oj.Global) -> None:
    """Output how many ions have been calculated"""
    if g.simstage == enums.SimStage.PRE:
        nion = g.cion
        nmaxion = g.npresimu
    else:
        nion = g.cion - g.npresimu
        nmaxion = g.nsimu - g.npresimu

    if nmaxion > 100 and nion % (nmaxion // 100) == 0:
        logging.info(f"Calculated {nion} of {nmaxion} ions {100.0 * nion / nmaxion:.0}%")


def output_trackpoint(
        g: oj.Global, cur_ion: oj.Ion, target: oj.Target, detector: oj.Detector, label: str) -> None:
    raise NotImplementedError
