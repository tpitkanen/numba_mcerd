import logging

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import erd_detector


# Untested (except for initial "if" which returns False in pre-simulation)
def _output_tof(g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    # https://stackoverflow.com/questions/3167494/how-often-does-python-flush-to-a-file

    # TODO: return early instead (removes one indentation level)
    if cur_ion.type == c.IonType.SECONDARY and (
            cur_ion.tlayer == target.nlayers or
            erd_detector.is_in_energy_detector(g, cur_ion, target, detector, True) or
            cur_ion.status == c.IonStatus.FIN_OUT_DET and g.output_misses):
        n = cur_ion.tlayer - target.ntarget  # Foil number  # TODO: Rename
        line_parts = []

        # Actually output if: a secondary particle (not the primary beam) is either:
        # 1. In the last layer
        # 2. In the energy detector or beyond (if we are outputting trackpoints and energy detector layer is given)
        # 3. The particle missed (exited detector, hit an aperture), but output misses is on.
        line_parts.append(
            f'{"S" if cur_ion.scale else "R"} {"V" if cur_ion.virtual else "R"} {"R" if g.simtype else "S"}')
        if g.output_trackpoints:
            line_parts.append(f"{cur_ion.trackid:12d}")
        if g.advanced_output:
            line_parts.append(str(cur_ion.status.value))
            line_parts.append(str(n))
        z = cur_ion.hist.tar_recoil.p.z

        if g.rough:
            raise NotImplementedError

        line_parts.append(f"{cur_ion.E / c.C_KEV:8.4f}")

        line_parts.append(f"{int(cur_ion.hist.Z + 0.5):3i}")
        line_parts.append(f"{cur_ion.hist.A / c.C_U:6.2f}")

        line_parts.append(f"{z / c.C_NM:10.4f}")
        line_parts.append(f"{cur_ion.w:14.7e}")

        if g.advanced_output:
            line_parts.append(f"{cur_ion.hist.ion_E / c.C_KEV:8.4f}")
            line_parts.append(f"{cur_ion.hist.ion_recoil.theta / c.C_DEG:7.3f}")
            line_parts.append(f"{cur_ion.hist.w:10.4e}")
            line_parts.append(f"{cur_ion.dt[0] / c.C_NS:7.3f}")
            line_parts.append(f"{cur_ion.dt[1] / c.C_NS:7.3f}")
        else:
            line_parts.append(f"{cur_ion.dt[1] - cur_ion.dt[0] / c.C_NS:10.3f}")  # ToF

        if not g.advanced_output:
            line_parts.append(f"{cur_ion.hit[0].x / c.C_MM:7.2f}")
            line_parts.append(f"{cur_ion.hit[0].y / c.C_MM:7.2f}")
        else:
            # x, y coordinates of hits
            for j in (0,  # First aperture
                      detector.tdet[0] - target.ntarget,  # T1
                      detector.tdet[1] - target.ntarget,  # T2
                      detector.edet[0] - target.ntarget,  # Energy detector
                      n):
                line_parts.append(f"{cur_ion.hit[j].x / c.C_MM:6.2f}")
                line_parts.append(f"{cur_ion.hit[j].y / c.C_MM:6.2f}")

        if g.advanced_output:  # Energy losses in layers
            for i in range(target.nlayers - target.ntarget - 1):
                line_parts.append(f"{(cur_ion.Ed[i] - cur_ion.Ed[i + 1]) / c.C_KEV:7.4f}")
                if (cur_ion.Ed[i] - cur_ion.Ed[i + 1]) / c.C_MEV < 0.0 and i:
                    logging.warning(f"Ion (trackid={cur_ion.trackid}, ion_i={cur_ion.ion_i}) gains energy between layers i={i} and i+1={i+1}")
            if cur_ion.tlayer == target.nlayers:
                line_parts.append(f"{(cur_ion.Ed[target.nlayers - target.ntarget - 1] - cur_ion.E) / c.C_MEV:7.4f}")
            else:
                line_parts.append(f"{0.0:7.4f}")

        # TODO: Does write_text add a linebreak automatically?
        # This causes a trailing space but it's the way to add a linebreak.
        # Original code generates a trailing space if g.advanced_output is False
        line_parts.append("\n")

        output = " ".join(line_parts)
        # TODO: Problematic for multi-threading, and possibly slow
        g.master.fperd.write_text(output)


def _output_gas(g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


def output_erd(g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    if detector.type == c.DetectorType.TOF:
        _output_tof(g, cur_ion, target, detector)
        return
    elif detector.type == c.DetectorType.GAS:
        _output_gas(g, cur_ion, target, detector)
    # Other types are ignored, namely DET_FOIL


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
