import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


def output_erd(g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector) -> None:
    raise NotImplementedError


def output_data(g: o.Global) -> None:
    raise NotImplementedError


def output_trackpoint(
        g: o.Global, cur_ion: o.Ion, target: o.Target, detector: o.Detector, label: str) -> None:
    raise NotImplementedError
