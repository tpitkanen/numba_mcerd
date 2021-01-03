import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


def make_screening_table(potential: o.Potential) -> None:
    raise NotImplementedError


def get_max_x() -> float:
    raise NotImplementedError


def get_npoints(xmax: float) -> int:
    raise NotImplementedError


def U(x: float) -> float:
    raise NotImplementedError
