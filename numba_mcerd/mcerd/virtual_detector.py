import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


def hit_virtual_detector(g: o.Global, ion: o.Ion, target: o.Target, det: o.Detector,
                         p_virt_lab: o.Point, p_out_tar: o.Point) -> None:
    raise NotImplementedError


def get_eloss_corr(ion: o.Ion, target: o.Target, dE1: float, theta1: float, theta2: float) -> float:
    raise NotImplementedError


def get_eloss(E: float, ion: o.Ion, target: o.Target, cos_theta: float) -> float:
    raise NotImplementedError


def rbs_cross(m1: float, m2: float, theta: float, err: bool) -> float:
    raise NotImplementedError


def rbs_kin(m1: float, m2: float, theta: float, err: bool) -> float:
    raise NotImplementedError

