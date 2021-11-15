from pathlib import Path
from typing import Tuple, List

import numba as nb
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_jit as oj
from numba_mcerd.mcerd import enums
from numba_mcerd.mcerd import rotate_jit

NPRESIMU = 500


class FitError(Exception):
    """Error while fitting to a curve"""


@nb.njit(cache=True)
def swap_presimus(presimu: np.ndarray, i: int, j: int) -> None:
    temp = presimu[i]["depth"], presimu[i]["angle"], presimu[i]["layer"]

    presimu[i]["depth"] = presimu[j]["depth"]
    presimu[i]["angle"] = presimu[j]["angle"]
    presimu[i]["layer"] = presimu[j]["layer"]

    presimu[j]["depth"], presimu[j]["angle"], presimu[j]["layer"] = temp


@nb.njit(cache=True)
def sort_presimu_by_depth(
        presimu: np.ndarray, start: int = None, stop: int = None) -> None:
    """Sort presimu by depth (ascending) in place.

    Uses insertion sort.
    """
    if start is None:
        start = 0
    if stop is None:
        stop = presimu.shape[0]

    for i in range(start + 1, stop):
        j = i
        while j > start and presimu[j - 1]["depth"] > presimu[j]["depth"]:
            swap_presimus(presimu, j, j - 1)
            j -= 1


@nb.njit(cache=True)
def sort_presimu_by_angle(
        presimu: np.ndarray, start: int = None, stop: int = None) -> None:
    """Sort presimu by angle (descending) in place.

    Uses insertion sort.
    """
    if start is None:
        start = 0
    if stop is None:
        stop = presimu.shape[0]

    for i in range(start + 1, stop):
        j = i
        while j > start and presimu[j - 1]["angle"] < presimu[j]["angle"]:
            swap_presimus(presimu, j, j - 1)
            j -= 1


@nb.njit(cache=True)
def finish_presimulation(g: oj.Global, detector: oj.Detector, recoil: oj.Ion) -> None:
    """In the presimulation stage we save the recoil angle relative to the
    detector when the recoil comes out of the target. Also the recoil
    depth and layer are saved for later use.
    """
    # Recoil direction in the laboratory coordinate system
    theta_lab, fii_lab = rotate_jit.rotate(recoil.lab.theta, recoil.lab.fii, recoil.theta, recoil.fii)

    # Recoil direction in the detector coordinate system
    theta, fii = rotate_jit.rotate(detector.angle, c.C_PI, theta_lab, fii_lab)

    g.presimu[g.cpresimu].depth = recoil.hist.tar_recoil.p.z
    g.presimu[g.cpresimu].angle = theta
    g.presimu[g.cpresimu].layer = recoil.hist.layer
    g.cpresimu += 1


# TODO: @nb.jit(forceobj=True) ?
def _output_to_files(g: oj.Global, master: oj.Master, target: oj.Target, detector: oj.Detector):
    """Output analyze_presimulation information to files.

    Call this inside a `with numba.objmode():` block.
    """
    # TODO: implement .out too
    pre_lines = []
    for i in range(target.ntarget):
        x_temp = target.recpar[i].x * c.C_NM / c.C_DEG
        y_temp = target.recpar[i].y / c.C_DEG
        # out_lines.append(f"{i:3d} {x_temp:10.5f} {y_temp:10.5f}\n")
        pre_lines.append(f"{x_temp:14.5e} {y_temp:14.5e}\n")

    pre_file = f"{g.basename}.pre"
    f = open(pre_file, "w")   # mode="w" overwrites
    f.writelines(pre_lines)
    f.close()


# TODO: I/O not supported in Numba
@nb.njit(cache=True)
def analyze_presimulation(g: oj.Global, master: oj.Master, target: oj.Target,
                          detector: oj.Detector) -> None:
    """We determine here the solid angle as function of recoiling depth
    and layer, which contains all but PRESIMU_LEVEL portion of the
    recoils (typically 99%). The result is given as a linear fit
    for half-angle vs. depth for each target layer.
    """
    depth = [[0.0] * NPRESIMU for _ in range(c.MAXLAYERS)]
    angle = [[0.0] * NPRESIMU for _ in range(c.MAXLAYERS)]
    nlayer = [0] * c.MAXLAYERS

    nlevel = int(10.0 / c.PRESIMU_LEVEL)
    nlevel = max(nlevel, int(g.cpresimu / (NPRESIMU + 1)))

    # Doesn't resize like in Vanilla
    sort_presimu_by_depth(g.presimu, start=0, stop=g.cpresimu)

    npre = 0
    while npre < g.cpresimu - int(nlevel / 2.0):
        layer = g.presimu[npre].layer
        n = 0
        sumdepth = 0.0

        while n < nlevel and n + npre < g.cpresimu and g.presimu[npre + n].layer == layer:
            sumdepth += g.presimu[npre + n].depth
            n += 1

        if n > int(nlevel / 2):
            sort_presimu_by_angle(g.presimu, npre, npre + n)
            i = int(n * c.PRESIMU_LEVEL)
            depth[layer][nlayer[layer]] = sumdepth / n
            angle[layer][nlayer[layer]] = g.presimu[npre + i].angle
            nlayer[layer] += 1

        npre += n

    # TODO: Output to Master_data (see objects_dtype)
    # out_lines = []
    a = b = 0.0
    for i in range(target.ntarget):
        for j in range(nlayer[i]):
            pass
            # C code uses format code 'i' instead of 'd', but Python doesn't
            # support 'i' for int anymore so 'd' is used instead
            # out_lines.append(f"{i:3d} {depth[i][j] / c.C_NM:10.4f} {angle[i][j] / c.C_DEG:10.4f}\n")

        # out_lines.append("\n")
        # out_lines.append(
        #     f"{i:3d} {a * c.C_NM / c.C_DEG:10.5f} {b / c.C_DEG:10.5f} + 1.1 * {detector.thetamax / c.C_DEG:6.3f} * {max(1.0, max(detector.vsize[0], detector.vsize[1])):6.2f}\n")
        if nlayer[i] > 2:
            a, b = fit_linear(depth[i], angle[i], nlayer[i])
            target.recpar[i].x = a
            target.recpar[i].y = b + 1.1 * detector.thetamax \
                                 * max(1.0, max(detector.vsize[0], detector.vsize[1]))
        else:
            if i > 0:
                target.recpar[i].x = target.recpar[i - 1].x
                target.recpar[i].y = target.recpar[i - 1].y
            else:
                target.recpar[i].x = 0.0
                target.recpar[i].y = \
                    5.0 * c.C_DEG + 1.1 * detector.thetamax \
                    * max(1.0, max(detector.vsize[0], detector.vsize[1]))

    # out_lines.append("\n")

    # pre_lines = []
    # for i in range(target.ntarget):
    #     x_temp = target.recpar[i].x * c.C_NM / c.C_DEG
    #     y_temp = target.recpar[i].y / c.C_DEG
    #     out_lines.append(f"{i:3d} {x_temp:10.5f} {y_temp:10.5f}\n")
    #     pre_lines.append(f"{x_temp:14.5e} {y_temp:14.5e}\n")

    # with Path(master.fpout).open("a") as f:
    #     f.writelines(out_lines)

    # # TODO: Move pre_file to master
    # pre_file = Path(f"{g.basename}.pre")
    # with pre_file.open("w") as f:  # mode="w" overwrites
    #     f.writelines(pre_lines)

    with nb.objmode():
        _output_to_files(g, master, target, detector)

    print("Presimulation finished")

    # del g.presimu  # Doesn't work with Numpy types
    g.simstage = enums.SimStage.REAL


@nb.njit(cache=True)
def fit_linear(x: List[float], y: List[float], n: int) -> Tuple[float, float]:
    """This is a weighted least-squares fit of a straight line according
    to P.R. Bevington's Data Reduction and Error Analysis for the
    Physical Sciences.

    Here we use constantly weight 1.0 since we have the same statistics
    for every point.
    """
    if n < 3:
        raise FitError("Too few points in the LS-fit in the pre-simulation analysis")

    xsum = ysum = xysum = x2sum = wsum = 0.0
    w = 1.0
    for i in range(n):
        xsum += x[i] * w
        ysum += y[i] * w
        xysum += x[i] * y[i] * w
        x2sum += x[i] * x[i] * w
        wsum += w

    Delta = 1.0 / (wsum * x2sum - xsum * xsum)
    a = Delta * (wsum * xysum - xsum * ysum)
    b = Delta * (x2sum * ysum - xsum * xysum)

    return a, b
