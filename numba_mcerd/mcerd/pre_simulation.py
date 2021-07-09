from pathlib import Path
from typing import Tuple, List

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
from numba_mcerd.mcerd import enums
from numba_mcerd.mcerd import rotate

NPRESIMU = 500


class FitError(Exception):
    """Error while fitting to a curve"""


def finish_presimulation(g: o.Global, detector: o.Detector, recoil: o.Ion) -> None:
    """In the presimulation stage we save the recoil angle relative to the
    detector when the recoil comes out of the target. Also the recoil
    depth and layer are saved for later use.
    """
    # Recoil direction in the laboratory coordinate system
    theta_lab, fii_lab = rotate.rotate(recoil.lab.theta, recoil.lab.fii, recoil.theta, recoil.fii)

    # Recoil direction in the detector coordinate system
    theta, fii = rotate.rotate(detector.angle, c.C_PI, theta_lab, fii_lab)

    g.presimu[g.cpresimu].depth = recoil.hist.tar_recoil.p.z
    g.presimu[g.cpresimu].angle = theta
    g.presimu[g.cpresimu].layer = recoil.hist.layer
    g.cpresimu += 1


def analyze_presimulation(g: o.Global, target: o.Target, detector: o.Detector) -> None:
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

    # This also resizes g.presimu
    g.presimu = sorted(g.presimu[:g.cpresimu], key=lambda presimu: presimu.depth)

    npre = 0
    while npre < g.cpresimu - int(nlevel / 2.0):
        layer = g.presimu[npre].layer
        n = 0
        sumdepth = 0.0

        while n < nlevel and n + npre < g.cpresimu and g.presimu[npre + n].layer == layer:
            sumdepth += g.presimu[npre + n].depth
            n += 1

        if n > int(nlevel / 2):
            g.presimu[npre:npre + n] = sorted(
                g.presimu[npre:npre + n], key=lambda presimu: presimu.angle, reverse=True)
            i = int(n * c.PRESIMU_LEVEL)
            depth[layer][nlayer[layer]] = sumdepth / n
            angle[layer][nlayer[layer]] = g.presimu[npre + i].angle
            nlayer[layer] += 1

        npre += n

    out_lines = []
    a = b = 0.0
    for i in range(target.ntarget):
        for j in range(nlayer[i]):
            # C code uses format code 'i' instead of 'd', but Python doesn't
            # support 'i' for int anymore so 'd' is used instead
            out_lines.append(f"{i:3d} {depth[i][j] / c.C_NM:10.4f} {angle[i][j] / c.C_DEG:10.4f}")

        out_lines.append("")
        out_lines.append(
            f"{i:3d} {a * c.C_NM / c.C_DEG:10.5f} {b / c.C_DEG:10.5f} + 1.1 * {detector.thetamax / c.C_DEG:6.3f} * {max(1.0, max(detector.vsize[0], detector.vsize[1])):6.2f}")
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

    out_lines.append("")

    pre_lines = []
    for i in range(target.ntarget):
        x_temp = target.recpar[i].x * c.C_NM / c.C_DEG
        y_temp = target.recpar[i].y / c.C_DEG
        out_lines.append(f"{i:3d} {x_temp:10.5f} {y_temp:10.5f}")
        pre_lines.append(f"{x_temp:14.5e} {y_temp:14.5e}")

    with g.master.fpout.open("a") as f:
        f.write("\n".join(out_lines))

    pre_file = Path(f"{g.basename}.pre")
    with pre_file.open("a") as f:
        f.write("\n".join(pre_lines))

    print("Presimulation finished")

    del g.presimu
    g.simstage = enums.SimStage.REALSIMULATION


def fit_linear(x: List[float], y: List[float], n: int) -> Tuple[float, float]:
    """This is a weighted least-squares fit of a straight line according
    to P.R. Bevington's Data Reduction and Error Analysis for the
    Physical Sciences.

    Here we use constantly weight 1.0 since were have same statistics
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
