import math
from typing import Tuple

import numba as nb
import numpy as np


PI = 3.14159265358979323846


@nb.njit(cache=True, nogil=True)
def rotate(theta2, fii2, theta1, fii1) -> Tuple[float, float]:
    # From original:
    """Definition of the angles: theta2,fii2 is the angle of the
    second coordinate system in the first coordinate system.
    Theta1,fii1 is the specific direction in the second coordinate
    system, which direction in the first system is calculated
    in this routine (theta,fii).

    This routine cannot be explained easily. Read eg. Goldstein
    about the Euler angles and coordinate transforms.
    """
    cos_theta = math.cos(theta1)
    sin_theta = math.sin(theta1)
    cos_fii = math.cos(fii1)
    sin_fii = math.sin(fii1)

    x = sin_theta * cos_fii
    y = sin_theta * sin_fii
    z = cos_theta

    cosa1 = math.cos(theta2)
    sina1 = math.sin(theta2)

    cosa2 = math.cos(fii2 + PI / 2.0)
    sina2 = math.sin(fii2 + PI / 2.0)

    cosa3 = cosa2
    sina3 = -sina2

    rx = (x * (cosa3 * cosa2 - cosa1 * sina2 * sina3)
          + y * (-sina3 * cosa2 - cosa1 * sina2 * cosa3)
          + z * sina1 * sina2)

    ry = (x * (cosa3 * sina2 + cosa1 * cosa2 * sina3)
          + y * (-sina3 * sina2 + cosa1 * cosa2 * cosa3)
          - z * sina1 * cosa2)

    rz = (x * sina1 * sina3
          + y * sina1 * cosa3
          + z * cosa1)

    rz = max(min(rz, 1.0), -1.0)  # Clamp rz to [-1.0, 1.0]

    theta = math.acos(rz)
    if rx != 0.0:
        fii = math.atan2(ry, rx)
    else:
        fii = 0.0
    if math.fabs(fii) > 2.0 * PI:
        fii = np.fmod(fii, 2.0 * PI)  # math.fmod is not implemented in Numba
    if fii < 0.0:
        fii += 2.0 * PI

    return theta, fii
