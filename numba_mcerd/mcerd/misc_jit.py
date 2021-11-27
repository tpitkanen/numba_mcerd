import copy
import math

import numba as nb
import numpy as np

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import rotate_jit, enums, copy_jit


@nb.njit(cache=True, nogil=True)
def coord_transform(porig: oj.Point, theta: float, fii: float, pin: oj.Point,
                    flag: enums.CoordTransformDirection) -> oj.Point:
    # TODO: Replace copy-paste description with own words
    """This routine will calculate the cartesian coordinates of point pin
    in another coordinate system. The origin of pin system in this
    other system is given by point porig. The rotation angles of pin
    system is given by theta and fii.
    Flag says which way the conversion is done.
    """
    pout = oj.Point()

    if flag == enums.CoordTransformDirection.BACK:
        pout.x = pin.x - porig.x
        pout.y = pin.y - porig.y
        pout.z = pin.z - porig.z
    else:
        pout.x = pin.x
        pout.y = pin.y
        pout.z = pin.z

    r = math.sqrt(pout.x**2 + pout.y**2 + pout.z**2)

    if r == 0.0:
        copy_jit.copy_point(pout, porig)
        return pout

    in_theta = math.acos(pout.z / r)
    in_fii = math.atan2(pout.y, pout.x)

    out_theta, out_fii = rotate_jit.rotate(theta, fii, in_theta, in_fii)

    pout.x = r * math.sin(out_theta) * math.cos(out_fii)
    pout.y = r * math.sin(out_theta) * math.sin(out_fii)
    pout.z = r * math.cos(out_theta)

    if flag == enums.CoordTransformDirection.FORW:
        pout.x += porig.x
        pout.y += porig.y
        pout.z += porig.z

    return pout
