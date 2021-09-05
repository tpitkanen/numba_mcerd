import numba as nb

from numba_mcerd.mcerd import objects_jit as oj


# TODO: Check for types

"""Tools for copying dtypes or Jitclasses. 

numpy.copy() would be easier, but it doesn't work with custom dtypes in JIT.
"""


@nb.njit(cache=True)
def copy_point(point_a: oj.Point, point_b: oj.Point) -> None:
    """Copy values to point_a from point_b"""
    point_a.x = point_b.x
    point_a.y = point_b.y
    point_a.z = point_b.z


@nb.njit(cache=True)
def copy_vector(vector_a: oj.Vector, vector_b: oj.Vector) -> None:
    """Copy values to vector_a from vector_b"""
    copy_point(vector_a.p, vector_b.p)
    vector_a.theta = vector_b.theta
    vector_a.fii = vector_b.fii
    vector_a.v = vector_b.v
