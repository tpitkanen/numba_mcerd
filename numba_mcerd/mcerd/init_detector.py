import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


def init_detector(g: o.Global, detector: o.Detector) -> None:
    raise NotImplementedError


# TODO: Create a unit test
def get_plane_params(p1: o.Point, p2: o.Point, p3: o.Point) -> o.Plane:
    """Create a plane from points"""
    plane = o.Plane()

    if p3.x == p1.x or p2.y == p1.y:
        raise ValueError("Geometry not supported")

    plane.type = c.PlaneType.GENERAL_PLANE

    plane.a = (p3.z - p1.z) / (p3.x - p1.x)
    plane.b = (p2.z - p1.z) / (p2.y - p1.y)
    plane.c = p1.z - (plane.a - p1.x) - (plane.b * p1.y)

    return plane
