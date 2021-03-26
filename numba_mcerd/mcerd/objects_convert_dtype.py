"""Utilities for converting normal Python objects to NumPy dtype objects.

Warning: expect conversions to modify/break original objects and for
converted objects to share their attributes with with originals.
"""
from typing import Any

import numba as nb
import numpy as np

import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_dtype as od
import numba_mcerd.mcerd.objects_convert_jit as ocj


class DtypeConvertError(Exception):
    """Error while converting"""


def setkey_all(obj: Any, values: dict) -> None:
    """Set dict for all key-values. Values with None value are skipped."""
    for key, val in values.items():
        if val is not None:
            obj[key] = val


def _base_convert(obj, target_dtype, converter):
    """Base conversion function for all dtypes

    Args:
        obj: Original object to convert
        target_dtype: Target dtype for the converted object
        converter: Conversion function for mapping attributes to converted types

    Returns:
        Converted object
    """
    # TODO: Non-destructive version:
    # values = copy.deepcopy(vars(obj))
    values = vars(obj)
    converter(values)

    target_obj = np.zeros((), dtype=target_dtype)
    setkey_all(target_obj, values)

    return target_obj


# TODO: Correct return types for instances of dtypes?

def convert_point(point: o.Point) -> od.Point:
    def convert(values):
        pass

    return _base_convert(point, od.Point, convert)


def convert_point2(point: o.Point2) -> od.Point2:
    pass


def convert_presimu(presimu: o.Presimu) -> od.Presimu:
    pass


# TODO: Return Jibal and Master too
def convert_global(g: o.Global) -> od.Global:
    pass


def convert_ion_opt(ion_opt: o.Ion_opt) -> od.Ion_opt:
    pass


def convert_vector(vector: o.Vector) -> od.Vector:
    def convert(values):
        values["p"] = convert_point(values["p"])

    return _base_convert(vector, od.Vector, convert)


def convert_rec_hist(rec_hist: o.Rec_hist) -> od.Rec_hist:
    pass


def convert_isotopes(isotopes: o.Isotopes) -> od.Isotopes:
    pass


def convert_ion(ion: o.Ion) -> od.Ion:
    pass


def convert_cross_section(cross_section: o.Cross_section) -> od.Cross_section:
    pass


# Not needed
# def convert_potential(pot: o.Potential) -> od.Potential:
#     raise NotImplementedError


def convert_scattering(scat: o.Scattering) -> od.Scattering:
    pass


# TODO: correct type hints
def convert_scattering_nested(scat: Any) -> Any:
    pass


def convert_snext(snext: o.SNext) -> od.SNext:
    pass


# def convert_surface(): pass  # Needed


def convert_target_ele(ele: o.Target_ele) -> od.Target_ele:
    pass


def convert_target_sto(sto: o.Target_sto) -> od.Target_sto:
    pass


def convert_target_layer(layer: o.Target_layer) -> od.Target_layer:
    pass


def convert_plane(plane: o.Plane) -> od.Plane:
    pass


def convert_target(target: o.Target) -> od.Target:
    pass


def convert_line(line: o.Line) -> od.Line:
    pass


def convert_det_foil(foil: o.Det_foil) -> od.Det_foil:
    pass


def convert_detector(detector: o.Detector) -> od.Detector:
    pass


def main():
    point = o.Point(1., 2., 3.)
    conv_point = convert_point(point)
    print(point)
    print(conv_point)
    print()

    vector = o.Vector(p=o.Point(2., 4., 6.), theta=1.0, fii=0.5, v=0.2)
    conv_vector = convert_vector(vector)
    print(vector)
    print(conv_vector)


if __name__ == "__main__":
    main()
