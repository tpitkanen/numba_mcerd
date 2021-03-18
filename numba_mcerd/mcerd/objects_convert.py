"""Utilities for converting normal Python objects to Numba jitclass objects"""
from enum import Enum
from typing import Any

import numpy as np

import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj


class ConvertError(Exception):
    """Error while converting"""


# TODO: Turn this into a wrapper
def _base_convert(obj, target_class, converter):
    values = vars(obj)
    converter(values)

    target_obj = target_class()
    setattr_all(target_obj, values)

    return target_obj


def _array_convert(array) -> np.ndarray:
    """Helper for converting arrays to np.ndarray.

    Needed because np.array(array) tries to do int32 arrays and such,
    but Numba requires 64 bit variables.
    """
    # Numpy can't infer types for empty arrays so they are ignored here
    if isinstance(array[0], int):
        return np.array(array, dtype=np.int64)
    if isinstance(array[0], float):
        return np.array(array, dtype=np.float64)
    if isinstance(array[0], bool):
        return np.array(array, dtype=np.bool)
    if isinstance(array, np.ndarray):
        return array
    raise ConvertError(f"Unsupported array type: '{type(array)}'")



def setattr_all(obj: Any, values: dict) -> None:
    """Set attributes for all values"""
    for key, val in values.items():
        setattr(obj, key, val)


def convert_point(point: o.Point) -> oj.Point:
    def convert(values):
        pass

    return _base_convert(point, oj.Point, convert)


def convert_point2(point: o.Point2) -> oj.Point2:
    def convert(values):
        pass

    return _base_convert(point, oj.Point2, convert)


def convert_presimu(presimu: o.Presimu) -> oj.Presimu:
    raise NotImplementedError


# TODO: Return Jibal and Master too
def convert_global(g: o.Global) -> oj.Global:
    raise NotImplementedError


def convert_ion_opt(ion_opt: o.Ion_opt) -> oj.Ion_opt:
    raise NotImplementedError


def convert_vector(vector: o.Vector) -> oj.Vector:
    raise NotImplementedError


def convert_rec_hist(rec_hist: o.Rec_hist) -> oj.Rec_hist:
    raise NotImplementedError


def convert_isotopes(isotopes: o.Isotopes) -> oj.Isotopes:
    raise NotImplementedError


def convert_ion(ion: o.Ion) -> oj.Ion:
    raise NotImplementedError


def convert_cross_section(cross_section: o.Cross_section) -> oj.Cross_section:
    raise NotImplementedError


# Not needed
# def convert_potential(pot: o.Potential) -> oj.Potential:
#     raise NotImplementedError


def convert_scattering(scat: o.Scattering) -> oj.Scattering:
    raise NotImplementedError


# TODO: correct type hints
def convert_scattering_nested(scat: Any) -> Any:
    raise NotImplementedError


def convert_snext(snext: o.SNext) -> oj.SNext:
    snext_jit = oj.SNext()
    values = vars(snext)

    setattr_all(snext_jit, values)
    return snext_jit


# def convert_surface(): pass  # Needed


def convert_target_ele(ele: o.Target_ele) -> oj.Target_ele:
    raise NotImplementedError


def convert_target_sto(sto: o.Target_sto) -> oj.Target_sto:
    raise NotImplementedError


def convert_target_layer(layer: o.Target_layer) -> oj.Target_layer:
    raise NotImplementedError


# Short way to define converter:
def convert_plane(plane: o.Plane) -> oj.Plane:
    def convert(values):
        values["type"] = values["type"].value

    return _base_convert(plane, oj.Plane, convert)


# Long way to define converter:
# def convert_plane(plane: o.Plane) -> oj.Plane:
#     plane_jit = oj.Plane()
#     values = vars(plane)
#
#     values["type"] = values["type"].value
#
#     setattr_all(plane_jit, values)
#     return plane_jit


def convert_target(target: o.Target) -> oj.Target:
    raise NotImplementedError


def convert_line(line: o.Line) -> oj.Line:
    def convert(values):
        values["type"] = values["type"].value

    return _base_convert(line, oj.Line, convert)

# def convert_rect(): pass
# def convert_circ(): pass


def convert_det_foil(foil: o.Det_foil) -> oj.Det_foil:
    def convert(values):
        values["type"] = values["type"].value
        values["size"] = _array_convert(values["size"])
        values["plane"] = convert_plane(values["plane"])
        values["center"] = convert_point(values["center"])

    return _base_convert(foil, oj.Det_foil, convert)


def convert_detector(detector: o.Detector) -> oj.Detector:
    def convert(values):
        values["type"] = values["type"].value
        values["vsize"] = _array_convert(values["vsize"])
        values["tdet"] = _array_convert(values["tdet"])
        values["edet"] = _array_convert(values["edet"])
        values["foil"] = [convert_det_foil(foil) for foil in values["foil"]
                          if foil.type is not None]
        values["vfoil"] = convert_det_foil(values["vfoil"])

    return _base_convert(detector, oj.Detector, convert)
