"""Utilities for converting normal Python objects to Numba jitclass objects.

Warning: expect conversions to modify/break original objects and for
converted objects to share their attributes with with originals.
"""
from typing import Any

import numba as nb
import numpy as np

import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj

# TODO: Replace reflected lists:
# https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types


class ConvertError(Exception):
    """Error while converting"""


def _base_convert(obj, target_class, converter):
    values = vars(obj)
    converter(values)

    target_obj = target_class()
    setattr_all(target_obj, values)

    return target_obj


def _convert_array(array) -> np.ndarray:
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
    """Set attributes for all values. Attributes with None value are skipped."""
    for key, val in values.items():
        if val is not None:
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
    def convert(values):
        pass

    return _base_convert(presimu, oj.Presimu, convert)


# TODO: Return Jibal and Master too
def convert_global(g: o.Global) -> oj.Global:
    def convert(values):
        values["simtype"] = values["simtype"].value
        values["bspot"] = convert_point2(values["bspot"])
        values["simstage"] = values["simstage"].value
        values["presimu"] = [convert_presimu(presimu) for presimu in values["presimu"]]
        values["master"] = None  # TODO: Implement
        values["recwidth"] = values["recwidth"].value
        values["finstat"] = np.array(values["finstat"], dtype=np.int64)
        values["beamprof"] = values["beamprof"].value
        values["jibal"] = None  # TODO: Implement

    return _base_convert(g, oj.Global, convert)


def convert_ion_opt(ion_opt: o.Ion_opt) -> oj.Ion_opt:
    def convert(values):
        pass

    return _base_convert(ion_opt, oj.Ion_opt, convert)


def convert_vector(vector: o.Vector) -> oj.Vector:
    def convert(values):
        values["p"] = convert_point(values["p"])

    return _base_convert(vector, oj.Vector, convert)


def convert_rec_hist(rec_hist: o.Rec_hist) -> oj.Rec_hist:
    def convert(values):
        values["tar_recoil"] = convert_vector(values["tar_recoil"])
        values["ion_recoil"] = convert_vector(values["ion_recoil"])
        values["lab_recoil"] = convert_vector(values["lab_recoil"])
        values["tar_primary"] = convert_vector(values["tar_primary"])
        values["lab_primary"] = convert_vector(values["lab_primary"])

    return _base_convert(rec_hist, oj.Rec_hist, convert)


def convert_isotopes(isotopes: o.Isotopes) -> oj.Isotopes:
    def convert(values):
        values["A"] = _convert_array(values["A"])
        values["c"] = _convert_array(values["c"])

    return _base_convert(isotopes, oj.Isotopes, convert)


def convert_ion(ion: o.Ion) -> oj.Ion:
    def convert(values):
        values["I"] = convert_isotopes(values["I"])
        values["p"] = convert_point(values["p"])
        values["status"] = values["status"].value
        values["opt"] = convert_ion_opt(values["opt"])
        values["lab"] = convert_vector(values["lab"])
        values["type"] = values["type"].value
        values["hist"] = convert_rec_hist(values["hist"])
        values["hit"] = [convert_point(point) for point in values["hit"]]
        values["Ed"] = _convert_array(values["Ed"])
        values["dt"] = _convert_array(values["dt"])

    return _base_convert(ion, oj.Ion, convert)


def convert_cross_section(cross_section: o.Cross_section) -> oj.Cross_section:
    def convert(values):
        values["b"] = np.array(values["b"], dtype=np.float64)

    return _base_convert(cross_section, oj.Cross_section, convert)


# Not needed
# def convert_potential(pot: o.Potential) -> oj.Potential:
#     raise NotImplementedError


def convert_scattering(scat: o.Scattering) -> oj.Scattering:
    def convert(values):
        values["angle"] = np.array(values["angle"], dtype=np.float64)
        values["cross"] = convert_cross_section(values["cross"])
        # values["pot"] =  # Originally commented out

    return _base_convert(scat, oj.Scattering, convert)


# TODO: correct type hints
def convert_scattering_nested(scat: Any) -> Any:
    scat_new = nb.typed.List()
    for i in range(len(scat)):
        # Numba can't infer types of nested empty lists, so fill first
        inner_list = nb.typed.List()
        for j in range(len(scat[i])):
            if not isinstance(scat[i][j].angle, np.ndarray):
                break  # Remove unused arrays
            inner_list.append(convert_scattering(scat[i][j]))
        scat_new.append(inner_list)
    return scat_new


def convert_snext(snext: o.SNext) -> oj.SNext:
    def convert(values):
        pass

    return _base_convert(snext, oj.SNext, convert)


# def convert_surface(): pass  # Needed


def convert_target_ele(ele: o.Target_ele) -> oj.Target_ele:
    def convert(values):
        pass

    return _base_convert(ele, oj.Target_ele, convert)


# Untested
def convert_target_sto(sto: o.Target_sto) -> oj.Target_sto:
    def convert(values):
        values["vel"] = _convert_array(values["vel"])
        values["sto"] = _convert_array(values["sto"])
        values["stragg"] = _convert_array(values["stragg"])

    return _base_convert(sto, oj.Target_sto, convert)


def convert_target_layer(layer: o.Target_layer) -> oj.Target_layer:
    def convert(values):
        values["atom"] = _convert_array(values["atom"])
        values["N"] = _convert_array(values["N"])
        values["sto"] = None  # TODO: Implement
        values["type"] = values["type"].value

    return _base_convert(layer, oj.Target_layer, convert)


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
    def convert(values):
        values["ele"] = [convert_target_ele(ele) for ele in values["ele"]]
        values["layer"] = [convert_target_layer(layer) for layer in values["layer"]
                           if layer.type is not None]  # Trim -> not the same as original
        values["recdist"] = [convert_point2(rec) for rec in values["recdist"]]
        values["plane"] = convert_plane(values["plane"])
        values["efin"] = _convert_array(values["efin"])
        values["recpar"] = [convert_point2(rec) for rec in values["recpar"]]
        values["surface"] = None  # TODO: Implement
        values["cross"] = np.array(values["cross"], dtype=np.float64)

    return _base_convert(target, oj.Target, convert)


def convert_line(line: o.Line) -> oj.Line:
    def convert(values):
        values["type"] = values["type"].value

    return _base_convert(line, oj.Line, convert)

# def convert_rect(): pass
# def convert_circ(): pass


def convert_det_foil(foil: o.Det_foil) -> oj.Det_foil:
    def convert(values):
        values["type"] = values["type"].value
        values["size"] = _convert_array(values["size"])
        values["plane"] = convert_plane(values["plane"])
        values["center"] = convert_point(values["center"])

    return _base_convert(foil, oj.Det_foil, convert)


def convert_detector(detector: o.Detector) -> oj.Detector:
    def convert(values):
        values["type"] = values["type"].value
        values["vsize"] = _convert_array(values["vsize"])
        values["tdet"] = _convert_array(values["tdet"])
        values["edet"] = _convert_array(values["edet"])
        # TODO: This trims empty values away -> not the same as original
        values["foil"] = [convert_det_foil(foil) for foil in values["foil"]
                          if foil.type is not None]
        values["vfoil"] = convert_det_foil(values["vfoil"])

    return _base_convert(detector, oj.Detector, convert)
