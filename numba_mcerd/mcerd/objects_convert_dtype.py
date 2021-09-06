"""Utilities for converting normal Python objects to NumPy dtype objects.

Warning: expect conversions to modify/break original objects and for
converted objects to share their attributes with originals.
"""
from typing import Any, Callable

import numpy as np

import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_dtype as od


class DtypeConvertError(Exception):
    """Error while converting"""


def setkey_all(obj: Any, values: dict) -> None:
    """Set dict for all key-values. Values with None value are skipped."""
    for key, val in values.items():
        if val is not None:
            obj[key] = val


def _base_convert(obj: Any, target_dtype: np.dtype, converter: Callable):
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

    target_obj = np.zeros(1, dtype=target_dtype)[0]
    setkey_all(target_obj, values)

    # TODO: Is this the only place that needs .view(np.recarray)?
    # TODO: Should some things be np.record instead?
    # TODO: Does this affect performance?
    return target_obj.view(np.recarray)


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
        return np.array(array, dtype=bool)
    if isinstance(array, np.ndarray):
        return array
    raise DtypeConvertError(f"Unsupported array type: '{type(array)}'")


# TODO: Correct return types for instances of dtypes?

def convert_point(point: o.Point) -> od.Point:
    def convert(values):
        pass

    return _base_convert(point, od.Point, convert)


def convert_point2(point: o.Point2) -> od.Point2:
    def convert(values):
        pass

    return _base_convert(point, od.Point2, convert)


def convert_presimu(presimu: o.Presimu) -> od.Presimu:
    def convert(values):
        pass

    return _base_convert(presimu, od.Presimu, convert)


def convert_master(g: o.Global) -> od.Master:
    def convert(values):
        values["args"] = None  # TODO: Implement if needed
        values["fdata"] = str(g.master.fdata) if g.master.fdata else ""
        values["fpout"] = str(g.master.fpout) if g.master.fpout else ""
        values["fpdebug"] = str(g.master.fpdebug) if g.master.fpdebug else ""
        values["fpdat"] = str(g.master.fpdat) if g.master.fpdat else ""
        values["fperd"] = str(g.master.fperd) if g.master.fperd else ""
        values["fprange"] = str(g.master.fprange) if g.master.fprange else ""
        values["fptrack"] = str(g.master.fptrack) if g.master.fptrack else ""

    master_dtype = od.get_master_dtype(g)
    return _base_convert(g.master, master_dtype, convert)


# TODO: Return Jibal if needed
def convert_global(g: o.Global) -> od.Global:
    def convert(values):
        values["simtype"] = values["simtype"].value
        values["bspot"] = convert_point2(values["bspot"])
        values["simstage"] = values["simstage"].value
        values["presimu"] = np.array([convert_presimu(presimu) for presimu in values["presimu"]])
        values["master"] = None  # Implemented in a separate function
        values["recwidth"] = values["recwidth"].value
        values["finstat"] = np.array(values["finstat"], dtype=np.int64)
        values["beamprof"] = values["beamprof"].value
        values["jibal"] = None  # TODO: Implement

    global_dtype = od.get_global_dtype(len(g.presimu))
    return _base_convert(g, global_dtype, convert)


def convert_ion_opt(ion_opt: o.Ion_opt) -> od.Ion_opt:
    def convert(values):
        pass

    return _base_convert(ion_opt, od.Ion_opt, convert)


def convert_vector(vector: o.Vector) -> od.Vector:
    def convert(values):
        values["p"] = convert_point(values["p"])

    return _base_convert(vector, od.Vector, convert)


def convert_rec_hist(rec_hist: o.Rec_hist) -> od.Rec_hist:
    def convert(values):
        values["tar_recoil"] = convert_vector(values["tar_recoil"])
        values["ion_recoil"] = convert_vector(values["ion_recoil"])
        values["lab_recoil"] = convert_vector(values["lab_recoil"])
        values["tar_primary"] = convert_vector(values["tar_primary"])
        values["lab_primary"] = convert_vector(values["lab_primary"])

    return _base_convert(rec_hist, od.Rec_hist, convert)


def convert_isotopes(isotopes: o.Isotopes) -> od.Isotopes:
    def convert(values):
        values["A"] = _convert_array(values["A"])
        values["c"] = _convert_array(values["c"])

    return _base_convert(isotopes, od.Isotopes, convert)


def convert_ion(ion: o.Ion) -> od.Ion:
    def convert(values):
        values["I"] = convert_isotopes(values["I"])
        values["p"] = convert_point(values["p"])
        values["status"] = values["status"].value
        values["opt"] = convert_ion_opt(values["opt"])
        values["lab"] = convert_vector(values["lab"])
        values["type"] = values["type"].value
        values["hist"] = convert_rec_hist(values["hist"])
        values["hit"] = np.array([convert_point(point) for point in values["hit"]])
        values["Ed"] = _convert_array(values["Ed"])
        values["dt"] = _convert_array(values["dt"])

    return _base_convert(ion, od.Ion, convert)


def convert_cross_section(cross_section: o.Cross_section) -> od.Cross_section:
    def convert(values):
        values["b"] = np.array(values["b"], dtype=np.float64)

    return _base_convert(cross_section, od.Cross_section, convert)


# Not needed
# def convert_potential(pot: o.Potential) -> od.Potential:
#     raise NotImplementedError


def convert_scattering(scat: o.Scattering) -> od.Scattering:
    def convert(values):
        values["angle"] = np.array(values["angle"], dtype=np.float64)
        values["cross"] = convert_cross_section(values["cross"])
        # values["pot"] =  # Originally commented out

    return _base_convert(scat, od.Scattering, convert)


# TODO: correct type hints
def convert_scattering_nested(scat: Any) -> Any:
    # type(np.array(scat)[i][j]) == o.Scattering
    scat_new = np.array(
        [[convert_scattering(scat) for scat in inner_list]
         for inner_list in scat])

    return scat_new.view(np.recarray)


def convert_snext(snext: o.SNext) -> od.SNext:
    def convert(values):
        pass

    return _base_convert(snext, od.SNext, convert)


# def convert_surface(): pass  # Not needed


def convert_target_ele(ele: o.Target_ele) -> od.Target_ele:
    def convert(values):
        pass

    return _base_convert(ele, od.Target_ele, convert)


def convert_target_sto(sto: o.Target_sto) -> od.Target_sto:
    def convert(values):
        values["vel"] = _convert_array(values["vel"])
        values["sto"] = _convert_array(values["sto"])
        values["stragg"] = _convert_array(values["stragg"])

    return _base_convert(sto, od.Target_sto, convert)


def convert_target_layer(layer: o.Target_layer) -> od.Target_layer:
    def convert(values):
        values["atom"] = _convert_array(values["atom"])
        values["N"] = _convert_array(values["N"])
        values["sto"] = np.array([convert_target_sto(sto) for sto in values["sto"]])  # dtype needed?
        values["type"] = values["type"].value

    return _base_convert(layer, od.Target_layer, convert)


def convert_plane(plane: o.Plane) -> od.Plane:
    def convert(values):
        values["type"] = values["type"].value

    return _base_convert(plane, od.Plane, convert)


def convert_target(target: o.Target) -> od.Target:
    def convert(values):
        values["ele"] = np.array([convert_target_ele(ele) for ele in values["ele"]])
        values["layer"] = np.array(
            [convert_target_layer(layer) if layer.type is not None else np.zeros(1, dtype=od.Target_layer)[0]
             for layer in values["layer"]])

        values["recdist"] = np.array([convert_point2(rec) for rec in values["recdist"]])
        values["plane"] = convert_plane(values["plane"])
        values["efin"] = _convert_array(values["efin"])
        values["recpar"] = np.array([convert_point2(rec) for rec in values["recpar"]])
        values["surface"] = None  # TODO: Implement
        values["cross"] = np.array(values["cross"], dtype=np.float64)

    return _base_convert(target, od.Target, convert)


def convert_line(line: o.Line) -> od.Line:
    def convert(values):
        values["type"] = values["type"].value

    return _base_convert(line, od.Line, convert)


def convert_det_foil(foil: o.Det_foil) -> od.Det_foil:
    def convert(values):
        values["type"] = values["type"].value
        values["size"] = _convert_array(values["size"])
        values["plane"] = convert_plane(values["plane"])
        values["center"] = convert_point(values["center"])

    return _base_convert(foil, od.Det_foil, convert)


def convert_detector(detector: o.Detector) -> od.Detector:
    def convert(values):
        values["type"] = values["type"].value
        values["vsize"] = _convert_array(values["vsize"])
        values["tdet"] = _convert_array(values["tdet"])
        values["edet"] = _convert_array(values["edet"])
        values["foil"] = np.array(
            [convert_det_foil(foil) if foil.type is not None else np.zeros(1, dtype=od.Det_foil)[0]
             for foil in values["foil"]])
        values["vfoil"] = convert_det_foil(values["vfoil"])

    return _base_convert(detector, od.Detector, convert)


def main():
    # TODO: copy.deepcopy() for test objects
    # import copy

    from numba_mcerd.mcerd import constants, enums

    point = o.Point(1., 2., 3.)
    conv_point = convert_point(point)
    print(point)
    print(conv_point)
    print()

    g = o.Global(simtype=enums.SimType.ERD, simstage=enums.SimStage.ANY,
                 recwidth=enums.RecWidth.WIDE, beamprof=enums.BeamProf.NONE)
    g.presimu = [o.Presimu() for _ in range(50)]
    conv_g = convert_global(g)
    print(g)
    print(conv_g)

    vector = o.Vector(p=o.Point(2., 4., 6.), theta=1.0, fii=0.5, v=0.2)
    conv_vector = convert_vector(vector)
    print(vector)
    print(conv_vector)

    rec_hist = o.Rec_hist()
    conv_rec_hist = convert_rec_hist(rec_hist)
    print(rec_hist)
    print(conv_rec_hist)

    ion = o.Ion(status=enums.IonStatus.NOT_FINISHED, type=enums.IonType.PRIMARY)
    conv_ion = convert_ion(ion)
    print(ion)
    print(conv_ion)

    target_sto = o.Target_sto()
    conv_target_sto = convert_target_sto(target_sto)
    print(target_sto)
    print(conv_target_sto)

    scat = o.Scattering()
    scat.cross.b = [0.0] * constants.EPSIMP
    conv_scat = convert_scattering(scat)
    print(scat)
    print(conv_scat)

    snext = o.SNext()
    conv_snext = convert_snext(snext)
    print(snext)
    print(conv_snext)

    target = o.Target()
    target.plane.type = enums.PlaneType.GENERAL_PLANE
    target.layer[0].sto = [o.Target_sto() for _ in range(2)]
    target.layer[0].type = enums.TargetType.FILM
    target_conv = convert_target(target)
    print(target)
    print(target_conv)

    line = o.Line(type=enums.LineType.GENERAL)
    conv_line = convert_line(line)
    print(line)
    print(conv_line)

    det = o.Detector(type=enums.DetectorType.TOF)
    det.foil[0].plane.type = enums.PlaneType.GENERAL_PLANE
    det.vfoil.plane.type = enums.PlaneType.GENERAL_PLANE
    det.vfoil.type = enums.FoilType.CIRC
    conv_det = convert_detector(det)
    print(det)
    print(conv_det)


if __name__ == "__main__":
    main()
