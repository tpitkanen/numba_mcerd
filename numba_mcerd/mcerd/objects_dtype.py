import numpy as np

from numba_mcerd.mcerd import constants, enums

# TODO: Copy comments from original

LAYER_STO_COUNT_ERD = 2  # One sto for each beam ion in simulation (2 for ERD, 3 for RBS)
LAYER_STO_COUNT_RBS = 3


Point = np.dtype([
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float64)
])


Point2 = np.dtype([
    ("x", np.float64),
    ("y", np.float64)
])


Presimu = np.dtype([
    ("depth", np.float64),
    ("angle", np.float64),
    ("layer", np.int64)
])


# Jibal


# Use get_master_dtype in code, this is just for use as a type annotation
Master = np.dtype([
    # ("args", str, (2, 100)),
    ("fdata", str, 100),
    ("fpout", str, 100),
    ("fpdebug", str, 100),
    ("fpdat", str, 100),
    ("fperd", str, 100),
    ("fprange", str, 100),
    ("fptrack", str, 100)
    # Not needed:
    # ("argc", np.int64)
])


def get_master_dtype(g):
    return np.dtype([
        # FIXME: ValueError: invalid itemsize in generic type tuple
        # ("args", str, (len(g.master.args), len(max(g.master.args, key=len)))),
        ("fdata", str, len(str(g.master.fdata)) if g.master.fdata else 0),
        ("fpout", str, len(str(g.master.fpout)) if g.master.fpout else 0),
        ("fpdebug", str, len(str(g.master.fpdebug)) if g.master.fpdebug else 0),
        ("fpdat", str, len(str(g.master.fpdat)) if g.master.fpdat else 0),
        ("fperd", str, len(str(g.master.fperd)) if g.master.fperd else 0),
        ("fprange", str, len(str(g.master.fprange)) if g.master.fprange else 0),
        ("fptrack", str, len(str(g.master.fptrack)) if g.master.fptrack else 0)
        # Not needed:
        # argc
    ])


Global = np.dtype([
    ("mpi", bool),
    ("E0", np.float64),
    ("nions", np.int64),
    ("ncascades", np.int64),
    ("nsimu", np.int64),
    ("emin", np.float64),
    ("ionemax", np.float64),
    ("minangle", np.float64),
    ("seed", np.int64),
    ("cion", np.int64),
    ("simtype", np.int64),  # enums.SimType
    ("beamangle", np.float64),
    ("bspot", Point2),
    ("simstage", np.int64),  # enums.SimStage
    ("npresimu", np.int64),
    ("nscale", np.int64),
    ("nrecave", np.int64),
    ("cpresimu", np.int64),
    # ("presimu", Presimu, 10000),  # Real size is known only during runtime
    ("predata", np.int64),
    # ("master", Master),
    ("frecmin", np.float64),
    ("costhetamax", np.float64),
    ("costhetamin", np.float64),
    ("recwidth", np.int64),  # enums.RecWidth
    ("virtualdet", bool),
    ("basename", str, 100),  # TODO: size?
    ("finstat", np.int64, (enums.IonType.SECONDARY + 1, len(enums.IonStatus))),
    ("beamdiv", np.float64),
    ("beamprof", np.int64),  # enums.BeamProf
    ("rough", bool),
    ("nmclarge", np.int64),
    ("nmc", np.int64),
    ("output_trackpoints", bool),
    ("output_misses", bool),
    ("cascades", bool),
    ("advanced_output", bool),
    # ( "jibal", Jibal),
    ("nomc", bool)
])


Ion_opt = np.dtype([
    ("valid", bool),
    ("cos_theta", np.float64),
    ("sin_theta", np.float64),
    ("e", np.float64),
    ("y", np.float64)
])


Vector = np.dtype([
    ("p", Point),
    ("theta", np.float64),
    ("fii", np.float64),
    ("v", np.float64)
])


Rec_hist = np.dtype([
    ("tar_recoil", Vector),
    ("ion_recoil", Vector),
    ("lab_recoil", Vector),
    ("tar_primary", Vector),
    ("lab_primary", Vector),
    ("ion_E", np.float64),
    ("recoil_E", np.float64),
    ("layer", np.int64),
    ("nsct", np.int64),
    ("time", np.float64),
    ("w", np.float64),
    ("Z", np.float64),
    ("A", np.float64)
])


Isotopes = np.dtype([
    ("A", np.float64, constants.MAXISOTOPES),
    ("c", np.float64, constants.MAXISOTOPES),
    ("c_sum", np.float64),
    ("n", np.int64),
    ("Am", np.float64)
])


Ion = np.dtype([
    ("Z", np.float64),
    ("A", np.float64),
    ("E", np.float64),
    ("I", Isotopes),
    ("p", Point),
    ("theta", np.float64),
    ("fii", np.float64),
    ("nsct", np.int64),
    ("status", np.int64),  # enums.IonStatus
    ("opt", Ion_opt),
    ("w", np.float64),
    ("wtmp", np.float64),
    ("time", np.float64),
    ("tlayer", np.int64),
    ("lab", Vector),
    ("type", np.int64),  # enums.IonType
    ("hist", Rec_hist),
    ("dist", np.float64),
    ("virtual", bool),
    ("hit", Point, constants.MAXLAYERS),
    ("Ed", np.float64, constants.MAXLAYERS),
    ("dt", np.float64, constants.MAXLAYERS),
    ("scale", bool),
    ("effrecd", np.float64),
    ("trackid", np.int64),
    ("scatindex", np.int64),
    ("ion_i", np.int64),
    ("E_nucl_loss_det", np.float64)
])


Cross_section = np.dtype([
    ("emin", np.float64),
    ("emax", np.float64),
    ("estep", np.float64),
    ("b", np.float64, constants.EPSIMP)  # TODO: Correct size?
])


# Use get_potential_dtype in code, this is just for use as a type annotation
Potential = np.dtype([
    ("n", np.int64),
    ("d", np.int64),
    ("u", Point2, 30000)  # potential.py MAXPOINTS
])


def get_potential_dtype(u_size: int) -> Potential:
    return np.dtype([
        ("n", np.int64),
        ("d", np.int64),
        ("u", Point2, u_size)  # Variable size
    ])


Scattering = np.dtype([
    ("angle", np.float64, (constants.EPSNUM, constants.YNUM)),
    ("cross", Cross_section),
    ("logemin", np.float64),
    ("logymin", np.float64),
    ("logediv", np.float64),
    ("logydiv", np.float64),
    ("a", np.float64),
    ("E2eps", np.float64)
    # ("pot", Potential)  # Originally commented out
])


SNext = np.dtype([
    ("d", np.float64),
    ("natom", np.int64),
    ("r", bool)
])


Target_ele = np.dtype([
    ("Z", np.float64),
    ("A", np.float64)
])


Target_sto = np.dtype([
    ("vel", np.float64, constants.MAXSTO),
    ("sto", np.float64, constants.MAXSTO),
    ("stragg", np.float64, constants.MAXSTO),
    ("stodiv", np.float64),
    ("n_sto", np.int64)
])


Target_layer = np.dtype([
    ("natoms", np.int64),
    ("dlow", np.float64),
    ("dhigh", np.float64),
    ("atom", np.int64, constants.MAXATOMS),
    ("N", np.float64, constants.MAXATOMS),
    ("Ntot", np.float64),
    ("sto", Target_sto, LAYER_STO_COUNT_ERD),  # TODO: Get correct size like in get_global_dtype
    ("type", np.int64),  # enums.TargetType
    ("gas", bool),
    ("stofile_prefix", str, constants.MAXSTOFILEPREFIXLEN)
])


Plane = np.dtype([
    ("a", np.float64),
    ("b", np.float64),
    ("c", np.float64),
    ("type", np.int64)  # enums.PlaneType
])


# Use get_target_dtype in code, this is just for use as a type annotation
Target = np.dtype([
    ("minN", np.float64),
    ("ele", Target_ele, constants.MAXELEMENTS),
    ("layer", Target_layer, constants.MAXLAYERS),
    ("nlayers", np.int64),
    ("ntarget", np.int64),
    ("natoms", np.int64),
    ("recdist", Point2, constants.NRECDIST),
    ("nrecdist", np.int64),
    ("effrecd", np.float64),
    ("recmaxd", np.float64),
    ("plane", Plane),
    ("efin", np.float64, constants.NRECDIST),
    ("recpar", Point2, constants.MAXLAYERS),
    ("angave", np.float64),
    # ("surface", Surface),
    ("cross", np.float64, (constants.NSENE, constants.NSANGLE)),
    ("table", bool)
])


def get_target_dtype(target) -> np.dtype:
    layer_count = sum((1 for layer in target.layer if layer.type is not None))

    dtype = [
        ("minN", np.float64),
        ("ele", Target_ele, constants.MAXELEMENTS),
        ("layer", Target_layer, layer_count),
        ("nlayers", np.int64),
        ("ntarget", np.int64),
        ("natoms", np.int64),
        ("recdist", Point2, constants.NRECDIST),
        ("nrecdist", np.int64),
        ("effrecd", np.float64),
        ("recmaxd", np.float64),
        ("plane", Plane),
        ("efin", np.float64, constants.NRECDIST),
        ("recpar", Point2, layer_count),
        ("angave", np.float64),
        # ("surface", Surface),
        # TODO: replace dummy cross with a separate Target_cross or remove completely
        ("cross", np.float64, (1, 1)),
        ("table", bool)
    ]

    return np.dtype(dtype)


Line = np.dtype([
    ("a", np.float64),
    ("b", np.float64),
    ("c", np.float64),
    ("d", np.float64),
    ("type", np.int64)  # enums.LineType
])


# Unused
# Rect

# Unused
# Circ


Det_foil = np.dtype([
    ("type", np.int64),  # enums.FoilType
    ("virtual", bool),
    ("dist", np.float64),
    ("angle", np.float64),
    ("size_", np.float64, 2),
    ("plane", Plane),
    ("center", Point)
])


Detector = np.dtype([
    ("type", np.int64),  # enums.DetectorType
    ("angle", np.float64),
    ("nfoils", np.int64),
    ("virtual", bool),
    ("vsize", np.float64, 2),
    ("tdet", np.int64, 2),
    ("edet", np.int64, constants.MAXLAYERS),
    ("thetamax", np.float64),
    ("vthetamax", np.float64),
    ("foil", Det_foil, constants.MAXFOILS),
    ("vfoil", Det_foil)
])


# TODO: specific type
Buffer = np.ndarray


def get_buffer_dtype(length: int, row_width: int) -> np.dtype:
    dtype = [
        ("row_i", np.int64),
        ("col_i", np.int64),
        ("types", np.int64, (row_width,)),
        ("formats", "U5", (row_width,)),
        ("buf", np.float64, (length, row_width))
    ]

    return np.dtype(dtype)


def main():
    # Usage examples

    # Values
    point = np.zeros(1, dtype=Point)[0]
    point["x"], point["y"], point["z"] = 1., 2., 3.
    print(point)

    # Values (alternative)
    # https://numpy.org/devdocs/reference/arrays.scalars.html#indexing
    point_2 = np.array((1., 2., 3.), dtype=Point)[()]
    print(point_2)

    # Views
    point_view = point.view(np.recarray)
    point_view.x, point_view.y, point_view.z = 4., 5., 6.
    print(point)  # Views are shared with the original array
    print(point_view)

    # Zeros
    vector = np.zeros(1, dtype=Vector)[0]
    print(vector)

    # Array
    points = np.zeros(5, dtype=Point)  # shape with 5 or (5,)
    print(points)

    # Object with arrays
    target_sto = np.zeros(1, dtype=Target_sto)[0]
    target_sto["vel"][0] = 1.
    print(target_sto)
    print(target_sto["vel"][0:5])  # Array slicing, not sure if supported in Numba

    target_layer = np.zeros(1, dtype=Target_layer)[0]
    target_layer["stofile_prefix"] = "pref"
    print(target_layer)

    # -------------------------------------------------------------------------
    # Object initialization test
    # -------------------------------------------------------------------------

    print("--------------------------------------------------------------------------------")

    # These arrays are too big to print without filling the output buffer

    target = np.zeros(1, dtype=Target)[0]
    # print(target)

    g = np.zeros(1, dtype=Global)[0]
    # print(g)

    ion = np.zeros(1, dtype=Ion)[0]
    # print(ion)

    scattering = np.zeros(1, dtype=Scattering)[0]
    # print(scattering)

    line = np.zeros(1, dtype=Line)[0]
    # print(line)

    detector = np.zeros(1, dtype=Detector)[0]
    # print(detector)


if __name__ == '__main__':
    main()
