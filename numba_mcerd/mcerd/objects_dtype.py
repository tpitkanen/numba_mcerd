import numpy as np

from numba_mcerd.mcerd import constants

# TODO: How to convert vanilla objects to these?
# TODO: Copy comments from original
# TODO: Are enums supported?

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


# Master


Global = np.dtype([
    ("mpi", np.bool),
    ("E0", np.float64),
    ("nions", np.int64),
    ("ncascades", np.int64),
    ("nsimu", np.int64),
    ("emin", np.float64),
    ("ionemax", np.float64),
    ("minangle", np.float64),
    ("seed", np.int64),
    ("cion", np.int64),
    ("simtype", np.int64),  # constants.SimType
    ("beamangle", np.float64),
    ("bspot", Point2),
    ("simstage", np.int64),  # constants.SimStage
    ("npresimu", np.int64),
    ("nscale", np.int64),
    ("nrecave", np.int64),
    ("cpresimu", np.int64),
    ("presimu", Presimu, 10000),  # TODO: Variable size
    ("predata", np.int64),
    # ( "master", Master),
    ("frecmin", np.float64),
    ("costhetamax", np.float64),
    ("costhetamin", np.float64),
    ("recwidth", np.int64),  # constants.RecWidth
    ("virtualdet", np.bool),
    ("basename", np.str, 100),  # TODO: size?
    ("finstat", np.int64, (2, 11)),  # len [SECONDARY + 1][NIONSTATUS]
    ("beamdiv", np.int64),
    ("beamprof", np.int64),  # constants.BeamProf
    ("rough", np.bool),
    ("nmclarge", np.int64),
    ("nmc", np.int64),
    ("output_trackpoints", np.bool),
    ("output_misses", np.bool),
    ("cascades", np.bool),
    ("advanced_output", np.bool),
    # ( "jibal", Jibal),
    ("nomc", np.bool)
])


Ion_opt = np.dtype([
    ("valid", np.bool),
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
    ("", np.float64, constants.MAXISOTOPES),
    ("", np.float64, constants.MAXISOTOPES),
    ("", np.float64),
    ("", np.int64),
    ("", np.float64)
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
    ("status", np.int64),  # constants.IonStatus
    ("opt", Ion_opt),
    ("w", np.float64),
    ("wtmp", np.float64),
    ("time", np.float64),
    ("tlayer", np.int64),
    ("lab", Vector),
    ("type", np.int64),  # constants.IonType
    ("hist", Rec_hist),
    ("dist", np.float64),
    ("virtual", np.bool),
    ("hit", Point, constants.MAXLAYERS),
    ("Ed", np.float64, constants.MAXLAYERS),
    ("dt", np.float64, 50),
    ("scale", np.bool),
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
    ("b", np.float, constants.EPSIMP)  # TODO: Correct size?
])


Potential = np.dtype([
    ("n", np.int64),
    ("d", np.int64),
    ("u", Point2, 30000)  # TODO: Correct size? Now using potential.py MAXPOINTS
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
    ("r", np.bool)
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
    ("sto", Target_sto, 3),  # TODO: Correct number for sto
    ("type", np.int64),  # constants.TargetType
    ("gas", np.bool),  # TODO: np.bool or np.bool_ ?
    ("stofile_prefix", np.str, constants.MAXSTOFILEPREFIXLEN)
])


Plane = np.dtype([
    ("a", np.float64),
    ("b", np.float64),
    ("c", np.float64),
    ("type", np.int64)
])


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
    ("table", np.bool)
])


Line = np.dtype([
    ("a", np.float64),
    ("b", np.float64),
    ("c", np.float64),
    ("d", np.float64),
    ("type", np.int64)
])


# Unused
# Rect

# Unused
# Circ


Det_foil = np.dtype([
    ("type", np.int64),
    ("virtual", np.bool),
    ("dist", np.float64),
    ("angle", np.float64),
    ("size", np.float64, 2),
    ("plane", Plane),
    ("center", Point)
])


Detector = np.dtype([
    ("type", np.int64),
    ("angle", np.float64),
    ("nfoils", np.int64),
    ("virtual", np.bool),
    ("vsize", np.float64, 2),
    ("tdet", np.int64, 2),
    ("edet", np.int64, constants.MAXLAYERS),
    ("thetamax", np.float64),
    ("vthetamax", np.float64),
    ("foil", Det_foil, constants.MAXFOILS),
    ("vfoil", Det_foil)
])


def main():
    # Usage examples

    # Values
    point = np.array((1., 2., 3.), dtype=Point)
    print(point)

    # Zeros
    vector = np.zeros((), dtype=Vector)
    print(vector)

    # Array
    points = np.zeros(5, dtype=Point)  # shape with 5 or (5,)
    print(points)

    # Object with arrays
    target_sto = np.zeros((), dtype=Target_sto)
    target_sto["vel"][0] = 1.
    print(target_sto)
    print(target_sto["vel"][0:5])  # Array slicing, not sure if supported in Numba

    target_layer = np.zeros((), dtype=Target_layer)
    target_layer["stofile_prefix"] = "moi"
    print(target_layer)

    # -------------------------------------------------------------------------
    # Object initialization test
    # -------------------------------------------------------------------------

    print("--------------------------------------------------------------------------------")

    # Arrays are too big to print without filling the output buffer

    target = np.zeros((), dtype=Target)
    # print(target)

    g = np.zeros((), dtype=Global)
    # print(g)

    ion = np.zeros((), dtype=Ion)
    # print(ion)

    scattering = np.zeros((), dtype=Scattering)
    # print(scattering)

    line = np.zeros((), dtype=Line)
    # print(line)

    detector = np.zeros((), dtype=Detector)
    # print(detector)


if __name__ == '__main__':
    main()
