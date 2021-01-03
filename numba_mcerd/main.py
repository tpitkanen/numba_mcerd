import logging

from numba_mcerd import config
from numba_mcerd.mcerd import random, init_params, read_input, potential, ion_stack, init_simu
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


def setup_logging():
    """Setup logging for all modules"""
    level = logging.DEBUG
    filename = "numba_mcerd.log"
    # Format example: "2020-12-18 15:34:26 DEBUG    [main.py:12] Initializing variables"
    logging_format = "{asctime} {levelname:8} [{filename}:{lineno}] {message}"
    style = "{"
    date_format = "%Y-%m-%d %H:%M:%S"

    # PyCharm doesn't recognize the 'style' keyword
    # noinspection PyArgumentList
    logging.basicConfig(filename=filename, level=level, style=style,
                        format=logging_format, datefmt=date_format)


def main(args):
    # Misc setup

    random.seed_rnd(1)
    setup_logging()

    # Variables

    logging.debug("Initializing variables")

    g = o.Global()
    ion = o.Ion()
    cur_ion = o.Ion()
    previous_trackpoint_ion = o.Ion()  # Not used unless simulation is RBS
    ions_moving = []  # TODO: Initialize a list of ions?
    target = o.Target()
    scat = None
    snext = o.SNext()
    detector = o.Detector()
    pot = o.Potential()

    i = None
    j = None
    nscat = None
    primary_finished = None
    trackid = None
    prev_layer = 0
    prev_layer_debug = 0
    ion_i = 0
    new_track = 1
    E_previous = 0.0
    E_difference = None

    # Preprocessing

    logging.debug("Starting preprocessing")
    logging.info("Initializing Jibal")
    g.jibal.initialize()
    logging.info("Initializing parameters")
    init_params.init_params(g, target, args)

    logging.info("Initializing input files")
    read_input.read_input(g, ion, cur_ion, previous_trackpoint_ion, target, detector)

    # Package ions into an array. Ions are (sometimes) accessed in the
    # original code like this
    if g.nions == 2:
        ions = [ion, cur_ion]
    elif g.nions == 3:
        ions = [ion, cur_ion, previous_trackpoint_ion]
    else:
        raise NotImplementedError

    logging.info("Initializing output files")
    init_params.init_io(g, ion, target)

    potential.make_screening_table(pot)

    ion_stack.cascades_create_additional_ions(g, detector, target, [])
    scat = [o.Scattering() for _ in range(g.nions)]

    logging.info(f"{g.nions} ions, {target.natoms} target atoms")

    # (g.jibal.gsto.extrapolate = True)

    scat = []
    for i in range(g.nions):
        if g.simtype == c.SimType.SIM_RBS and i == c.IonType.TARGET_ATOM.value:
            continue
        ions[i].scatindex = i
        scat.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
        for j in range(target.natoms):
            # logging.debug(f"Calculating scattering between ions ...")
            init_simu.scattering_table(g, ions[i], target, scat[i][j], pot, j)
            # cross_section.calc_cross_sections(g, scat[i][j], pot)

    # TODO
    pass


if __name__ == '__main__':
    # Path to data is hardcoded, change it to match your location
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", f"{config.PROJECT_ROOT}/data/input/Cl-Default"]
    main(main_args)
