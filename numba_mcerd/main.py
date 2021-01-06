import copy
import logging

from numba_mcerd import config, timer
from numba_mcerd.mcerd import random, init_params, read_input, potential, ion_stack, init_simu, cross_section, \
    potential_jit, init_simu_jit, cross_section_jit
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jitclass as oj


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

    # Time screening tables
    ptimer = timer.SplitTimer.init_and_start()

    pot = o.Potential()
    potential.make_screening_table(pot)
    ptimer.split()

    pot_oj = potential_jit.make_screening_table()
    ptimer.split()

    # Cached function
    pot_oj = potential_jit.make_screening_table()
    ptimer.stop()

    n, d, ux, uy = potential_jit.make_screening_table_cached()
    pot_oj2 = oj.Potential(n, d)
    pot_oj2.ux = ux
    pot_oj2.uy = uy

    ion_stack.cascades_create_additional_ions(g, detector, target, [])

    logging.info(f"{g.nions} ions, {target.natoms} target atoms")

    # (g.jibal.gsto.extrapolate = True)

    ions_jit = copy.deepcopy(ions)

    # Only one of these versions (jit or without) is needed
    table_timer = timer.SplitTimer.init_and_start()
    scat_jit = []
    for i in range(g.nions):
        if g.simtype == c.SimType.SIM_RBS and i == c.IonType.TARGET_ATOM.value:
            continue
        ions_jit[i].scatindex = i
        scat_jit.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
        for j in range(target.natoms):
            init_simu_jit.scattering_table(g, ions_jit[i], target, scat_jit[i][j], pot_oj, j)
            cross_section_jit.calc_cross_sections(g, scat_jit[i][j], pot_oj)

    table_timer.split()

    scat = []
    for i in range(g.nions):
        if g.simtype == c.SimType.SIM_RBS and i == c.IonType.TARGET_ATOM.value:
            continue
        ions[i].scatindex = i
        scat.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
        for j in range(target.natoms):
            # logging.debug(f"Calculating scattering between ions ...")
            init_simu.scattering_table(g, ions[i], target, scat[i][j], pot, j)
            cross_section.calc_cross_sections(g, scat[i][j], pot)

    table_timer.stop()

    print(table_timer.elapsed_laps)
    # Times (03f95b0559f2c0eea3d92eb34eb8a37306a39231)
    # [4.2860329, 28.060028000000003] (run)
    # [11.4422601, 158.6126873] (debug)

    # TODO
    pass


if __name__ == '__main__':
    # Path to data is hardcoded, change it to match your location
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", f"{config.PROJECT_ROOT}/data/input/Cl-Default"]
    main(main_args)
