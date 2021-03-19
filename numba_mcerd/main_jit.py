import copy
import logging

from numba_mcerd import config, timer
from numba_mcerd.mcerd import (
    random_jit, init_params, read_input, potential, ion_stack, init_simu, cross_section,
    potential_jit, init_simu_jit, cross_section_jit, elsto, init_detector, output, ion_simu_jit
)
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_jit as oj
import numba_mcerd.mcerd.objects_convert as oc


# These are too annoying to type
PRIMARY = c.IonType.PRIMARY.value
SECONDARY = c.IonType.SECONDARY.value
TARGET_ATOM = c.IonType.TARGET_ATOM.value


# This is less useful for numba-optimized functions (doesn't work, unlike print)
def setup_logging():
    """Setup logging for all modules"""
    level = logging.DEBUG
    filename = "numba_mcerd_jit.log"
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

    setup_logging()

    # Variables

    logging.debug("Initializing variables")

    g = o.Global()
    primary_ion = o.Ion()
    secondary_ion = o.Ion()
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
    read_input.read_input(g, primary_ion, secondary_ion, previous_trackpoint_ion, target, detector)

    # TODO: read_input needlessly seeds built-in random
    # TODO: Non-JIT modules may use random_vanilla, how to fix this without
    #       excessively copy-pasting JIT-versions of each module?
    random_jit.seed_rnd(g.seed)

    # Package ions into an array. Ions are (sometimes) accessed in the
    # original code like this
    if g.nions == 2:
        ions = [primary_ion, secondary_ion]
    elif g.nions == 3:
        ions = [primary_ion, secondary_ion, previous_trackpoint_ion]
    else:
        raise NotImplementedError

    logging.info("Initializing output files")
    init_params.init_io(g, primary_ion, target)

    # TODO: Rename to pot(ential)
    n, d, ux, uy = potential_jit.make_screening_table_cached()
    pot = oj.Potential(n, d)
    pot.ux = ux
    pot.uy = uy

    ion_stack.cascades_create_additional_ions(g, detector, target, [])

    logging.info(f"{g.nions} ions, {target.natoms} target atoms")

    # (g.jibal.gsto.extrapolate = True)

    table_timer = timer.SplitTimer.init_and_start()
    scat = []
    for i in range(g.nions):
        if g.simtype == c.SimType.SIM_RBS and i == c.IonType.TARGET_ATOM.value:
            continue
        ions[i].scatindex = i
        scat.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
        for j in range(target.natoms):
            init_simu_jit.scattering_table(g, ions[i], target, scat[i][j], pot, j)
            cross_section_jit.calc_cross_sections(g, scat[i][j], pot)

    table_timer.split()

    print(table_timer.start_time, table_timer.elapsed_laps)

    gsto_index = -1
    for j in range(target.nlayers):
        target.layer[j].sto = [o.Target_sto() for _ in range(g.nions)]
        for i in range(g.nions):
            gsto_index += 1
            if g.simtype == c.SimType.SIM_RBS and i == c.IonType.TARGET_ATOM.value:
                continue
            elsto.calc_stopping_and_straggling_const(g, ions[i], target, j, gsto_index)
            # TODO: Real gsto instead

    init_detector.init_detector(g, detector)
    detector = oc.convert_detector(detector)

    if g.predata:
        init_params.init_recoiling_angle(target)
    target = oc.convert_target(target)

    trackid = int(ions[SECONDARY].Z) * 1_000 + g.seed % 1_000
    trackid *= 1_000_000
    ions_moving.append(copy.deepcopy(ions[PRIMARY]))
    ions_moving.append(copy.deepcopy(ions[SECONDARY]))
    if g.simtype == c.SimType.SIM_RBS:
        ions_moving.append(copy.deepcopy(ions[TARGET_ATOM]))

    # TODO: Set default status for ions, and maybe something else

    logging.info("Starting simulation")

    g = oc.convert_global(g)

    for i in range(g.nsimu):
        g.cion = i  # TODO: Replace/remove for MT

        outer_loop_count = 0
        inner_loop_count = 0

        # output.output_data(g)  # Only prints status info

        cur_ion = ions_moving[PRIMARY]

        ion_simu_jit.create_ion(g, cur_ion, target)  # Doesn't work yet, wrong types

    # TODO
    pass


if __name__ == '__main__':
    # Path to data is hardcoded, change it to match your location
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", f"{config.PROJECT_ROOT}/data/input/Cl-Default"]
    main(main_args)
