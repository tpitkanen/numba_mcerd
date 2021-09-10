import copy
import logging

import numpy as np

from numba_mcerd import config, timer, patch_numba
from numba_mcerd.mcerd import (
    cross_section_jit,
    elsto,
    enums,
    erd_detector_jit,
    erd_scattering_jit,
    finalize_jit,
    finish_ion_jit,
    init_detector,
    init_params,
    init_params_jit,
    init_simu_jit,
    ion_simu_jit,
    ion_stack,
    output_jit,
    potential_jit,
    pre_simulation_jit,
    print_data_jit,
    random_jit,
    read_input
)
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_convert_jit as ocj
import numba_mcerd.mcerd.objects_convert_dtype as ocd


# These are too annoying to type
PRIMARY = enums.IonType.PRIMARY.value
SECONDARY = enums.IonType.SECONDARY.value
TARGET_ATOM = enums.IonType.TARGET_ATOM.value


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
    patch_numba.patch_nested_array()

    # Variables

    logging.debug("Initializing variables")

    initialization_timer = timer.SplitTimer.init_and_start()

    # _o stands for original (not converted to Numpy dtype array/record)
    g_o = o.Global()
    primary_ion_o = o.Ion()  # Not really used in the simulation loop
    secondary_ion_o = o.Ion()  # Not really used in the simulation loop
    previous_trackpoint_ion_o = o.Ion()  # Not used unless simulation is RBS
    ions_moving_o = []
    target_o = o.Target()
    scat_o = None
    snext_o = o.SNext()
    detector_o = o.Detector()

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
    g_o.jibal.initialize()
    logging.info("Initializing parameters")
    init_params.init_params(g_o, target_o, args)

    logging.info("Initializing input files")
    read_input.read_input(g_o, primary_ion_o, secondary_ion_o, previous_trackpoint_ion_o, target_o, detector_o)

    # TODO: read_input needlessly seeds built-in random
    # TODO: Non-JIT modules may use random_vanilla, how to fix this without
    #       excessively copy-pasting JIT-versions of each module?
    random_jit.seed_rnd(g_o.seed)

    # Package ions into an array. Ions are (sometimes) accessed in the
    # original code like this
    if g_o.nions == 2:
        ions = [primary_ion_o, secondary_ion_o]
    elif g_o.nions == 3:
        ions = [primary_ion_o, secondary_ion_o, previous_trackpoint_ion_o]
    else:
        raise NotImplementedError

    logging.info("Initializing output files")
    init_params.init_io(g_o, primary_ion_o, target_o)

    # TODO: Rename to pot(ential)
    # Vanilla Python:
    # pot = potential_jit.make_screening_table_dtype.py_func()  # type(pot) == <class 'numpy.void'>
    # pot = pot.view(np.recarray)  # type(pot) == <class 'numpy.record'>

    # Converting ndarray objects to views makes them easier to debug
    # outside of Numba. They also seem to be slightly faster in Numba.
    pot = potential_jit.make_screening_table_dtype().view(np.recarray)

    ion_stack.cascades_create_additional_ions(g_o, detector_o, target_o, [])

    logging.info(f"{g_o.nions} ions, {target_o.natoms} target atoms")

    # (g.jibal.gsto.extrapolate = True)

    table_timer = timer.SplitTimer.init_and_start()
    scat_o = []
    for i in range(g_o.nions):
        if g_o.simtype == enums.SimType.RBS and i == enums.IonType.TARGET_ATOM.value:
            continue
        ions[i].scatindex = i
        scat_o.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
        for j in range(target_o.natoms):
            init_simu_jit.scattering_table(g_o, ions[i], target_o, scat_o[i][j], pot, j)
            cross_section_jit.calc_cross_sections(g_o, scat_o[i][j], pot)

    table_timer.stop()
    print(f"table_timer: {table_timer}")

    gsto_index = -1
    for j in range(target_o.nlayers):
        target_o.layer[j].sto = [o.Target_sto() for _ in range(g_o.nions)]
        for i in range(g_o.nions):
            gsto_index += 1
            if g_o.simtype == enums.SimType.RBS and i == enums.IonType.TARGET_ATOM.value:
                continue
            elsto.calc_stopping_and_straggling_const(g_o, ions[i], target_o, j, gsto_index)
            # TODO: Real gsto instead

    init_detector.init_detector(g_o, detector_o)

    print_data_jit.print_data(g_o)

    if g_o.predata:
        init_params.init_recoiling_angle(target_o)

    trackid = int(ions[SECONDARY].Z) * 1_000 + g_o.seed % 1_000
    trackid *= 1_000_000
    ions_moving_o.append(copy.deepcopy(ions[PRIMARY]))
    ions_moving_o.append(copy.deepcopy(ions[SECONDARY]))
    if g_o.simtype == enums.SimType.RBS:
        ions_moving_o.append(copy.deepcopy(ions[TARGET_ATOM]))

    logging.info("Converting objects to JIT")

    initialization_timer.stop()
    print(f"initialization_timer: {initialization_timer}")

    for ion in ions_moving_o:
        ion.status = enums.IonStatus.NOT_FINISHED

    # dtype conversions
    dtype_conversion_timer = timer.SplitTimer.init_and_start()

    g = ocd.convert_global(copy.deepcopy(g_o))
    master = ocd.convert_master(copy.deepcopy(g_o))
    ions_moving = np.array([ocd.convert_ion(copy.deepcopy(ion)) for ion in ions_moving_o])
    target = ocd.convert_target(copy.deepcopy(target_o))
    scat = ocd.convert_scattering_nested(copy.deepcopy(scat_o))
    snext = ocd.convert_snext(copy.deepcopy(snext_o))
    detector = ocd.convert_detector(copy.deepcopy(detector_o))

    dtype_conversion_timer.stop()
    print(f"dtype_conversion_timer: {dtype_conversion_timer}")

    # Jitclass conversions
    # jitclass_conversion_timer = timer.SplitTimer.init_and_start()
    #
    # g = ocj.convert_global(g_o)
    # ions_moving = [ocj.convert_ion(ion_o) for ion in ions_moving]
    # target = ocj.convert_target(target_o)
    # scat = ocj.convert_scattering_nested(scat_o)
    # snext = ocj.convert_snext(snext_o)
    # detector = ocj.convert_detector(detector_o)
    #
    # jitclass_conversion_timer.stop()
    # print(f"jitclass_conversion_timer: {jitclass_conversion_timer}")

    outer_loop_counts = np.zeros(shape=g.nsimu, dtype=np.int64)
    inner_loop_counts = np.zeros(shape=g.nsimu, dtype=np.int64)

    logging.info("Starting simulation")

    presim_timer = timer.SplitTimer.init_and_start()
    # TODO: Move this to a separate jit-compiled function to eliminate
    #       context-switching overhead
    for i in range(g.nsimu):
        print(i)

        g.cion = i  # TODO: Replace/remove for MT

        outer_loop_count = 0
        inner_loop_count = 0

        # output.output_data(g)  # Only prints status info

        cur_ion = ions_moving[PRIMARY]

        ion_simu_jit.create_ion(g, cur_ion, target)
        if g.rough:
            ion_simu_jit.move_target(target)

        primary_finished = False
        while not primary_finished:
            outer_loop_count += 1

            ion_simu_jit.next_scattering(g, cur_ion, target, scat, snext)
            nscat = ion_simu_jit.move_ion(g, cur_ion, target, snext)

            if nscat == enums.ScatteringType.ERD_SCATTERING:
                if erd_scattering_jit.erd_scattering(
                        g, ions_moving[PRIMARY], ions_moving[SECONDARY], target, detector):
                    cur_ion = ions_moving[SECONDARY]

            if cur_ion.status == enums.IonStatus.FIN_RECOIL or cur_ion.status == enums.IonStatus.FIN_OUT_DET:
                if g.simstage == enums.SimStage.PRE:
                    pre_simulation_jit.finish_presimulation(g, detector, cur_ion)
                    cur_ion = ions_moving[PRIMARY]
                else:
                    # TODO: ZeroDivisionError sometimes in JIT
                    erd_detector_jit.move_to_erd_detector(g, cur_ion, target, detector)

            # TODO: Separate loop to pre and main, move this in-between
            if g.simstage == enums.SimStage.PRE and g.cion == g.npresimu - 1:
                pre_simulation_jit.analyze_presimulation(g, master, target, detector)
                init_params_jit.init_recoiling_angle(target)

                presim_timer.stop()
                print(f"presim_timer: {presim_timer}")

                main_sim_timer = timer.SplitTimer.init_and_start()

            if (nscat == enums.ScatteringType.MC_SCATTERING
                    and cur_ion.status == enums.IonStatus.NOT_FINISHED
                    and not g.nomc):
                if ion_simu_jit.mc_scattering(
                        g, cur_ion, ions_moving[SECONDARY], target, detector, scat, snext):  # ion_stack.next_ion()
                    # This block is never reached in ERD mode
                    cur_ion = ions_moving[SECONDARY]  # ion_stack.next_ion()
                    found = False
                    for j in range(g.nions):
                        if j == TARGET_ATOM and g.simtype == enums.SimType.RBS:
                            continue
                        if (round(ions[j].Z) == round(cur_ion.Z)
                                and round(ions[j].A / c.C_U) == round(ions[j].A / c.C_U)):
                            # FIXME: Comparing average mass by rounding is a bad idea.
                            #        See the original code for more information.
                            found = True
                            cur_ion.scatindex = j
                    if not found:
                        logging.warning(
                            f"Recoil cascade not possible, since recoiling ion Z={cur_ion.Z} and A={cur_ion.A / c.C_U} u are not in ion table (and therefore not in scattering table or stopping/straggling tables)")
                        raise NotImplementedError
                        # cur_ion = ion_stack.prev_ion()
                    else:
                        ion_i += 1
                        cur_ion.ion_i = ion_i
                        cur_ion.trackid = trackid

                    # logging.debug(...)

            # debug: loop over layers, print cur_ion.tlayer and set prev_layer_debug

            if g.output_trackpoints:
                raise NotImplementedError

            if cur_ion.type == SECONDARY and cur_ion.status != enums.IonStatus.NOT_FINISHED:
                g.finstat[SECONDARY][cur_ion.status] += 1

            while ion_simu_jit.ion_finished(g, cur_ion, target):
                inner_loop_count += 1

                # logging.debug(...)

                if g.output_trackpoints:
                    raise NotImplementedError

                cur_ion.trackid = trackid if not new_track else 0
                # No new track is made if ion doesn't make it to the
                # energy detector or if it's a scaling ion

                if cur_ion.type <= SECONDARY:
                    output_jit.output_erd(g, master, cur_ion, target, detector)
                if cur_ion.type == PRIMARY:
                    primary_finished = True
                    break
                cur_ion = ions_moving[PRIMARY]  # ion_stack.prev_ion()
                if cur_ion.type != PRIMARY and g.output_trackpoints:
                    raise NotImplementedError

        outer_loop_counts[g.cion] = outer_loop_count
        inner_loop_counts[g.cion] = inner_loop_count

        # logging.debug(...)

        g.finstat[PRIMARY][cur_ion.status] += 1
        finish_ion_jit.finish_ion(g, cur_ion)  # Print info if FIN_STOP or FIN_TRANS

    finalize_jit.finalize(g, master)  # Print statistics

    # noinspection PyUnboundLocalVariable
    main_sim_timer.stop()
    print(f"main_sim_timer: {main_sim_timer}")


if __name__ == '__main__':
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", f"{config.PROJECT_ROOT}/data/input/Cl-Default"]
    main(main_args)
