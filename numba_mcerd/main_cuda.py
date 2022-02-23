import logging

import numba as nb
import numpy as np

from numba_mcerd import config, timer, patch_numba, logging_jit, list_conversion
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
    read_input, random_cuda
)
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
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

    if g_o.nions == 2:
        ions_o = [primary_ion_o, secondary_ion_o]
    elif g_o.nions == 3:
        ions_o = [primary_ion_o, secondary_ion_o, previous_trackpoint_ion_o]
    else:
        raise NotImplementedError

    logging.info("Initializing output files")
    init_params.init_io(g_o, primary_ion_o, target_o)

    pot = potential_jit.make_screening_table_dtype()

    ion_stack.cascades_create_additional_ions(g_o, detector_o, target_o, [])

    logging.info(f"{g_o.nions} ions, {target_o.natoms} target atoms")

    # (g.jibal.gsto.extrapolate = True)

    table_timer = timer.SplitTimer.init_and_start()
    scat_o = []
    for i in range(g_o.nions):
        if g_o.simtype == enums.SimType.RBS and i == enums.IonType.TARGET_ATOM.value:
            continue
        ions_o[i].scatindex = i
        scat_o.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
        for j in range(target_o.natoms):
            init_simu_jit.scattering_table(g_o, ions_o[i], target_o, scat_o[i][j], pot, j)
            cross_section_jit.calc_cross_sections(g_o, scat_o[i][j], pot)

    del pot

    table_timer.stop()
    print(f"table_timer: {table_timer}")

    gsto_index = -1
    for j in range(target_o.nlayers):
        target_o.layer[j].sto = [o.Target_sto() for _ in range(g_o.nions)]
        for i in range(g_o.nions):
            gsto_index += 1
            if g_o.simtype == enums.SimType.RBS and i == enums.IonType.TARGET_ATOM.value:
                continue
            elsto.calc_stopping_and_straggling_const(g_o, ions_o[i], target_o, j, gsto_index)
            # TODO: Real gsto instead

    init_detector.init_detector(g_o, detector_o)

    print_data_jit.print_data(g_o)

    if g_o.predata:
        init_params.init_recoiling_angle(target_o)

    trackid = int(ions_o[SECONDARY].Z) * 1_000 + g_o.seed % 1_000
    trackid *= 1_000_000

    logging.info("Converting objects to JIT")

    initialization_timer.stop()
    print(f"initialization_timer: {initialization_timer}")

    for ion in ions_o:
        ion.status = enums.IonStatus.NOT_FINISHED

    # dtype conversions
    dtype_conversion_timer = timer.SplitTimer.init_and_start()

    del primary_ion_o
    del secondary_ion_o
    del previous_trackpoint_ion_o

    presimus = ocd.convert_presimus(g_o)
    master = ocd.convert_master(g_o)
    g = ocd.convert_global(g_o)
    del g_o
    ions = np.array([ocd.convert_ion(ion) for ion in ions_o])
    for ion in ions_o:
        del ion
    del ions_o
    target = ocd.convert_target(target_o)
    del target_o
    scat = ocd.convert_scattering_nested(scat_o)
    del scat_o
    snext = ocd.convert_snext(snext_o)
    del snext_o
    detector = ocd.convert_detector(detector_o)
    del detector_o

    erd_buf = output_jit.create_erd_buffer(g)
    range_buf = finish_ion_jit.create_range_buffer(g)

    # TODO: Kernel config should be in config
    blocks_per_grid = 1
    threads_per_block = 1
    kernel_config = blocks_per_grid, threads_per_block

    rng_states = random_cuda.seed_rnds(1, 100)

    dtype_conversion_timer.stop()
    print(f"dtype_conversion_timer: {dtype_conversion_timer}")

    presimu_timer = timer.SplitTimer.init_and_start()
    trackid, ion_i, new_track = simulation_loop(
        g, presimus, master, ions, target, scat, snext, detector, trackid, ion_i, new_track, erd_buf, range_buf,
        kernel_config, rng_states)
    presimu_timer.stop()
    print(f"presimu_timer: {presimu_timer}")

    return  # TODO: Remove once the main loop is done

    analysis_timer = timer.SplitTimer.init_and_start()
    pre_simulation_jit.analyze_presimulation(g, presimus, master, target, detector)
    init_params_jit.init_recoiling_angle(target)
    analysis_timer.stop()
    print(f"analysis_timer: {analysis_timer}")

    main_simu_timer = timer.SplitTimer.init_and_start()
    # TODO: Don't pass presimus to main simulation.
    #       Numba doesn't like it if presimus is replaced with None.
    simulation_loop(
        g, presimus, master, ions, target, scat, snext, detector, trackid, ion_i, new_track, erd_buf, range_buf,
        kernel_config, rng_states)
    main_simu_timer.stop()
    print(f"main_sim_timer: {main_simu_timer}")

    print_timer = timer.SplitTimer.init_and_start()
    list_conversion.buffer_to_file(erd_buf, master["fperd"])
    list_conversion.buffer_to_file(range_buf, master["fprange"])
    finalize_jit.finalize(g, master)
    print(g.finstat)
    print_timer.stop()
    print(f"print_timer: {print_timer}")


# TODO: (not njit)
def run_simulation(g, master, ions, target, scat, snext, detector,
                   trackid, ion_i, new_track):
    # <Convert objects>

    # presim_timer = ...
    # run_pre_simulation(...)
    # presim_timer.split()
    # analyze_presimulation(...)
    # presim_timer.stop()

    # main_sim_timer = ...
    # run_main_simulation(...)
    # main_sim_timer.split()
    # finalize_jit.finalize(g, master)
    # main_sim_timer.stop()
    raise NotImplementedError


# @nb.njit(cache=True, nogil=True)  # TODO: Add back later, if possible
def simulation_loop(g, presimus, master, ions, target, scat, snext, detector,
                    trackid, ion_i, new_track, erd_buf, range_buf, kernel_config, rng_states):
    # logging_jit.info("Starting simulation")

    if g.simstage == enums.SimStage.PRE:
        start = 0
        stop = g.npresimu
    else:
        start = g.npresimu
        stop = g.nsimu

    # for i in range(start, stop):  # TODO: Add back later
    for i in range(0, 5):
        if i % 10000 == 0:
            print(i)

        g.cion = i  # TODO: Replace/remove for MT

        # output.output_data(g)  # Only prints status info

        cur_ion = ions[PRIMARY]

        ion_simu_jit.create_ion(g, cur_ion, target)
        if g.rough:
            ion_simu_jit.move_target(target)

        primary_finished = False
        while not primary_finished:
            ion_simu_jit.next_scattering_cuda[1, 1](g, cur_ion, target, scat, snext, rng_states)
    #         nscat = ion_simu_jit.move_ion(g, cur_ion, target, snext)
    #
    #         if nscat == enums.ScatteringType.ERD_SCATTERING:
    #             if erd_scattering_jit.erd_scattering(
    #                     g, ions[PRIMARY], ions[SECONDARY], target, detector):
    #                 cur_ion = ions[SECONDARY]
    #
    #         if cur_ion.status == enums.IonStatus.FIN_RECOIL or cur_ion.status == enums.IonStatus.FIN_OUT_DET:
    #             if g.simstage == enums.SimStage.PRE:
    #                 pre_simulation_jit.finish_presimulation(g, presimus, detector, cur_ion)
    #                 cur_ion = ions[PRIMARY]
    #             else:
    #                 erd_detector_jit.move_to_erd_detector(g, cur_ion, target, detector)
    #
    #         if (nscat == enums.ScatteringType.MC_SCATTERING
    #                 and cur_ion.status == enums.IonStatus.NOT_FINISHED
    #                 and not g.nomc):
    #             if ion_simu_jit.mc_scattering(
    #                     g, cur_ion, ions[SECONDARY], target, detector, scat, snext):  # ion_stack.next_ion()
    #                 # This block is never reached in ERD mode
    #                 cur_ion = ions[SECONDARY]  # ion_stack.next_ion()
    #                 found = False
    #                 for j in range(g.nions):
    #                     if j == TARGET_ATOM and g.simtype == enums.SimType.RBS:
    #                         continue
    #                     if (round(ions[j].Z) == round(cur_ion.Z)
    #                             and round(ions[j].A / c.C_U) == round(ions[j].A / c.C_U)):
    #                         # FIXME: Comparing average mass by rounding is a bad idea.
    #                         #        See the original code for more information.
    #                         found = True
    #                         cur_ion.scatindex = j
    #                 if not found:
    #                     # logging_jit.warning(
    #                     #     f"Recoil cascade not possible, since recoiling ion Z={float(cur_ion.Z)} and A={float(cur_ion.A / c.C_U)} u are not in ion table (and therefore not in scattering table or stopping/straggling tables)")
    #                     logging_jit.warning(f"Recoil cascade not possible, since recoiling ion Z and A are not in ion table (and therefore not in scattering table or stopping/straggling tables)")
    #                     raise NotImplementedError
    #                     # cur_ion = ion_stack.prev_ion()
    #                 else:
    #                     ion_i += 1
    #                     cur_ion.ion_i = ion_i
    #                     cur_ion.trackid = trackid
    #
    #                 # logging_jit.debug(...)
    #
    #         # debug: loop over layers, print cur_ion.tlayer and set prev_layer_debug
    #
    #         if g.output_trackpoints:
    #             raise NotImplementedError
    #
    #         if cur_ion.type == SECONDARY and cur_ion.status != enums.IonStatus.NOT_FINISHED:
    #             g.finstat[SECONDARY, cur_ion.status] += 1
    #
    #         while ion_simu_jit.ion_finished(g, cur_ion, target):
    #             # logging_jit.debug(...)
    #
    #             if g.output_trackpoints:
    #                 raise NotImplementedError
    #
    #             cur_ion.trackid = trackid if not new_track else 0
    #             # No new track is made if ion doesn't make it to the
    #             # energy detector or if it's a scaling ion
    #
    #             if cur_ion.type <= SECONDARY:
    #                 output_jit.output_erd(g, cur_ion, target, detector, erd_buf)
    #             if cur_ion.type == PRIMARY:
    #                 primary_finished = True
    #                 break
    #             cur_ion = ions[PRIMARY]  # ion_stack.prev_ion()
    #             if cur_ion.type != PRIMARY and g.output_trackpoints:
    #                 raise NotImplementedError

            primary_finished = True  # TODO: Remove once the whole loop works

        # logging_jit.debug(...)

        g.finstat[PRIMARY, cur_ion.status] += 1
        finish_ion_jit.finish_ion(g, cur_ion, range_buf)  # Output info if FIN_STOP or FIN_TRANS

    return trackid, ion_i, new_track


if __name__ == '__main__':
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", f"{config.PROJECT_ROOT}/data/input/Cl-Default"]
    main(main_args)
