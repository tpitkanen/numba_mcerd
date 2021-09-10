import copy
import logging

import numpy as np

from numba_mcerd import config, pickler, timer
from numba_mcerd.mcerd import (
    cross_section,
    elsto,
    enums,
    erd_detector,
    erd_scattering,
    finalize,
    finish_ion,
    init_detector,
    init_params,
    init_simu,
    ion_simu,
    ion_stack,
    output,
    potential,
    pre_simulation,
    print_data,
    read_input
)
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o


# These are too annoying to type
PRIMARY = enums.IonType.PRIMARY.value
SECONDARY = enums.IonType.SECONDARY.value
TARGET_ATOM = enums.IonType.TARGET_ATOM.value


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
    setup_logging()

    # Variables

    logging.debug("Initializing variables")

    initialization_timer = timer.SplitTimer.init_and_start()

    g = o.Global()
    primary_ion = o.Ion()  # Not really used in the simulation loop
    secondary_ion = o.Ion()  # Not really used in the simulation loop
    previous_trackpoint_ion = o.Ion()  # Not used unless simulation is RBS
    target = o.Target()
    scat = None  # len [g.nions][MAXELEMENTS]
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

    if g.nions == 2:
        ions = [primary_ion, secondary_ion]
    elif g.nions == 3:
        ions = [primary_ion, secondary_ion, previous_trackpoint_ion]
    else:
        raise NotImplementedError

    logging.info("Initializing output files")
    init_params.init_io(g, primary_ion, target)

    pot = o.Potential()
    potential.make_screening_table(pot)

    ion_stack.cascades_create_additional_ions(g, detector, target, [])

    logging.info(f"{g.nions} ions, {target.natoms} target atoms")

    # (g.jibal.gsto.extrapolate = True)

    # FIXME: init_simu.scattering_table outputs to fpout.
    #        It should be output even if using pickler.
    if not config.LOAD_PICKLE:
        table_timer = timer.SplitTimer.init_and_start()

        scat = []
        for i in range(g.nions):
            if g.simtype == enums.SimType.RBS and i == enums.IonType.TARGET_ATOM.value:
                continue
            ions[i].scatindex = i
            scat.append([o.Scattering() for _ in range(c.MAXELEMENTS)])
            for j in range(target.natoms):
                # logging.debug(f"Calculating scattering between ions ...")
                init_simu.scattering_table(g, ions[i], target, scat[i][j], pot, j)
                cross_section.calc_cross_sections(g, scat[i][j], pot)

        table_timer.stop()
        pickler.dump(scat, "scat")

        print(f"table_timer: {table_timer}")
        # Times (03f95b0559f2c0eea3d92eb34eb8a37306a39231)
        # print(table_timer.start_time, table_timer.elapsed_laps)
        # 1.8580092 [4.2860329, 28.060028000000003] (run)
        # 4.8511551 [11.4422601, 158.6126873] (debug)
    else:
        scat = pickler.load("scat")
        for i in range(g.nions):  # Fix for pickling
            ions[i].scatindex = i

    gsto_index = -1
    for j in range(target.nlayers):
        target.layer[j].sto = [o.Target_sto() for _ in range(g.nions)]
        for i in range(g.nions):
            gsto_index += 1
            if g.simtype == enums.SimType.RBS and i == enums.IonType.TARGET_ATOM.value:
                continue
            # TODO: This is a temporary way to avoid implementing GSTO
            elsto.calc_stopping_and_straggling_const(g, ions[i], target, j, gsto_index)

            # TODO: ions[i] could be saved as a variable
            # elsto.calc_stopping_and_straggling(g, ions[i], target, j)
            # logging.info(f"Stopping calculated, scatindex={ions[i].scatindex}, layer={j}, stodiv="
            #              f"{target.layer[j].sto[ions[i].scatindex].stodiv}")

    # TODO: Needed for RBS
    # if g.simtype == c.SimType.RBS:
    #     ion_stack.copy_ions(primary_ion, target, TARGET_ATOM, SECONDARY, False)
    #     ion_stack.copy_ions(primary_ion, target, SECONDARY, PRIMARY, True)

    init_detector.init_detector(g, detector)

    print_data.print_data(g)

    if g.predata:
        init_params.init_recoiling_angle(target)

    trackid = int(ions[SECONDARY].Z) * 1_000 + g.seed % 1_000
    trackid *= 1_000_000

    initialization_timer.stop()
    print(f"initialization_timer: {initialization_timer}")

    # Debugging counters
    outer_loop_counts = np.zeros(shape=g.nsimu, dtype=np.int64)
    inner_loop_counts = np.zeros(shape=g.nsimu, dtype=np.int64)

    logging.info("Starting simulation")

    presim_timer = timer.SplitTimer.init_and_start()
    for i in range(g.nsimu):
        g.cion = i

        outer_loop_count = 0
        inner_loop_count = 0

        output.output_data(g)

        cur_ion = ions[PRIMARY]

        # Pointer stuff, probably not needed in Python:
        # if debug:
        # logging.debug(f"T moving to primary {id(cur_ion)}, i={...}")

        ion_simu.create_ion(g, cur_ion, target)
        if g.rough:
            ion_simu.move_target(target)

        primary_finished = False
        while not primary_finished:
            outer_loop_count += 1

            ion_simu.next_scattering(g, cur_ion, target, scat, snext)
            nscat = ion_simu.move_ion(g, cur_ion, target, snext)

            if nscat == enums.ScatteringType.ERD_SCATTERING:
                if erd_scattering.erd_scattering(
                        g, ions[PRIMARY], ions[SECONDARY], target, detector):
                    cur_ion = ions[SECONDARY]

            if cur_ion.status == enums.IonStatus.FIN_RECOIL or cur_ion.status == enums.IonStatus.FIN_OUT_DET:
                if g.simstage == enums.SimStage.PRE:
                    pre_simulation.finish_presimulation(g, detector, cur_ion)
                    cur_ion = ions[PRIMARY]
                else:
                    erd_detector.move_to_erd_detector(g, cur_ion, target, detector)

            # TODO: Separate loop to pre and main, move this in-between
            if g.simstage == enums.SimStage.PRE and g.cion == g.npresimu - 1:
                pre_simulation.analyze_presimulation(g, target, detector)
                init_params.init_recoiling_angle(target)

                presim_timer.stop()
                print(f"presim_timer: {presim_timer}")

                main_sim_timer = timer.SplitTimer.init_and_start()

            if (nscat == enums.ScatteringType.MC_SCATTERING
                    and cur_ion.status == enums.IonStatus.NOT_FINISHED
                    and not g.nomc):
                if ion_simu.mc_scattering(
                        g, cur_ion, ions[SECONDARY], target, detector, scat, snext):  # ion_stack.next_ion()
                    # This block is never reached in ERD mode
                    cur_ion = ions[SECONDARY]  # ion_stack.next_ion()
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

            if cur_ion.type.value == SECONDARY and cur_ion.status != enums.IonStatus.NOT_FINISHED:
                g.finstat[SECONDARY][cur_ion.status.value] += 1

            while ion_simu.ion_finished(g, cur_ion, target).value:
                inner_loop_count += 1

                # logging.debug(...)

                if g.output_trackpoints:
                    raise NotImplementedError

                cur_ion.trackid = trackid if not new_track else 0
                # No new track is made if ion doesn't make it to the
                # energy detector or if it's a scaling ion

                if cur_ion.type <= SECONDARY:
                    output.output_erd(g, cur_ion, target, detector)
                if cur_ion.type.value == PRIMARY:
                    primary_finished = True
                    break
                cur_ion = ions[PRIMARY]  # ion_stack.prev_ion()
                if cur_ion.type.value != PRIMARY and g.output_trackpoints:
                    raise NotImplementedError

        outer_loop_counts[g.cion] = outer_loop_count
        inner_loop_counts[g.cion] = inner_loop_count

        # logging.debug(...)

        g.finstat[PRIMARY][cur_ion.status.value] += 1
        finish_ion.finish_ion(g, cur_ion)  # Print info if FIN_STOP or FIN_TRANS

    finalize.finalize(g)  # Print statistics

    # noinspection PyUnboundLocalVariable
    main_sim_timer.stop()
    print(f"main_sim_timer: {main_sim_timer}")


if __name__ == '__main__':
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", f"{config.PROJECT_ROOT}/data/input/Cl-Default"]
    main(main_args)
