import numba as nb

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
    read_input
)
import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.objects_convert_dtype as ocd


# These are too annoying to type
PRIMARY = enums.IonType.PRIMARY.value
SECONDARY = enums.IonType.SECONDARY.value
TARGET_ATOM = enums.IonType.TARGET_ATOM.value

# TODO: should simulation_loop be copied too?


def run_simulation(*args, **kwargs):
    patch_numba.patch_nested_array()
    random_jit.seed_rnd(args[0].seed)

    try:
        simulation_loop(*args)
    except Exception as e:
        print(e)
    return args


@nb.njit(cache=True, nogil=True)
def simulation_loop(g, presimus, master, ions, target, scat, snext, detector,
                    trackid, ion_i, new_track, erd_buf, range_buf):
    # logging_jit.info("Starting simulation")

    if g.simstage == enums.SimStage.PRE:
        start = 0
        stop = g.npresimu
    else:
        start = g.npresimu
        stop = g.nsimu

    for i in range(start, stop):
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
            ion_simu_jit.next_scattering(g, cur_ion, target, scat, snext)
            nscat = ion_simu_jit.move_ion(g, cur_ion, target, snext)

            if nscat == enums.ScatteringType.ERD_SCATTERING:
                if erd_scattering_jit.erd_scattering(
                        g, ions[PRIMARY], ions[SECONDARY], target, detector):
                    cur_ion = ions[SECONDARY]

            if cur_ion.status == enums.IonStatus.FIN_RECOIL or cur_ion.status == enums.IonStatus.FIN_OUT_DET:
                if g.simstage == enums.SimStage.PRE:
                    pre_simulation_jit.finish_presimulation(g, presimus, detector, cur_ion)
                    cur_ion = ions[PRIMARY]
                else:
                    erd_detector_jit.move_to_erd_detector(g, cur_ion, target, detector)

            if (nscat == enums.ScatteringType.MC_SCATTERING
                    and cur_ion.status == enums.IonStatus.NOT_FINISHED
                    and not g.nomc):
                if ion_simu_jit.mc_scattering(
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
                        # logging_jit.warning(
                        #     f"Recoil cascade not possible, since recoiling ion Z={float(cur_ion.Z)} and A={float(cur_ion.A / c.C_U)} u are not in ion table (and therefore not in scattering table or stopping/straggling tables)")
                        logging_jit.warning(f"Recoil cascade not possible, since recoiling ion Z and A are not in ion table (and therefore not in scattering table or stopping/straggling tables)")
                        raise NotImplementedError
                        # cur_ion = ion_stack.prev_ion()
                    else:
                        ion_i += 1
                        cur_ion.ion_i = ion_i
                        cur_ion.trackid = trackid

                    # logging_jit.debug(...)

            # debug: loop over layers, print cur_ion.tlayer and set prev_layer_debug

            if g.output_trackpoints:
                raise NotImplementedError

            if cur_ion.type == SECONDARY and cur_ion.status != enums.IonStatus.NOT_FINISHED:
                g.finstat[SECONDARY, cur_ion.status] += 1

            while ion_simu_jit.ion_finished(g, cur_ion, target):
                # logging_jit.debug(...)

                if g.output_trackpoints:
                    raise NotImplementedError

                cur_ion.trackid = trackid if not new_track else 0
                # No new track is made if ion doesn't make it to the
                # energy detector or if it's a scaling ion

                if cur_ion.type <= SECONDARY:
                    output_jit.output_erd(g, cur_ion, target, detector, erd_buf)
                if cur_ion.type == PRIMARY:
                    primary_finished = True
                    break
                cur_ion = ions[PRIMARY]  # ion_stack.prev_ion()
                if cur_ion.type != PRIMARY and g.output_trackpoints:
                    raise NotImplementedError

        # logging_jit.debug(...)

        g.finstat[PRIMARY, cur_ion.status] += 1
        finish_ion_jit.finish_ion(g, cur_ion, range_buf)  # Output info if FIN_STOP or FIN_TRANS

    return trackid, ion_i, new_track
