import logging

from numba_mcerd.mcerd import random, init_params
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
    ion = o.Ion()  # TODO: are these ions pointers or separate objects?
    cur_ion = o.Ion()
    previous_trackpoint_ion = o.Ion()
    ions_moving = []  # TODO: Initialize a list of ions?
    target = o.Target()
    scat = []  # TODO: Initialize a matrix of scatterings?
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
    init_params.init_params(g, target, args)

    # TODO


if __name__ == '__main__':
    # Path to data is hardcoded, change it to match your location
    # mcerd.exe is unused but included for similarity with original MCERD
    main_args = ["mcerd.exe", r"C:\kurssit\gradu\koodi\numba_mcerd\data\input\Cl-Default"]
    main(main_args)
