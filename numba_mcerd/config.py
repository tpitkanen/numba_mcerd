# Select random source here
import numba_mcerd.mcerd.random_numpy as rand

PROJECT_ROOT = r"C:/kurssit/gradu/koodi/numba_mcerd"
DATA_ROOT = rf"{PROJECT_ROOT}/data"
CONSTANTS_ROOT = rf"{DATA_ROOT}/constants"
EXPORT_ROOT = rf"{DATA_ROOT}/export"
PICKLE_ROOT = rf"{DATA_ROOT}/pickled"

# Choose if pickled objects should be used instead of recalculating them.
# Objects must be generated once before using. See pickler.py for more information.
LOAD_PICKLE = False

# TODO: Add DUMP_PICKLE for more granularity
