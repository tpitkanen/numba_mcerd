from pathlib import Path

# Select random source here for non-Numba use.
# For Numba, import random_jit.py directly.
import numba_mcerd.mcerd.random_numpy as rand

PROJECT_ROOT = str(Path.cwd().parent)  # Automatic path
# PROJECT_ROOT = r"C:/path/to/numba_mcerd"  # Manual path
DATA_ROOT = rf"{PROJECT_ROOT}/data"
CONSTANTS_ROOT = rf"{DATA_ROOT}/constants"
EXPORT_ROOT = rf"{DATA_ROOT}/export"
PICKLE_ROOT = rf"{DATA_ROOT}/pickled"

# Choose if pickled objects should be used instead of recalculating them.
# Objects must be generated once before using. See pickler.py for more information.
LOAD_PICKLE = False

# TODO: Add DUMP_PICKLE for more granularity

# Choose how many threads to use in parallel mode.
# Must be between 1 and NUMBA_NUM_THREADS. If set to None, Numba defaults are used.
PARALLEL_THREAD_COUNT = None

# Set arguments here.
# mcerd.exe is unused but included for similarity with original MCERD.
MAIN_ARGS = ["mcerd.exe", rf"{PROJECT_ROOT}/data/input/O-Default"]

# Generate gsto files manually (no instructions available for now) and set their path here
CONST_STO_PATH = rf"{EXPORT_ROOT}/o_sto.txt"
CONST_STRAGG_PATH = rf"{EXPORT_ROOT}/o_stragg.txt"
