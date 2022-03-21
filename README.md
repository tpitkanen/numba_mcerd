# Numba MCERD

Numba MCERD is a version of [MCERD](https://github.com/JYU-IBA/mcerd) in Python.

## Requirements and installation

Python 3.8 or newer is required. Git is required for version control and cloning the repository.

These instructions are for CMD on Windows. Other operating systems use fundamentally similar commands, but the specifics differ.

1. Clone the repository
```
git clone git@github.com:tpitkanen/numba_mcerd.git
cd numba_mcerd
```

2. Install to a virtual environment

```
py -3.8 -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

3. Run the program:

```
env\Scripts\activate
set PYTHONPATH=%cd%
cd numba_mcerd
python main_jit.py
```

Possible versions are: `main.py` (normal Python), `main_jit.py` (Numba), `main_jit_mt.py` (Numba with multithreading).

Activating the virtual environment and setting the [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) is required once for each new shell session (window).

`PYTHONPATH` must be set for imports to work.

## Versions

There are currently two versions of Numba MCERD:
- Normal Python: slow, easy to debug.
- Just-in-time compiled: fast, harder to debug. Uses [Numba](https://numba.pydata.org/).
- Just-in-time compiled multithreaded. Same as previous, but uses all cores.

A CUDA-based version using Numba is planned.

Note that different versions have varying levels of completeness (the normal version is more complete).

## Data files

Files under `data/input/` contain file paths which may need to be updated to match their correct location. Both relative and absolute paths are supported. Files with paths:
- Cl-Default
- Cl-Default.erd_detector

As of 2022-03-21, [Potku](https://github.com/JYU-IBA/potku) generates settings files with absolute paths.

## Config

The program's behavior can be configured in `numba_mcerd/config.py`.

## Random number generation

The used random number generator can be selected in [config](#Config).

`random_vanilla.py` provides 20000 pre-generated random numbers. They were generated using the original MCERD's RNG. Select this to match the original MCERD's numbers (for debugging). Using more than 20000 numbers results in an `IndexError`.

`random_numpy.py` uses NumPy to generate random numbers. Select this if matching the original MCERD's numbers is not needed.

`random_jit.py` uses NumPy to generate random numbers for the JIT version. It *should* match `random_numpy.py`.
