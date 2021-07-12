# Numba MCERD

Numba MCERD is a version of [MCERD](https://github.com/JYU-IBA/mcerd) in Python.

## Requirements and installation

Python 3.8 or newer is required. Git is required for version control and cloning the repository.

1. Clone the repository
```
git clone git@github.com:tpitkanen/numba_mcerd.git
cd numba_mcerd
```

2. Install to a virtual environtment (on Windows)

```
py -3.8 -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

3. Run the program (normal version)

```
env\Scripts\activate
set PYTHONPATH=%cd%
python numba_mcerd\main.py
```

or (Numba version)

```
env\Scripts\activate
set PYTHONPATH=%cd%
python numba_mcerd\main_jit.py 
```

Activating the virtual environment and setting the [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) is required once for each new shell session (window).

`PYTHONPATH` must be set for imports to work.

## Versions

There are currently two versions of Numba MCERD:
- Normal Python: slow, easy to debug.
- Just-in-time compiled: fast, harder to debug. Uses [Numba](https://numba.pydata.org/).

A CUDA-based version using Numba is planned.

Note that different versions have varying levels of completeness (the normal version is more complete).

## Data files

Files under `data/input/` contain absolute paths. Update them to match their real location. Files with absolute paths:
- Cl-Default
- Cl-Default.erd_detector

Absolute paths are used to match the original MCERD's behavior as closely as possible.

## Config

The program's behavior can be configured in `numba_mcerd/config.py`. Changing the `PROJECT_ROOT` absolute path is required.

## Random number generation

The used random number generator can be selected in [config](#Config).

`random_vanilla.py` provides 20000 pre-generated random numbers. They were generated using the original MCERD's RNG. Select this to match the original MCERD's numbers (for debugging). Using more than 20000 numbers results in an `IndexError`.

`random_numpy.py` uses NumPy to generate random numbers. Select this if matching the original MCERD's numbers is not needed.

`random_jit.py` uses NumPy to generate random numbers for the JIT version. It *should* match `random_numpy.py`.
