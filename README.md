# Numba MCERD

Numba MCERD is a version of [MCERD](https://github.com/JYU-IBA/mcerd) in Python.

## Requirements and installation

Python 3.8 or newer is required.

Installation on Windows:

```
py -3.8 -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

Running the program (normal version):

```
env\Scripts\activate
python numba_mcerd\main.py
```

or (Numba version):

```
env\Scripts\activate
python numba_mcerd\main_jit.py 
```

Activating the environment (`env\Scripts\activate`) is required once for each new shell session (window).

## Data files

Files under `data/input/` contain absolute paths. Update them to match their real location. Files with absolute paths:
- Cl-Default
- Cl-Default.erd_detector

## Config

The program's behavior can be configured in `numba_mcerd/config.py`. Changing the `PROJECT_ROOT` absolute path is required.

## Versions

There are currently two versions of Numba MCERD:

- Normal Python: slow, easy to debug.
- Just-in-time compiled (Numba): fast, harder to debug.

A CUDA-based version using Numba is planned.

Note that different versions have varying levels of completeness (the normal version is more complete).

## Random number generation

Random Number Generator can be selected in [Config](#Config).

`random_vanilla.py` provides 20000 pre-generated random numbers. They were generated using the original MCERD's RNG. Select this to match the original MCERD's numbers (for debugging). Using more than 20000 numbers results in an `IndexError`.

`random_numpy.py` uses NumPy to generate random numbers. Select this if matching the original MCERD's numbers is not needed.

`random_jit.py` uses NumPy to generate random numbers for the JIT version. It *should* match `random_numpy.py`.
