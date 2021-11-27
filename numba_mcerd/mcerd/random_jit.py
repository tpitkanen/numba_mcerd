"""RNG using Numba-optimized NumPy random.

This file is probably ok to import directly, as opposed to selecting
the RNG source in config.py.
"""

import numba as nb
import numpy as np

from numba_mcerd import logging_jit
from numba_mcerd.mcerd import enums


# https://numba.pydata.org/numba-doc/dev/reference/pysupported.html
# Calling random.seed() from non-Numba code (or from object mode code)
# will seed the Python random generator, not the Numba random generator.


class RndError(Exception):
    """Error in random number generator"""


# TODO: Unit tests


@nb.njit(cache=True, nogil=True)
def seed_rnd(seed: int) -> None:
    """Seed the random number generator. This must be called once,
    before generating any numbers."""
    np.random.seed(seed)


@nb.njit(cache=True, nogil=True)
def rnd(low: float, high: float, period=None) -> float:
    """Generate a random number from low to high."""
    length = high - low

    if length < 0.0:
        raise RndError("Length negative or zero")

    value = length * np.random.random() + low
    if value <= low or value >= high:  # Unlikely to occur even once
        # print(f"RNG value={float(value)} may exceed bounds of low={float(low)}, high={float(high)}, period={period}")
        logging_jit.warning(f"RNG may exceed bounds")

    return value


@nb.njit(cache=True, nogil=True)
def gaussian() -> float:
    return np.random.normal()


def main():
    seed_rnd(100)
    print([rnd(0., 10., None) for _ in range(10)])
    print([gaussian() for _ in range(10)])


if __name__ == '__main__':
    main()

