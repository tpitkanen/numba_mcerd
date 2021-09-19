"""RNG using NumPy random.

Do not import this directly, select RNG to use in config.py instead.
"""

import numpy as np

import numba_mcerd.mcerd.constants as c


# TODO: Unit tests


class RndError(Exception):
    """Error in random number generator"""


def seed_rnd(seed: int) -> None:
    """Seed the random number generator. This must be called once,
    before generating any numbers."""
    np.random.seed(seed)


def rnd(low: float, high: float, period=None) -> float:
    """Generate a random number from low to high."""
    length = high - low

    if length < 0.0:
        raise RndError("Length negative or zero")

    value = length * np.random.random() + low
    if value <= low or value >= high:  # Unlikely to occur even once
        print(f"RNG value={value} may exceed bounds of low={low}, high={high}, period={period}")

    return value


def gaussian() -> float:
    return np.random.normal()
