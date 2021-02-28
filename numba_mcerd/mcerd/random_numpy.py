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

    return length * np.random.random() + low
    # TODO: Implement period

    # TODO: Slightly simpler option, but doesn't check that length > 0:
    # return np.random.uniform(low, high)  # [low, high)


def gaussian() -> float:
    return np.random.normal()
