import numpy as np
import numba as nb

import numba_mcerd.mcerd.constants as c


# https://numba.pydata.org/numba-doc/dev/reference/pysupported.html
# Calling random.seed() from non-Numba code (or from object mode code)
# will seed the Python random generator, not the Numba random generator.


class RndError(Exception):
    """Error in random number generator"""


# TODO: Unit tests


@nb.njit(cache=True)
def seed_rnd(seed: int) -> None:
    """Seed the random number generator. This must be called once,
    before generating any numbers."""
    np.random.seed(seed)


@nb.njit(cache=True)
def rnd(low: float, high: float, period=None) -> float:
    """Generate a random number from low to high."""
    length = high - low

    if length < 0.0:
        raise RndError("Length negative or zero")

    return length * np.random.random() + low
    # TODO: Implement period

    # TODO: Slightly simpler option, but doesn't check that length > 0:
    # return np.random.uniform(low, high)  # [low, high)


@nb.njit(cache=True)
def gaussian() -> float:
    return np.random.normal()


def main():
    seed_rnd(100)
    print([rnd(0., 10., None) for _ in range(10)])
    print([gaussian() for _ in range(10)])


if __name__ == '__main__':
    main()

