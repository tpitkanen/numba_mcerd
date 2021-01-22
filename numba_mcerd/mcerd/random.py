# import random
import math

import numpy as np
import numpy.random

import numba_mcerd.mcerd.constants as c


# Used in place of ldexp(<random>, -32)
from numba_mcerd import config

LDEXP_MULTIPLIER = 1 ** -32


# Numpy random generator
random_generator = None


class TempRNG:
    """Temporary random number generator that uses pre-generated numbers."""
    def __init__(self) -> None:
        self._numbers = None
        self._i = -1
        self._cached_normal = None

    def load(self, file=None) -> None:
        """Load pre-generated random numbers from a file"""
        if file is None:
            file = config.EXPORT_ROOT + "/ran_pcg_first_20000.txt"
        self._numbers = np.loadtxt(file)

    def random(self) -> float:
        """Return the next random number"""
        self._i += 1
        return self._numbers[self._i]

    def normal(self) -> float:
        """Return the next random number from normal distribution."""
        if self._cached_normal is not None:
            rv = self._cached_normal
            self._cached_normal = None
            return rv

        in_unit_circle = False
        while not in_unit_circle:
            v1 = 2.0 * self.random() - 1.0  # 0..1 -> -1..1
            v2 = 2.0 * self.random() - 1.0  # 0..1 -> -1..1
            r_squared = v1 * v1 + v2 * v2
            in_unit_circle = r_squared < 1.0 and r_squared != 0.0

        # Create two normal deviates with a Box-Muller transformation
        factor = math.sqrt(-2.0 * math.log(r_squared) / r_squared)
        self._cached_normal = v1 * factor
        return v2 * factor


class RndError(Exception):
    """Error in random number generator"""


def seed_rnd(seed: int):
    """Seed the random number generator. This must be called once,
    before generating any numbers."""
    # random.seed(seed)  # Native random

    # global random_generator
    # random_generator = numpy.random.default_rng(seed)  # Numpy random

    global random_generator
    random_generator = TempRNG()  # Fake RNG, seed is ignored
    random_generator.load()


# TODO: This is probably slow
def rnd(low: float, high: float, period: c.RndPeriod = c.RndPeriod.RND_CLOSED) -> float:
    """Generate a random number from low to high."""
    # value = 0.0
    length = high - low

    if length < 0.0:
        raise RndError("Length negative or zero")

    # TODO: Use PCG here or swap the RNG in the original code. Otherwise
    #       debugging by comparison is impossible.
    # return length * (random.random() * LDEXP_MULTIPLIER) + low  # Native random
    return length * (random_generator.random() * LDEXP_MULTIPLIER) + low

    # TODO: Implement these unlikely cases:
    # if period == c.RndPeriod.RND_CLOSED:
    #     raise NotImplementedError
    # elif period == c.RndPeriod.RND_OPEN:
    #     raise NotImplementedError
    # elif period == c.RndPeriod.RND_LEFT:
    #     raise NotImplementedError
    # elif period == c.RndPeriod.RND_RIGHT:
    #     raise NotImplementedError
    # elif period == c.RndPeriod.RND_SEED:
    #     raise NotImplementedError
    # else:
    #     raise exceptions.IonSimulationError
    #
    # assert low <= value <= high

    # return value


def gaussian() -> float:
    # return gasdev()
    # TODO: Is this equivalent to gasdev?
    return random_generator.normal()


# def gasdev() -> float:
#     raise NotImplementedError
