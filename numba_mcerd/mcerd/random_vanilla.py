"""RNG using precalculated numbers.

Do not import this directly, select RNG to use in config.py instead.
"""
import math

import numpy as np

from numba_mcerd import config
from numba_mcerd.mcerd import enums


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
    # random_generator = np.random.default_rng(seed)  # Numpy random

    global random_generator
    random_generator = TempRNG()  # Fake RNG, seed is ignored
    random_generator.load()


def rnd(low: float, high: float, period: enums.RndPeriod = enums.RndPeriod.RND_CLOSED) -> float:
    """Generate a random number from low to high."""
    length = high - low

    if length < 0.0:
        raise RndError("Length negative or zero")

    return length * random_generator.random() + low
    # TODO: Implement these unlikely cases:
    # if period == enums.RndPeriod.RND_CLOSED:
    #     raise NotImplementedError
    # elif period == enums.RndPeriod.RND_OPEN:
    #     raise NotImplementedError
    # elif period == enums.RndPeriod.RND_LEFT:
    #     raise NotImplementedError
    # elif period == enums.RndPeriod.RND_RIGHT:
    #     raise NotImplementedError
    # elif period == enums.RndPeriod.RND_SEED:
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
