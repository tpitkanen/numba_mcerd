import random

import numba_mcerd.mcerd.constants as c


class RndError(Exception):
    """Error in random number generator"""


def seed_rnd(seed: int):
    """Seed the random number generator. This must be called once,
    before generating any numbers."""
    random.seed(seed)


# TODO: This is probably slow
def rnd(low: float, high: float, period: c.RndPeriod = c.RndPeriod.RND_CLOSED) -> float:
    """Generate a random number from low to high."""
    # value = 0.0
    length = high - low

    if length < 0.0:
        raise RndError("Length negative or zero")

    # TODO: Use PCG here or swap the RNG in the original code. Otherwise
    #       debugging by comparison is impossible.
    return length * random.random() + low

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


def gaussian(seed: int) -> float:
    return gasdev(seed)


def gasdev(idum: int) -> float:
    raise NotImplementedError
