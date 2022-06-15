"""RNG for CUDA using Numba's own version RNG.

This file is ok to import directly, as opposed to selecting
the RNG source in config.py.
"""

from typing import Any

from numba import cuda
from numba.cuda import random


def seed_rnds(n, seed) -> Any:
    """Create and return RNG states"""
    return random.create_xoroshiro128p_states(n, seed=seed)


@cuda.jit(device=True)
def rnd(low: float, high: float, rng_states, thread_id) -> float:
    """Generate a random number from low to high"""
    return (high - low) * random.xoroshiro128p_uniform_float64(rng_states, thread_id) + low


@cuda.jit(device=True)
def gaussian(rng_states, thread_id) -> float:
    """Generate a random normally distributed number"""
    return random.xoroshiro128p_normal_float64(rng_states, thread_id)
