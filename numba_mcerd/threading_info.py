"""Utilities for Numba threading."""


import numba as nb
import numpy as np


class ThreadInfoError(Exception):
    """Error while determining threading information"""


@nb.njit(cache=True, nogil=True)
def insertion_sort(list_):
    """Sort a list in place using insertion sort"""
    for i in range(len(list_)):
        j = i
        while j > 0 and list_[j - 1] > list_[j]:
            temp = list_[j]
            list_[j] = list_[j - 1]
            list_[j - 1] = temp
            j -= 1


# Would be more accurate as a part of simulation_loop, but doesn't work
# because the objects must be created outside of Numba.
# Uncacheable
@nb.njit(parallel=True, nogil=True)
def get_thread_ids(iterations_multiplier: int = 4) -> list:
    """Get a list of thread IDs used by a parallelized function

    Args:
        iterations_multiplier: how many iterations
            (iterations_multiplier * thread_count) should be performed

    Raises:
        ThreadInfoError: if the amount of found IDs does not match the amount of
            threads or if the IDs are not continuous (e.g. 0, 2, 3, 4)

    Returns:
        Ascending list of thread IDs
    """
    thread_count = get_thread_count()
    iterations = thread_count * iterations_multiplier
    thread_ids = np.zeros(iterations, dtype=np.int64)

    for i in nb.prange(iterations):
        thread_ids[i] = get_thread_id()

    unique_ids = list(set(thread_ids))
    # Importing a built-in sort function for the first time in Numba seems to
    # slow down execution by 2-3 seconds
    insertion_sort(unique_ids)
    unique_id_count = len(unique_ids)
    min_id = unique_ids[0]
    max_id = unique_ids[-1]

    if unique_id_count > thread_count:
        raise ThreadInfoError("Found more unique thread IDs than threads")

    if unique_id_count < thread_count:
        raise ThreadInfoError("Found fewer unique thread IDs than threads")

    if max_id - min_id + 1 != unique_id_count:
        raise ThreadInfoError("Found thread IDs were not continuous")

    return unique_ids


# Uncacheable
@nb.njit(nogil=True)
def get_thread_id() -> int:
    """Get current thread ID"""
    return nb.np.ufunc.parallel._get_thread_id()


# Uncacheable
@nb.njit(nogil=True)
def get_thread_count() -> int:
    """Get number of threads usable by Numba"""
    return nb.get_num_threads()


def set_thread_count(thread_count: int) -> None:
    """Set number of threads used by Numba"""
    nb.set_num_threads(thread_count)


if __name__ == "__main__":
    print(get_thread_ids())
