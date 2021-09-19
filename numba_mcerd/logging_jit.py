import numba as nb

"""A simplified logger for use with JIT functions.

Doesn't support setting a minimum level (global values are problematic in Numba).
"""

# TODO: Is there an easy way to display time and line number?
# TODO: Can't cast float64 or int64 to str in Numba -> can't use f-strings,
#       even if manually wrapped in float() or int().

@nb.njit(cache=True)
def critical(message: str):
    print(f"Critical: {message}")


@nb.njit(cache=True)
def error(message: str):
    print(f"Error: {message}")


@nb.njit(cache=True)
def warning(message: str):
    print(f"Warning: {message}")


@nb.njit(cache=True)
def info(message: str):
    print(f"Info: {message}")


@nb.njit(cache=True)
def debug(message: str):
    print(f"Debug: {message}")
