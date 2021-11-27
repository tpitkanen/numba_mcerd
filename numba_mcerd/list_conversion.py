"""Utilities for working around the lack of I/O in Numba."""


from enum import IntEnum
from typing import Any

import numba as nb

from numba_mcerd.mcerd import objects_dtype as od


class TypeInt(IntEnum):
    """Basic types represented by ints, for use in np.ndarray"""
    FLOAT = 1
    INT = 2
    STR = 3


_conversion_map = {
    TypeInt.FLOAT.value: lambda value: value,
    TypeInt.INT.value: lambda value: int(value),
    TypeInt.STR.value: lambda value: chr(int(value))
}


def from_float(type_int: int, value: float) -> Any:
    """Convert float-coded `value` to the type represented by `type_int`"""
    return _conversion_map[type_int](value)


def buffer_to_file(buf: od.Buffer, file) -> None:
    """Format and append buffer contents to file."""
    lines = []
    types = buf["types"]
    formats = buf["formats"]
    length = len(buf["types"])

    for row in buf["buf"][:buf["row_i"]]:
        converted = (from_float(types[i], row[i]) for i in range(length))
        formatted = (format(val, formats[i]) for i, val in enumerate(converted))

        # Single-line versions:
        # formatted = (format(_conversion_map[types[i]](row[i]), formats[i]) for i in range(length))
        # formatted = (format(from_float(types[i], row[i]), formats[i]) for i in range(length))

        lines.append(" ".join(formatted) + "\n")

    with open(file, "a") as f:
        f.writelines(lines)


@nb.njit(cache=True, nogil=True)
def set_buf(buf: od.Buffer, value: float) -> None:
    """Set a value to buffer.

    Increments column index, but doesn't handle row index management.
    """
    buf["buf"][buf["row_i"], buf["col_i"]] = value
    buf["col_i"] += 1
