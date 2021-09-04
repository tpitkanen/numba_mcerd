import numba


"""Module with patches to Numba"""


# New definition for NestedArray.__init__
def _new_init(self, dtype, shape):
    # assert dtype.bitwidth % 8 == 0, \
    #     "Dtype bitwidth must be a multiple of bytes"
    self._shape = shape
    name = "nestedarray(%s, %s)" % (dtype, shape)
    ndim = len(shape)
    super(numba.core.types.npytypes.NestedArray, self).__init__(dtype, ndim, 'C', name=name)


# New definition for NestedArray.size
def _new_size(self):
    if isinstance(self.dtype, numba.core.types.npytypes.Record):
        return self.dtype.size
    return self.dtype.bitwidth // 8


def patch_nested_array() -> None:
    """Monkey patch Numba to fix https://github.com/numba/numba/issues/3158

    Original issue: Numba can't handle nested custom dtype arrays.
    Error message:
    "AttributeError: 'Record' object has no attribute 'bitwidth'"
    """
    numba.core.types.npytypes.NestedArray.__init__ = _new_init
    numba.core.types.npytypes.NestedArray.size = property(_new_size)
