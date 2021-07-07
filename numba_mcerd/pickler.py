"""Utilities for serializing and deserializing objects.

This module is intended to speed up debugging. Dump time-consuming
objects once (in run mode), then load them in debugging mode.

Note that loading untrusted pickle files is unsafe.
"""

from pathlib import Path
import pickle
from typing import Any

from numba_mcerd import config


# TODO: Create tests


def _get_path(name: str) -> Path:
    """Get path to pickle file"""
    return Path(config.PICKLE_ROOT) / f"{name}.pickle"


def dump(obj: Any, name: str) -> None:
    """Dump obj to "{name}.pickle" file"""
    pickle_file = _get_path(name)

    if not pickle_file.parent.exists():
        pickle_file.parent.mkdir(parents=True)

    with pickle_file.open("wb") as f:
        pickle.dump(obj, f)


def load(name: str) -> Any:
    """Load obj from "{name}.pickle" file"""
    pickle_file = _get_path(name)
    with pickle_file.open("rb") as f:
        obj = pickle.load(f)
    return obj
