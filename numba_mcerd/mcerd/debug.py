def print_ion_position(g, ion, label, flag):
    """Print ion position in the laboratory coordinates"""
    raise NotImplementedError


def out(*a, **k) -> None:
    """Helper function for debugging with numba.objmode context manager"""
    # simulation_loop parameters:
    # debug.objmode_out("simulation_loop", g=g, master=master, ions=ions, target=target, scat=scat, snext=snext,
    #                   detector=detector, trackid=trackid, ion_i=ion_i, new_track=new_track)
    pass


def out_print(*a, **k) -> None:
    """Helper function for printing debug info with numba.objmode context manager"""
    print(f"args: {a}")
    print(f"kwargs: {k}")


def out_convert(*a, **k) -> None:
    """Helper function for debugging with numba.objmode context manager that
    converts objects to dict
    """
    a_dicts = [numpy_to_dict(obj) for obj in a]
    k_dicts = {key: numpy_to_dict(k[key]) for key in k}
    pass


def numpy_to_dict(obj):
    """Convert numpy object to dict. Doesn't affect inner objects."""
    if not hasattr(obj, "dtype"):
        return obj
    d = {name: obj[name] for name in obj.dtype.names}
    # Sort keys for debugger
    return {key: d[key] for key in sorted(d)}
