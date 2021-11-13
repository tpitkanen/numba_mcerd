def print_ion_position(g, ion, label, flag):
    """Print ion position in the laboratory coordinates"""
    raise NotImplementedError


def objmode_out(*args, **kwargs) -> None:
    """Helper function for debugging with numba.objmode context manager"""
    # simulation_loop parameters:
    # debug.objmode_out(g=g, master=master, ions=ions, target=target, scat=scat, snext=snext, detector=detector, trackid=trackid, ion_i=ion_i, new_track=new_track)
    pass


def objmode_print(*args, **kwargs) -> None:
    """Helper function for printing debug info with numba.objmode context manager"""
    print(f"args: {args}")
    print(f"kwargs: {kwargs}")

