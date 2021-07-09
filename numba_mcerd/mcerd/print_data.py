import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s


def print_data(g: o.Global) -> None:
    """Output some simulation settings to g.master.fpdat"""
    with g.master.fpdat.open("a") as f:
        # TODO: The original uses varying amounts of tabs. Replace them with spaces?

        f.write(f"Number of simulated ion histories: \t\t{g.nsimu}\n")
        f.write(f"Minimum energy for simulation: \t\t\t{g.emin / c.C_KEV:.3f} keV\n")
        # TODO: There could be a space before s.SYM_DEG, but the original doesn't have it either
        f.write(f"Minimum scattering angle: \t\t\t{g.minangle / c.C_DEG:.2f}{s.SYM_DEG}\n")
        f.write(f"Seed number of random number generator: \t{g.seed}\n")
        f.write(f"Initial energy of the ion:\t\t\t{g.E0 / c.C_MEV:.4f} MeV\n")
