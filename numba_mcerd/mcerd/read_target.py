import logging
from pathlib import Path

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import read_input, enums


class ReadTargetError(Exception):
    """Error while reading target file"""


# Called from two locations
def read_target_file(filename: str, g: o.Global, target: o.Target) -> None:
    """Read target configuration from file

    Args:
        filename: file to read configuration from
        g: global object
        target: target to set
    """
    concentrations = [0.0] * c.MAXELEMENTS  # Originally con
    atoms = [0.0] * c.MAXELEMENTS  # Originally atom
    depth = 0.0  # Originally n

    logging.info(f"Opening target file {filename}")
    fp = Path(filename).open("r")

    natoms = norigatom = target.natoms
    nlayer = noriglayer = target.nlayers

    # TODO: This is Python 3.8 only, maybe replace it with a non-walrus version
    #       for Python 3.7 compatibility
    # Read target elements
    while line := fp.readline().strip():
        M, symbol = read_input.get_float(line)
        M *= c.C_U

        jibal_atom = g.jibal.get_element_by_name(symbol)
        if jibal_atom is None:
            raise ReadTargetError(f"Could not find element for symbol '{symbol}'")
        target.ele[natoms].Z = jibal_atom.Z
        target.ele[natoms].A = M

        logging.info(f"Atom {natoms} {jibal_atom.Z} {M / c.C_U}")
        natoms += 1
    target.natoms = natoms

    # Read layers
    while line := fp.readline().strip():
        logging.info(f"layer: {nlayer}")
        layer = target.layer[nlayer]
        layer.type = enums.TargetType.FILM
        layer.gas = False

        if line[0].isalpha():
            raise NotImplementedError  # TODO

        if layer.type == enums.TargetType.FILM:
            number, line = read_input.get_float(line)
            unit, line = read_input.get_unit_value(line, c.C_NM)
            logging.info(f"thickness {number * unit / c.C_NM} nm")
            layer.dlow = depth
            layer.dhigh = depth + number * unit
            depth = layer.dhigh

        line = fp.readline().strip()
        stofile, line = read_input.get_word(line)
        logging.debug(f"stofile: {stofile}")
        # TODO: c.MAXSTOFILEPREFIXLEN (in else)
        layer.stofile_prefix = "" if stofile == "ZBL" else stofile

        line = fp.readline().strip()
        if line != "ZBL":
            # This isn't handled in any way in the original code
            raise ReadTargetError("Non-ZBL second stofile not supported")

        line = fp.readline().strip()
        number, line = read_input.get_float(line)
        if not line:
            raise ReadTargetError(f"Unitless density '{number}' is not an acceptable value")
        unit, line = read_input.get_unit_value(line, c.C_G_CM3)
        density = number * unit
        logging.info(f"Density: {density / c.C_G_CM3} {s.SYM_G_CM3}")

        # ifdef GAS_OR_SOLID ...

        logging.info(f"{'gaseous' if layer.gas else 'solid'}")

        i = 0
        sum_mass = 0.0  # Mass  # Originally sumM
        sum_concentration = 0.0  # Concentration  # Originally sumcon
        while line := fp.readline().strip():
            atoms[i], line = read_input.get_float(line)
            concentrations[i], line = read_input.get_float(line)
            # TODO: This seems to be just rounding, but why is norigatom added?
            j = int(atoms[i] + norigatom + 0.5)
            layer.atom[i] = j
            sum_mass += target.ele[j].A * concentrations[i]
            sum_concentration += concentrations[i]
            logging.info(
                f"Atom: {j} {atoms[i]}, con: {concentrations[i]} {target.ele[j].A / c.C_U}")
            i += 1

        sum_mass /= sum_concentration
        layer.natoms = i
        sum_amount_of_substance = density / sum_mass  # Originally sumN
        layer.Ntot = sum_amount_of_substance
        logging.info(f"sumM: {sum_mass / c.C_U}, sumN: {sum_amount_of_substance} {s.SYM_1_M3}")
        for i in range(layer.natoms):
            natoms = layer.atom[i]
            concentrations[i] /= sum_concentration
            layer.N[i] = concentrations[i] * sum_amount_of_substance
            logging.info(
                f"{target.ele[natoms].A / c.C_U} {target.ele[natoms].Z} {layer.N[i]} {s.SYM_1_M3}")
        nlayer += 1
    target.nlayers = nlayer

    target.minN = min(target.layer[:nlayer], key=lambda l: l.Ntot).Ntot

    logging.info(f"Number of layers: {target.nlayers}, minN: {target.minN}")
    logging.info(f"Number of atoms: {target.natoms}")

    fp.close()
