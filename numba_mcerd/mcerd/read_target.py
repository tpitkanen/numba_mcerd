import logging
from pathlib import Path

import numba_mcerd.mcerd.constants as c
import numba_mcerd.mcerd.objects as o
import numba_mcerd.mcerd.symbols as s
from numba_mcerd.mcerd import read_input


class ReadTargetError(Exception):
    """Error while reading target file"""


# TODO: Test this
def read_target_file(filename: str, g: o.Global, target: o.Target) -> None:
    """Read target configuration from file

    Args:
        filename: file to read configuration from
        g: global object
        target: target to set
    """
    con = [0.0] * c.MAXELEMENTS
    atom = [0.0] * c.MAXELEMENTS
    d = 0.0

    logging.info(f"Opening target file {filename}")
    fp = Path(filename).open("r")

    natoms = norigatom = target.natoms
    nlayer = noriglayer = target.nlayers  # TODO: Why is noriglayer unused?

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
        layer.type = c.TargetType.TARGET_FILM
        layer.gas = False

        if line[0].isalpha():
            raise NotImplementedError

        if layer.type == c.TargetType.TARGET_FILM:
            number, line = read_input.get_float(line)
            unit, line = read_input.get_unit_value(line, c.C_NM)
            logging.info(f"thickness {number * unit / c.C_NM}")
            layer.dlow = d
            layer.dhigh = d + number * unit
            d = layer.dhigh

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
        sumM = 0.0  # Mass
        sumcon = 0.0  # Concentration
        while line := fp.readline().strip():
            atom[i], line = read_input.get_float(line)
            con[i], line = read_input.get_float(line)
            # TODO: This seems to be just rounding, but why is norigatom added?
            j = int(atom[i] + norigatom + 0.5)
            layer.atom[i] = j
            sumM += target.ele[j].A * con[i]
            sumcon += con[i]
            logging.info(f"Atom: {j} {atom[i]}, con: {con[i]} {target.ele[j].A / c.C_U}")
            i += 1

        sumM /= sumcon
        layer.natoms = i
        sumN = density / sumM
        layer.Ntot = sumN
        logging.info(f"sumM: {sumM / c.C_U}, sumN: {sumN} {s.SYM_1_M3}")
        for i in range(layer.natoms):
            natoms = layer.atom[i]
            con[i] /= sumcon
            layer.N[i] = con[i] * sumN
            logging.info(
                f"{target.ele[natoms].A / c.C_U} {target.ele[natoms].Z} {layer.N[i]} {s.SYM_1_M3}")
        nlayer += 1
    target.nlayers = nlayer

    target.minN = min(target.layer[:nlayer], key=lambda l: l.Ntot).Ntot

    logging.info(f"Number of layers: {target.nlayers}, minN: {target.minN}")
    logging.info(f"Number of atoms: {target.natoms}")

    fp.close()
