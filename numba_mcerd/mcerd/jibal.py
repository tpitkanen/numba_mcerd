from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

from numba_mcerd import config
import numba_mcerd.mcerd.constants as c


MASSES_FILE = f"{config.CONSTANTS_ROOT}/masses.dat"
ABUNDANCES_FILE = f"{config.CONSTANTS_ROOT}/abundances.dat"
ABUNDANCE_THRESHOLD = 1e-6


class JibalError(Exception):
    """Error in Jibal"""


class JibalSelectIsotopes(IntEnum):
    """Which isotopes to select. Positive isotopes are not listed."""
    ALL = -1      # Originally JIBAL_ALL_ISOTOPES
    NATURAL = 0  # Originally JIBAL_NAT_ISOTOPES
    # Positive number is a specific isotope


@dataclass
class JibalIsotope:
    """Single isotope"""
    name: str = None
    N: int = 0  # Neutrons
    Z: int = 0  # Protons
    A: int = 0  # Mass number, A = Z + N
    mass: float = 0.0
    abundance: float = 0.0


# TODO: Methods that add isotopes or elements don't overwrite previous ones anymore,
#       make them overwrite again?


# noinspection PyPep8Naming
@dataclass
class JibalElement:
    """Element and its isotopes"""
    name: str = None
    Z: int = 0
    # n_isotopes: int = 0  # len(isotopes)
    isotopes: List[JibalIsotope] = None
    concs: List[float] = None
    avg_mass: float = 0.0

    def __post_init__(self):
        if self.isotopes is None:
            self.isotopes = []
        if self.concs is None:
            self.concs = []

    def get_isotope_by_neutron_number(self, N: int) -> Optional[JibalIsotope]:
        """Get isotope by N (neutron number)"""
        return next((iso for iso in self.isotopes if iso.N == N), None)

    def get_isotope_by_mass_number(self, A: int) -> Optional[JibalIsotope]:
        """Get isotope by A (mass number)"""
        N = A - self.Z
        return self.get_isotope_by_neutron_number(N)

    def add_isotope(self, isotope: JibalIsotope) -> None:
        """Add an isotope to element"""
        self.isotopes.append(isotope)
        self.concs.append(0.0)


# noinspection PyPep8Naming
@dataclass
class Jibal:
    elements: List[JibalElement] = None

    def __post_init__(self):
        if self.elements is None:
            self.elements = []

    def initialize(self):
        self.load_elements()
        self.load_abundances()
        self.update_avg_masses()

    def get_element(self, Z: int) -> Optional[JibalElement]:
        """Get element with specific Z (proton number)"""
        return next((ele for ele in self.elements if ele.Z == Z), None)

    def get_element_by_name(self, name: str) -> Optional[JibalElement]:
        """Get element by name (e.g. 'Cl')"""
        return next((ele for ele in self.elements if ele.name == name), None)

    def add_element(self, element: JibalElement) -> None:
        """Add element to elements"""
        self.elements.append(element)

    def get_isotope_by_neutron_number(self, Z, N) -> Optional[JibalIsotope]:
        """Get isotope with specific Z and N (proton number and neutron number)"""
        element = self.get_element(Z)
        if element is None:
            return None
        return element.get_isotope_by_neutron_number(N)

    def get_isotope_by_mass_number(self, Z, A) -> Optional[JibalIsotope]:
        """Get isotope with specific Z and A (proton number and mass number)"""
        N = A - Z
        return self.get_isotope_by_neutron_number(Z, N)

    def create_isotope(self, name: str, N: int, Z: int, A: int, mass: float) -> None:
        """Create and add isotope to its parent element. The parent element is
        created first if it doesn't exist yet."""
        element = self.get_element(Z)
        if element is None:
            element = JibalElement(name=name, Z=Z)
            self.add_element(element)
        isotope_name = str(A) + name  # e.g. 4He
        isotope = JibalIsotope(name=isotope_name, N=N, Z=Z, A=A, mass=mass)
        element.add_isotope(isotope)

    def load_elements(self, masses_file: str = MASSES_FILE) -> None:
        """Load element information from masses_file"""
        file = Path(masses_file)
        with file.open() as f:
            lines = f.readlines()

        for line in lines:
            i, name, N, Z, A, mass = line.split(maxsplit=5)
            i = int(i)  # Unused
            N = int(N)
            Z = int(Z)
            A = int(A)
            mass = float(mass) * c.C_U  # Probably not the best place to multiply
            self.create_isotope(name=name, N=N, Z=Z, A=A, mass=mass)

    def load_abundances(self, abundances_file: str = ABUNDANCES_FILE) -> None:
        """Load abundances for elements from abundances_file"""
        file = Path(abundances_file)
        with file.open() as f:
            lines = f.readlines()

        for line in lines:
            # The last 2 values per line in abundances.dat are skipped in Jibal
            Z, A, abundance, _, _ = line.split(maxsplit=4)
            Z = int(Z)
            A = int(A)
            N = A - Z
            abundance = float(abundance)
            isotope = self.get_isotope_by_neutron_number(Z, N)
            if isotope is None:
                raise JibalError(f"Tried to add abundance for missing isotope: '{Z=}, {N=}, {A=}'")
            isotope.abundance = abundance

    def update_avg_masses(self) -> None:
        """Updates avg_mass for each element. Elements and abundances must be loaded."""
        for element in self.elements:
            element.avg_mass = sum(iso.mass * iso.abundance for iso in element.isotopes)

    # This could be a method for specific elements
    # Originally jibal_element_copy
    @staticmethod
    def copy_normalized_element(element: JibalElement, A: int) -> JibalElement:
        """Create a copy of element and calculate relative concentrations for it

        Args:
            element: element to copy
            A: isotopes to copy. All isotopes if -1, all natural isotopes if 0,
                otherwise a single isotope with this mass number

        Returns:
            Element with concentrations
        """

        if element is None:
            raise JibalError("No element given")
        if A < -1:
            raise JibalError(f"Incorrect isotope include type: {A}")

        # Copy element and calculate concentrations
        copied_element = JibalElement(name=element.name, Z=element.Z)
        if A == JibalSelectIsotopes.ALL.value:
            copied_element.name = element.name
            copied_element.isotopes = element.isotopes
            copied_element.concs = [iso.abundance for iso in element.isotopes]
        elif A == JibalSelectIsotopes.NATURAL.value:
            copied_element.name = "nat" + copied_element.name
            natural_isotopes = [
                iso for iso in element.isotopes if iso.abundance >= ABUNDANCE_THRESHOLD]
            copied_element.isotopes = natural_isotopes
            copied_element.concs = [iso.abundance for iso in natural_isotopes]
        else:  # Specific isotope
            target_isotope = element.get_isotope_by_mass_number(A)
            copied_element.name = target_isotope.name
            copied_element.isotopes = [target_isotope]
            copied_element.concs = [1.0]

        # Isotope count check shouldn't be needed, but it exists in the original code

        # Normalize element
        conc_sum = sum(copied_element.concs)
        if conc_sum == 0.0:
            raise JibalError(f"No concentrations found for element '{copied_element.Z=}'")
        for i in range(len(copied_element.concs)):
            copied_element.concs[i] = copied_element.concs[i] / conc_sum
            copied_element.avg_mass += copied_element.concs[i] * copied_element.isotopes[i].mass

        return copied_element


def main():
    jibal = Jibal()
    jibal.initialize()
    pass


if __name__ == '__main__':
    main()
