from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from numba_mcerd import config


MASSES_FILE = f"{config.PROJECT_ROOT}/data/constants/masses.dat"
ABUNDANCES_FILE = f"{config.PROJECT_ROOT}/data/constants/abundances.dat"


class JibalError(Exception):
    """Error in Jibal"""


@dataclass
class JibalIsotope:
    """Single isotope"""
    name: str = None
    N: int = 0
    Z: int = 0
    A: int = 0  # A = Z + N
    mass: float = 0.0
    abundance: float = 0.0


# noinspection PyPep8Naming
@dataclass
class JibalElement:
    """Element and its isotopes"""
    name: str = None
    Z: int = 0
    # n_isotopes: int = 0  # len(isotopes)
    # isotopes: List[JibalIsotope] = None  # original idea
    # concs: List[float] = None  # original idea
    isotopes: Dict[int, JibalIsotope] = None
    concs: Dict[int, float] = None  # Concentrations
    avg_mass: float = 0.0

    def __post_init__(self):
        if self.isotopes is None:
            self.isotopes = {}
        if self.concs is None:
            self.concs = {}

    def get_isotope_by_neutron_number(self, N: int) -> Optional[JibalIsotope]:
        """Get isotope by N (neutron number)"""
        return self.isotopes.get(N, None)

    def get_isotope_by_mass_number(self, A: int) -> Optional[JibalIsotope]:
        """Get isotope by A (mass number)"""
        N = A - self.Z
        return self.get_isotope_by_neutron_number(N)

    def update_avg_mass(self) -> None:
        """Calculate average mass weighted by abundance"""
        self.avg_mass = sum(isotope.mass * isotope.abundance for isotope in self.isotopes.values())

    def add_isotope(self, isotope: JibalIsotope) -> None:
        """Add an isotope to element"""
        self.isotopes[isotope.N] = isotope


# noinspection PyPep8Naming
@dataclass
class Jibal:
    # elements: List[JibalElement] = None  # Original idea
    elements: Dict[int, JibalElement] = None

    def __post_init__(self):
        if self.elements is None:
            self.elements = {}

    def initialize(self):
        self.load_elements()
        self.load_abundances()
        self.update_avg_masses()

    def get_element(self, Z: int) -> Optional[JibalElement]:
        """Get element with specific Z (proton number)"""
        return self.elements.get(Z, None)

    def get_element_by_name(self, name: str) -> Optional[JibalElement]:
        """Get element by name (e.g. 'Cl')"""
        for element in self.elements.values():
            if element.name == name:
                return element
        return None

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

    def add_isotope(self, isotope: JibalIsotope) -> None:
        """Add isotope to its parent element. The parent element is
        created first if it doesn't exist yet."""
        element = self.get_element(isotope.Z)
        if element is None:
            element = JibalElement(name=isotope.name, Z=isotope.Z)
            self.elements[isotope.Z] = element
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
            mass = float(mass)
            isotope = JibalIsotope(name=name, N=N, Z=Z, A=A, mass=mass)
            self.add_isotope(isotope)

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
        for element in self.elements.values():
            element.update_avg_mass()


# TODO: Add methods that copy the element and calculate concentrations
#       (JIBAL_NAT_ISOTOPES, JIBAL_ALL_ISOTOPES, default)


def main():
    jibal = Jibal()
    jibal.initialize()
    pass


if __name__ == '__main__':
    main()
