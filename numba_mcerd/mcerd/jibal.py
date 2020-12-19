from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from numba_mcerd import config


MASSES_FILE = f"{config.PROJECT_ROOT}/data/constants/masses.dat"
ABUNDANCES_FILE = f"{config.PROJECT_ROOT}/data/constants/abundances.dat"


@dataclass
class JibalIsotope:
    """Single isotope"""
    name: str = None
    N: int = 0
    Z: int = 0
    A: int = 0  # A = Z + N
    mass: float = 0.0
    abundance: float = 0.0


@dataclass
class JibalElement:
    """Element and its isotopes"""
    name: str = None
    Z: int = 0
    # n_isotopes: int = 0  # len(isotopes)
    # isotopes: List[JibalIsotope] = None  # original idea
    isotopes: Dict[int, JibalIsotope] = None
    # concs: List[float] = None  # TODO: what is this?
    avg_mass: float = 0.0  # TODO:

    def __post_init__(self):
        if self.isotopes is None:
            self.isotopes = {}

    def get_isotope(self, N: int) -> Optional[JibalIsotope]:
        """Get isotope by N (neutron number)"""
        return self.isotopes.get(N, None)

    def update_avg_mass(self) -> None:
        """Calculate average mass weighted by abundance"""
        self.avg_mass = sum(isotope.mass * isotope.abundance for isotope in self.isotopes.values())

    def add_isotope(self, isotope: JibalIsotope) -> None:
        """Add an isotope to element"""
        self.isotopes[isotope.N] = isotope


@dataclass
class Jibal:
    # elements: List[JibalElement] = None  # Original idea
    elements: Dict[int, JibalElement] = None

    def __post_init__(self):
        if self.elements is None:
            self.elements = {}

    def initialize(self):
        self.load_elements()
        # self.load_abundances()
        # self.update_avg_masses()

    def get_element(self, Z: int) -> Optional[JibalElement]:
        """Get element with specific Z (proton number)"""
        return self.elements.get(Z, None)

    def get_isotope(self, Z, N) -> Optional[JibalIsotope]:
        """Get isotope with specific Z and N (proton number and neutron number)"""
        element = self.get_element(Z)
        if element is None:
            return element
        return element.get_isotope(N)

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
        raise NotImplementedError

    def update_avg_masses(self) -> None:
        """Updates avg_mass for each element. Elements and abundances must be loaded."""
        for element in self.elements.values():
            element.update_avg_mass()


def main():
    jibal = Jibal()
    jibal.initialize()


if __name__ == '__main__':
    main()
