"""Constants from general.h"""

################################################################################
# Lines 16 .. 143 (roughly)
################################################################################

# SMALLNUM = 1e-10


# Constants which may effect the physics of simulation:
MAXANGLE = 177  # Minimum of maximum scattering angles


# Maximums for arrays etc.

EPSNUM = 50  # Number of energies in scattering angle table
YNUM = 100  # Number of impact factors in scattering angle table
EPSIMP = 200  # Number of points in maximum impact factor array
N_RAND_DIST = 500  # Number of points in random distribution
N_INPUT = 20000  # Maximum number of lines in input file
LINE = 1024  # Maximum length of line in input files
PATH = ""  # Path for input files
NFILE = 512  # Maximum name length of the input file
MAXSTO = 500  # Maximum number of values in stopping power array
MAXELOSS = 0.01  # Maximum relative change in electonic energy loss
MAXENEFF = 2000  # Maximum number of point in the energy eff. curve
NSURFDATA = 520  # Maximum width of the surface topography image

NRECDIST = 500

MAXLAYERS = 50  # Number of different target layers
MAXATOMS = 20  # Number of different atoms in the target layer
MAXELEMENTS = 20  # Maximum total number of different atoms in the target
NANGDIST = 200  # Number of points in reaction angular distribution

MAXISOTOPES = 20


C_NM = 1.0e-9  # From rough_surface.c
SCALE_DEPTH = (10.0*C_NM)  # All scaling particles are recoiled below this

PRESIMU_LEVEL = 0.01  # Ratio of ions for defining recoil solid angle

# # Booleans
# TRUE = 1
# FALSE = 0


NSENE = 2000  # Maximum number of energies in the cross section table
NSANGLE = 200  # Maximum number of angle in the cross section table


C_KEV = 1.602176634e-16  # from jibal_units.h

TRACKPOINT_INTERVAL = (10.0*C_KEV)
TRACKPOINT_LIMIT_SMALL = (50.0*C_KEV)
TRACKPOINT_INTERVAL_SMALL = (1.0*C_KEV)

MAXSTOFILEPREFIXLEN = 200


################################################################################
# Lines 436 .. 443
################################################################################


MAXFOILS = 10  # Maximum number of foils in detector

################################################################################
# Jibal: jibal_units.h
# Lines  22 .. 86
################################################################################

C_US = 1.0e-6
C_NS = 1.0e-9
C_PS = 1.0e-12
C_CM2 = 1.0e-4
C_CM3 = 1.0e-6
C_CM = 1.0e-2
C_MM = 1.0e-3
C_UM = 1.0e-6
C_NM = 1.0e-9
C_ANGSTROM = 1.0e-10
C_G = 1.0e-3
C_UG = 1.0e-9
C_G_CM3 = 1.0e3
C_MSR = 1.0e-3

C_PI = 3.14159265358979323846264338327950288  # math.pi == C_PI
C_DEG = 2.0*C_PI/360.0  # degree
C_FWHM = 2.35482004503
C_PERCENT = 0.01
C_U = 1.66053906660e-27  # atomic mass unit, kg
C_C = 299792458.0  # speed of light, in m/s, exact.
C_C2 = C_C*C_C  # speed of light, squared
C_E = 1.602176634e-19  # elementary charge, in C, exact.
C_UC = 1.0e-6/C_E
C_EV = C_E  # electronvolt, in J
C_KEV = 1.0e3*C_EV  # keV
C_MEV = 1.0e6*C_EV  # MeV

C_H = 6.62607015e-34  # Planck constant, exact.
C_HBAR = C_H/(2*C_PI)
C_ALPHA = 0.0072973525693  # fine structure constant. Using CODATA 2018 recommended value.
C_MU0 = 2*C_ALPHA*C_H/(C_E*C_E*C_C)  # vacuum permeability
C_EPSILON0 = 1.0/(C_MU0*C_C2)  # vacuum permittivity
C_ME = 9.1093837015e-31  # electron mass, kg
C_BOHR_RADIUS = C_HBAR/(C_ALPHA*C_ME*C_C)  # Bohr radius

C_TFU = 1.0e19  # Thin film units, i.e. 1e15 at./cm2 is actually 1e19/m^2 in SI-units
C_EV_TFU = C_EV/C_TFU  # Units for stopping cross sections (eV/(1e15 at./cm2))

C_BARN = 1.0e-28
C_MB_SR = 1.0e-3*C_BARN  # millibarns/sr
