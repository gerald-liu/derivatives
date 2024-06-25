import numpy as np
from scipy.special import zeta

# constants
BETA = - zeta(0.5) / np.sqrt(2 * np.pi) # = 0.5826

YEAR_LEN = 365 # 1Y = 365D
DT = 1 / YEAR_LEN
