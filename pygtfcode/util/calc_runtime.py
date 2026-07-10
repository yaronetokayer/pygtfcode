import numpy as np
from numba import njit, void, float64

@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def low_kn_boost(minkn, kn_threshold, boost, width):
    """
    Smooth multiplicative boost to eps_du in the low-Knudsen regime.

    Returns ~1 when minkn >> kn_threshold,
    ~boost when minkn << kn_threshold.
    """
    x = np.log10(minkn / kn_threshold)
    S = 1.0 / (1.0 + np.exp(x / width))

    return 1.0 + (boost - 1.0) * S
 
@njit(void(float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_ltemp(ltemp, v2, rmid):
    """
    ltemp = v2 / |dv2/dr|
    In-place update of ltemp.
    dv2dr via finite differences on possibly nonuniform rmid.
    """
    n = v2.size

    if n == 0:
        return

    if n == 1:
        ltemp[0] = np.nan
        return

    ltemp[0] = np.nan
    ltemp[1] = np.nan

    for i in range(2, n - 1):
        ltemp[i] = v2[i] * np.abs(rmid[i + 1] - rmid[i - 1]) / np.abs(v2[i + 1] - v2[i - 1])

    ltemp[n - 1] = v2[n - 1] * np.abs(rmid[n - 1] - rmid[n - 2]) / np.abs(v2[n - 1] - v2[n - 2])
