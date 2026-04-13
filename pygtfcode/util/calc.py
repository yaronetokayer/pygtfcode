import numpy as np
from numba import njit, void, float64

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True)
def solve_tridiagonal_thomas(a, b, c, y, x):
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.
    This follows the Numerical Recipes convention:
        a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = y[i]

    For an N x N system, all coefficient arrays have length N.
    The value a[0] is unused, and c[N-1] is unused.

    Parameters
    ----------
    a : ndarray
        Subdiagonal coefficients.
    b : ndarray
        Main diagonal coefficients.
    c : ndarray
        Superdiagonal coefficients.
    y : ndarray
        Right-hand side vector.
    x : ndarray
        Output solution vector, updated in place.
    """
    n = b.size
    gam = np.empty(n, dtype=np.float64)
    gam[0] = 0.0

    bet = b[0]
    x[0] = y[0] / bet

    for i in range(1, n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i] * gam[i]
        x[i] = (y[i] - a[i] * x[i-1]) / bet

    for i in range(n - 2, -1, -1):
        x[i] -= gam[i+1] * x[i+1]