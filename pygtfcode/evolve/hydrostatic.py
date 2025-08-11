import numpy as np
from scipy.linalg import solve_banded
from numba import njit

def revirialize(r, rho, p, m) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Re-virializes the system state by solving for radius adjustments and updating physical quantities.

    Arguments
    ---------
    r : ndarray
        Radial coordinates.
    rho : ndarray
        Density values.
    p : ndarray
        Pressure values.
    m : ndarray
        Enclosed mass values

    Returns
    -------
    tuple or None
        Tuple containing:
            - r_new : ndarray
                Updated radial coordinates.
            - rho_new : ndarray
                Updated density values.
            - p_new : ndarray
                Updated pressure values.
            - v2_new : ndarray
                Updated velocity dispersion.
            - dr_max_new : float
                Maximum absolute change in radius.
        Returns None if any updated velocity dispersion is unphysical (negative).

    Notes
    -----
    The function solves a tridiagonal system to compute radius corrections, then updates
    density, pressure, and velocity dispersion accordingly. If any velocity dispersion
    becomes negative, the function returns None.
    """

    # Solve for corrections to r
    a, b, c, y = build_tridiag_system(r, rho, p, m)
    # x = solve_banded((1, 1), ab, y)
    x = solve_tridiagonal_frank(a, b, c, y)

    # Update arrays accordingly
    r_new, p_new, rho_new, v2_new = _update_r_p_rho_v2(r, x, p, rho)

    # Check for unphysical negative velocity dispersion
    if np.any(v2_new < 0):
        # return None
        return r_new, p_new # FOR DEBUGGING
    
    dr_max_new = np.max(np.abs(x))

    return r_new, rho_new, p_new, v2_new, dr_max_new

@njit
def _update_r_p_rho_v2(r, x, p, rho) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_new = r.copy()
    r_new[1:-1] *= (1.0 + x)
    r3c = r[1:]**3 / (r[1:]**3 - r_new[:-1]**3)

    dV_over_V = np.empty_like(r3c)
    dV_over_V[0] = 3.0 * r3c[0] * x[0]
    dV_over_V[1:-1] = 3.0 * (r3c[1:-1] * x[1:] - (r3c[1:-1] - 1.0) * x[:-1])
    dV_over_V[-1] = -3.0 * (r3c[-1] - 1.0) * x[-1]

    p_new = p * (1.0 - (5.0 / 3.0) * dV_over_V)
    rho_new = rho * (1.0 - dV_over_V)
    v2_new = p_new / rho_new

    return r_new, p_new, rho_new, v2_new

@njit
def build_tridiag_system(r, rho, p, m) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the tridiagonal matrix system (A·X = Y) used in the revirialization step.

    Arguments
    ---------
    r : ndarray
        Radial grid points (length = n + 1)
    rho : ndarray
        Density at each radial grid point (length = n)
    p : ndarray
        Pressure at each radial grid point (length = n)
    m : ndarray
        Enclosed mass at each radial grid point (length = n + 1)

    Returns
    -------
    ab : ndarray
        Banded matrix (3, n-1) for use with solve_banded
    y : ndarray
        Right-hand side of the linear system (length = n-1)
    """
    rL = r[:-2]         # Left radial grid points
    rR = r[2:]          # Right radial grid points
    rC = r[1:-1]        # Central radial grid points

    # Pressure gradient and density sum for difference equations
    dP = p[1:] - p[:-1]                             # Pressure difference
    drho = np.maximum(rho[1:] + rho[:-1], 1e-20)    # Prevent divide-by-zero
    dr = rR - rL                                    # Central differences

    # Geometric volume factors
    r3a = rR**3 / (rR**3 - rC**3)
    r3c = rC**3 / (rC**3 - rL**3)
    r3b = r3a - 1.0
    r3d = r3c - 1.0

    q1 = rR / dr
    q2 = q1 - 1.0

    # Gravitational correction term
    # This is where extra mass would be added for baryons, perturber, etc
    mm = m[1:-1]
    dd = -(4.0 / mm) * (rC**2 / dr) * (dP / drho)

    c1 = 5.0 * dd * (p[1:] / dP) - 3.0 * (rho[1:] / drho)
    c2 = 5.0 * dd * (p[:-1] / dP) + 3.0 * (rho[:-1] / drho)

    y = dd - 1.0

    a = r3d * c2 - q2                               # Subdiagonal
    b = -2.0 - r3b * c1 - r3c * c2                  # Main diagonal, except first element
    c = r3a * c1 + q1                               # Superdiagonal

    # Banded matrix format for solve_banded
    # ab = np.zeros((3, len(rC)))
    # ab[0, 1:] = (r3a * c1 + q1)[:-1]               # Superdiagonal
    # ab[1, :]  = -2.0 - r3b * c1 - r3c * c2         # Main diagonal
    # ab[2, :-1] = (r3d * c2 - q2)[1:]               # Subdiagonal

    return a, b, c, y

@njit
def solve_tridiagonal_thomas(a, b, c, y):
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.

    Parameters
    ----------
    a : ndarray
        Subdiagonal (length n-1)
    b : ndarray
        Main diagonal (length n)
    c : ndarray
        Superdiagonal (length n-1)
    y : ndarray
        Right-hand side vector (length n)

    Returns
    -------
    x : ndarray
        Solution vector (length n)
    """
    n = len(b)
    cp = np.empty(n-1)
    dp = np.empty(n)

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = y[0] / b[0] # Frank's u
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (y[i] - a[i-1] * dp[i-1]) / denom
    dp[n-1] = (y[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])

    # Back substitution
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

@njit
def solve_tridiagonal_frank(a, b, c, y):
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.

    Parameters
    ----------
    a : ndarray
        Subdiagonal (length n-1)
    b : ndarray
        Main diagonal, except first element (length n-1)
    c : ndarray
        Superdiagonal (length n-1)
    y : ndarray
        Right-hand side vector (length n-1)

    Returns
    -------
    x : ndarray
        Solution vector (length n)
    """
    n = len(b)

    u = np.empty(n)
    gam = np.empty(n)
    bet = b[0]
    u[0] = y[0] / bet

    for i in range(1, n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i]*gam[i]
        u[i] = ( y[i] - a[i] * u[i-1] ) / bet

    for i in range(n - 2, -1, -1):
        u[i] -= gam[i+1] * u[i+1]

    return u