import numpy as np
# from scipy.linalg import solve_banded
from numba import njit, float64, types

def revirialize(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
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
        Total enclosed mass at each radial grid point, including baryons/perturbers.

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
    a, b, c, y = build_tridiag_system(r, rho, p, m_tot) # For Frank method
    # ab, y = build_tridiag_system(r, rho, p, m) # For scipy method
    # x = solve_banded((1, 1), ab, y)
    x = solve_tridiagonal_frank(a, b, c, y)

    # Update arrays accordingly
    r_new, p_new, rho_new, v2_new = _update_r_p_rho_v2(r, x, p, rho)

    # Check for unphysical negative velocity dispersion (allow tiny round-off)
    v2_max = float(np.max(v2_new)) if v2_new.size else 0.0
    rel_eps = 64.0 * np.finfo(np.float64).eps  # ~1e-14 safety margin
    thresh = -rel_eps * max(1.0, v2_max)
    if np.any(v2_new < thresh):
        return None
    # Clamp tiny negatives to zero (purely cosmetic / avoids nan in later sqrt)
    if np.any(v2_new < 0.0):
        v2_new = np.maximum(v2_new, 0.0)      
    
    dr_max_new = float(np.max(np.abs(x)))

    return r_new, rho_new, p_new, v2_new, dr_max_new

@njit(float64[:](float64[:]), cache=True, fastmath=True)
def compute_mass(m) -> np.ndarray:
    """
    Placeholder funcion to compute mass used in build_tridiag_system.
    Accounts for baryons, perturbers, etc. in future implementations.

    Arguments
    ---------
    m : ndarray
        Enclosed fluid mass at each radial grid point.

    Returns
    -------
    ndarray
        Total mass for hydrostatis equilibrium calculations.
    """

    return m

@njit(types.Tuple((float64[:], float64[:], float64[:], float64[:]))
      (float64[:], float64[:], float64[:], float64[:]),
      cache=True, fastmath=True)
def _update_r_p_rho_v2(r, x, p, rho):
    """
    Updates r, and then finds p, rho, and v2 based on exact volume ratios.
    Ensured positivity and stability.

    r: edge radii, shape (N+1,)
    x: interior stretch, x_j = dr_j / r_j for j=1..N-1, shape (N-1,)
    p, rho: shell-centered arrays, shape (N,)
    """
    r_new = r.copy()
    r_new[1:-1] *= (1.0 + x)  # inner/outer edges fixed

    V_old = r[1:]**3 - r[:-1]**3
    V_new = r_new[1:]**3 - r_new[:-1]**3

    # guard against underflow
    tiny = np.finfo(np.float64).tiny
    V_new = np.maximum(V_new, tiny)

    ratio = V_old / V_new  # = V / V'  (exact)
    gamma = 5.0 / 3.0

    rho_new = rho * ratio
    p_new   = p * ratio**gamma
    v2_new  = p_new / rho_new
    return r_new, p_new, rho_new, v2_new

# @njit(types.Tuple((float64[:], float64[:], float64[:], float64[:])) 
#       (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
# def _update_r_p_rho_v2(r, x, p, rho) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
#     r_new = r.copy()
#     r_new[1:-1] *= (1.0 + x)
#     # r3c = r[1:]**3 / (r[1:]**3 - r_new[:-1]**3) # Mistaken version in fortran code
#     # r3c = r[1:]**3 / (r[1:]**3 - r[:-1]**3) # Correct version to linear order
#     r3c = r_new[1:]**3 / (r_new[1:]**3 - r_new[:-1]**3) # Incorrect
    
#     dV_over_V = np.empty(r3c.shape, dtype=np.float64)
#     dV_over_V[0] = 3.0 * r3c[0] * x[0]
#     dV_over_V[1:-1] = 3.0 * (r3c[1:-1] * x[1:] - (r3c[1:-1] - 1.0) * x[:-1])
#     dV_over_V[-1] = -3.0 * (r3c[-1] - 1.0) * x[-1]
    
#     p_new = p * (1.0 - (5.0 / 3.0) * dV_over_V)
#     rho_new = rho * (1.0 - dV_over_V)
#     v2_new = p_new / rho_new
    
#     return r_new, p_new, rho_new, v2_new

# @njit(types.Tuple((float64[:], float64[:], float64[:], float64[:]))
#       (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
# def _update_r_p_rho_v2(r, x, p, rho) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     # Old method from Frank's (potentially mistaken) implementation
#     # r_new = r.copy()
#     # r_new[1:-1] *= (1.0 + x)
#     # r3c = r[1:]**3 / (r[1:]**3 - r_new[:-1]**3)

#     rL = r[:-1]
#     rR = r[1:]
#     rR3 = rR**3
#     r3c = rR3 / (rR3 - rL**3)
#     r3c_minus1 = r3c - 1.0

#     dV_over_V = np.empty(r3c.shape, dtype=np.float64)
#     dV_over_V[0]    = 3.0 * r3c[0] * x[0]
#     dV_over_V[1:-1] = 3.0 * (r3c[1:-1] * x[1:] - r3c_minus1[1:-1] * x[:-1])
#     dV_over_V[-1]   = -3.0 * r3c_minus1[-1] * x[-1]

#     p_new   = p   * (1.0 - (5.0 / 3.0) * dV_over_V)
#     rho_new = rho * (1.0 - dV_over_V)
#     v2_new  = p_new / rho_new

#     # build r_new without copying r
#     r_new = np.empty(r.shape, dtype=np.float64)
#     r_new[0]  = r[0]
#     r_new[-1] = r[-1]
#     r_new[1:-1] = r[1:-1] * (1.0 + x)

#     return r_new, p_new, rho_new, v2_new

# @njit(
#     types.Tuple((float64[:, :], float64[:]))   # returns (ab, y)
#     (float64[:], float64[:], float64[:], float64[:]),  # args: r, rho, p, m
#     cache=True
# )
@njit(types.Tuple((float64[:], float64[:], float64[:], float64[:]))
      (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray]:
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
    m_tot : ndarray
        Total enclosed mass at each radial grid point, including baryons/perturbers (length = n + 1)

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

    # Central differences
    dr = rR - rL
    inv_dr = 1.0 / dr

    # Pressure gradient and density sum for difference equations
    dP   = p[1:] - p[:-1]                           # Pressure difference
    drho = rho[1:] + rho[:-1]

    # floors to avoid divide-by-zero/inf
    tiny = np.finfo(np.float64).tiny
    dP   = np.where(np.abs(dP)   < tiny, np.copysign(tiny, dP),   dP)
    drho = np.where(drho         < tiny, tiny,                  drho)

    # Geometric volume factors
    rR3 = rR**3
    rC3 = rC**3
    rL3 = rL**3
    r3a = rR3 / (rR3 - rC3)
    r3c = rC3 / (rC3 - rL3)
    r3b = r3a - 1.0
    r3d = r3c - 1.0

    q1 = rR * inv_dr
    q2 = q1 - 1.0

    dd = -(4.0 / m_tot[1:-1]) * ( (rC * rC) * inv_dr ) * (dP / drho)

    c1 = 5.0 * dd * (p[1:] / dP) - 3.0 * (rho[1:] / drho)
    c2 = 5.0 * dd * (p[:-1] / dP) + 3.0 * (rho[:-1] / drho)

    y = dd - 1.0

    a = r3d * c2 - q2                               # Subdiagonal
    b = -2.0 - r3b * c1 - r3c * c2                  # Main diagonal, except first element
    c = r3a * c1 + q1                               # Superdiagonal

    # Banded matrix format for solve_banded
    # ab = np.zeros((3, (r.size - 2)), dtype=np.float64)
    # ab[0, 1:] = (r3a * c1 + q1)[:-1]               # Superdiagonal
    # ab[1, :]  = -2.0 - r3b * c1 - r3c * c2         # Main diagonal
    # ab[2, :-1] = (r3d * c2 - q2)[1:]               # Subdiagonal

    return a, b, c, y
    # return ab, y

@njit(float64[:](float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def solve_tridiagonal_frank(a, b, c, y):
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.
    This is Frank's implementation from numerical recipes.
.
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
    n = b.size

    u   = np.empty(n, dtype=np.float64)
    gam = np.empty(n, dtype=np.float64)
    bet = b[0]
    u[0] = y[0] / bet

    for i in range(1, n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i] * gam[i]
        u[i] = ( y[i] - a[i] * u[i-1] ) / bet

    for i in range(n - 2, -1, -1):
        u[i] -= gam[i+1] * u[i+1]

    return u