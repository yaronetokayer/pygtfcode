import numpy as np
from numba import njit, float64, types, void, int64
from pygtfcode.util.calc import solve_tridiagonal_thomas

STATUS_OK = 0
STATUS_SHELL_CROSSING = 1

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

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system(r, rho, p, m_tot, a, b, c, y):
    """
    Fill preallocated arrays a, b, c, y with the tridiagonal system
    for the interior fractional radius shifts.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii.
    rho : ndarray, shape (N,)
        Shell-centered densities.
    p : ndarray, shape (N,)
        Shell-centered pressures.
    m_tot : ndarray, shape (N+1,)
        Total enclosed mass at the same edge radii as `r`.
    a, b, c, y : ndarray, shape (N-2,)
        Preallocated output arrays to fill in place.

    Notes
    -----
    - The unknown vector x contains the interior fractional displacements x_j = Δr_j / r_j
      (excluding the fixed inner and outer edges), so the returned arrays all have length M-2.
    - The routine linearizes the hydrostatic update using finite differences and geometric
      volume factors. Small numerical floors are applied to pressure differences and density sums
      to prevent divide-by-zero or overflow. The outputs are arranged for direct use with the
      tridiagonal solver used elsewhere in this module.
    """
    tiny = 2.2250738585072014e-308  # np.finfo(np.float64).tiny

    n = r.shape[0] - 2  # number of interior unknowns

    for j in range(n):
        iL = j
        iC = j + 1
        iR = j + 2

        rL = r[iL]
        rC = r[iC]
        rR = r[iR]

        dr = rR - rL
        inv_dr = 1.0 / dr

        dP = p[j + 1] - p[j]
        drho = rho[j + 1] + rho[j]

        # floors to avoid divide-by-zero / inf
        if abs(dP) < tiny:
            if dP >= 0.0:
                dP = tiny
            else:
                dP = -tiny

        if drho < tiny:
            drho = tiny

        rL3 = rL * rL * rL
        rC3 = rC * rC * rC
        rR3 = rR * rR * rR

        r3a = rR3 / (rR3 - rC3)
        r3c = rC3 / (rC3 - rL3)
        r3b = r3a - 1.0
        r3d = r3c - 1.0

        q1 = rR * inv_dr
        q2 = q1 - 1.0

        dd = -(4.0 / m_tot[iC]) * ((rC * rC) * inv_dr) * (dP / drho)

        c1 = 5.0 * dd * (p[j + 1] / dP) - 3.0 * (rho[j + 1] / drho)
        c2 = 5.0 * dd * (p[j] / dP)     + 3.0 * (rho[j] / drho)

        y[j] = dd - 1.0
        a[j] = r3d * c2 - q2
        b[j] = -2.0 - r3b * c1 - r3c * c2
        c[j] = r3a * c1 + q1

    # # Enforce dp/dr = 0 for i=1
    # # Old code from out-of-place version, here for reference
    # den1 = rR3[0] - rC3[0]   # Δ(r^3)_1
    # den0 = rC3[0] - rL3[0]   # Δ(r^3)_0

    # a[0] = 0.0
    # b[0] = 5.0 * rC3[0] * ( p[1] / den1 + p[0] / den0 )
    # c[0] = -5.0 * p[1] * (rR3[0] / den1)   # note: rC3[1] == rR3[0]
    # y[0] = -(p[1] - p[0])

@njit(types.Tuple((float64[:], float64[:], float64[:], float64[:]))
      (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=False)
def build_tridiag_system_log(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the tridiagonal system A·x = y for interior radial corrections.
    These are the coefficients for the log form of the HE equation.

    Arguments
    ---------
    r : ndarray
        Radial edge coordinates, length = n + 1.
    rho : ndarray
        Shell-centered densities, length = n.
    p : ndarray
        Shell-centered pressures, length = n.
    m_tot : ndarray
        Total enclosed mass at edge points, length = n + 1.

    Returns
    -------
    a, b, c, y : ndarray
        Tuple of 1D arrays (each length = n-1) defining the tridiagonal system
        for the interior unknowns:
          - a: subdiagonal (A[i, i-1])
          - b: main diagonal (A[i, i])
          - c: superdiagonal (A[i, i+1])
          - y: right-hand side vector
    """
    # Geometric volume factors
    rL = r[:-2]         # Left radial grid points
    rR = r[2:]          # Right radial grid points
    rC = r[1:-1]        # Central radial grid points
    
    rC2 = rC**2
    rC3 = rC2 * rC
    rR3 = rR**3
    rL3 = rL**3

    rL3rL3 = rL3 / (rC3 - rL3)
    rR3rC3 = rR3 / (rR3 - rC3)
    rC3rC3 = rC3 / (rR3 - rC3)
    rC3rL3 = rC3 / (rC3 - rL3)
    rC2rC3 = rC2 / (rR3 - rC3)
    rC2rL3 = rC2 / (rC3 - rL3)
    
    lnr = np.empty_like(r)
    lnr[1:] = np.log(r[1:])             # Don't take ln0 - lnr[0] never used anyway
    lnr[0]  = lnr[1]                    # Arbitrary finite placeholder
    dlnr = 0.5 * ( lnr[2:] - lnr[:-2] ) # Central difference

    pL = p[:-1]
    pR = p[1:]
    rhoL = rho[:-1]
    rhoR = rho[1:]
    lnp = np.log(p)
    dlnp = lnp[1:] - lnp[:-1]           # Right-sided difference

    sr = rho[:-1] + rho[1:]
    sp = p[:-1] + p[1:]

    mr = m_tot[1:-1] / r[1:-1]

    # floors to avoid divide-by-zero/inf
    tiny = np.finfo(np.float64).tiny
    sp   = np.where(np.abs(sp)   < tiny, np.copysign(tiny, sp),   sp)
    dlnr   = np.where(np.abs(dlnr)   < tiny, np.copysign(tiny, dlnr),   dlnr)

    dpdr = 0.5 * dlnp / dlnr**2
    srsp = sr / sp

    # Terms in final expressions
    afac = 5.0 / dlnr + (mr / sp) * (5.0 * pL * srsp - 3.0 * rhoL)
    bfac1 = 5.0 / dlnr
    bfac2 = m_tot[1:-1] / sp
    bfac3 = 5.0 * pR  * srsp - 3.0 * rhoR
    bfac4 = 3.0 * rhoL - 5.0 * pL * srsp
    cfac = 5.0 / dlnr + (mr / sp) * (3.0 * rhoR - 5.0 * pR * srsp)
    dfac = 2.0 * dpdr * dlnr

    y = -mr * srsp - dfac

    a = dpdr - rL3rL3 * afac                                                                # Subdiagonal
    b = bfac1 * (rC3rC3 + rC3rL3) - mr * srsp - bfac2 * (rC2rC3 * bfac3 + rC2rL3 * bfac4)   # Main diagonal
    c = -dpdr - rR3rC3 * cfac                                                               # Superdiagonal

    # Enforce dp/dr = 0 for i=1
    a[0] = 0.0
    b[0] = 5.0 * ( rC3rC3[0] + rC3rL3[0] )
    c[0] = -5.0 * rR3rC3[0]
    y[0] = -dlnp[0]

    return a, b, c, y

@njit(void(float64[:], float64[:],  float64[:],  float64[:],  float64[:]), cache=True, fastmath=True)
def update_r_p_rho(r, x, p, rho, work):
    """
    Updates r, and then finds p, rho, and v2 based on exact volume ratios.
    Ensures positivity and stability.
    All updates are performed in place.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii. Updated in place.
    x : ndarray, shape (N-1,)
        Interior fractional stretches. x_j = dr_j / r_j for j=1..N-1, shape (N-1,)
    p : ndarray, shape (N,)
        Shell-centered pressures. Updated in place.
    rho : ndarray, shape (N,)
        Shell-centered densities. Updated in place.
    work : ndarray, shape (N,)
        Scratch array used to store old shell volumes.
    """
    tiny = 2.2250738585072014e-308  # np.finfo(np.float64).tiny
    gamma = 5.0 / 3.0
    n = p.shape[0]

    # Store old shell volumes
    for j in range(n):
        rL = r[j]
        rR = r[j + 1]
        work[j] = rR * rR * rR - rL * rL * rL

    # Update interior radii in place
    for j in range(n - 1):
        r[j + 1] *= (1.0 + x[j])

    # Update rho and p from volume ratios
    for j in range(n):
        rL = r[j]
        rR = r[j + 1]
        V_new = rR * rR * rR - rL * rL * rL

        if V_new < tiny:
            V_new = tiny

        ratio = work[j] / V_new
        rho[j] *= ratio
        p[j] *= ratio ** gamma

@njit(float64(float64[:], float64[:], float64[:], float64[:]), fastmath=True,cache=True)
def compute_he_resid_norm(r, rho, p, m):
    """
    Compute an (unscaled) norm of the HE residual
    """
    Np1 = r.shape[0]
    
    dp = p[1:] - p[:-1]
    srho = rho[1:] + rho[:-1]
    dr = r[2:] - r[:-2]
    rC = r[1:-1]

    res_vec = - (4.0 / m[1:-1]) * (rC**2 / dr) * (dp / srho) - 1.0

    return np.linalg.norm(res_vec)

@njit(types.Tuple((int64, float64))(
        float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64
    ), cache=True, fastmath=True,)
def revirialize(r, rho, p, m_tot, a, b, c, y, x, vol_old, Np1)  -> tuple[int, float, float]:
    """
    Re-virializes the system state by solving for radius adjustments and updating physical quantities.

    Arguments
    ---------
    r : ndarray
        Radial coordinates. Updated in-place.
    rho : ndarray
        Density values. Updated in-place.
    p : ndarray
        Pressure values. Updated in-place.
    m : ndarray
        Total enclosed mass at each radial grid point, including baryons/perturbers.
    a, b, c, y, x : ndarray (N-1,)
        Memory allocation for working arrays
    vol_old : ndarray (N,)
        Memory allocation for working array
    Np1 : float
        Length of radial grid

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.
    dr_max : float
        Global maximum |dr/r| across all species.

    Notes
    -----
    The function solves a tridiagonal system to compute radius corrections, then updates
    density, pressure, and velocity dispersion accordingly. If any velocity dispersion
    becomes negative, the function returns None.
    """

    # Solve for corrections to r
    build_tridiag_system(r, rho, p, m_tot, a, b, c, y)
    solve_tridiagonal_thomas(a, b, c, y, x)
    dr_max = float(np.max(np.abs(x)))

    # Update arrays accordingly
    update_r_p_rho(r, x, p, rho, vol_old)

    # Check for shell crossing
    for i in range(Np1 - 1):
        if r[i + 1] - r[i] <= 0.0:
            return STATUS_SHELL_CROSSING, dr_max
    
    return STATUS_OK, dr_max

@njit(types.Tuple((int64, float64, float64))(
        float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64
    ), cache=True, fastmath=True,)
def revirialize_w_he_resid(r, rho, p, m_tot, a, b, c, y, x, vol_old, Np1)  -> tuple[int, float, float]:
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
    a, b, c, y, x : ndarray (N-1,)
        Memory allocation for working arrays
    vol_old : ndarray (N,)
        Memory allocation for working array
    Np1 : float
        Length of radial grid

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.
    dr_max : float
        Global maximum |dr/r| across all species.
    he_res : float
        Norm of HE residual for updated profile.
        If shell crossing occurs, returns -1.0 as a sentinel.

    Notes
    -----
    The function solves a tridiagonal system to compute radius corrections, then updates
    density, pressure, and velocity dispersion accordingly. If any velocity dispersion
    becomes negative, the function returns None.
    """

    # Solve for corrections to r
    build_tridiag_system(r, rho, p, m_tot, a, b, c, y)
    solve_tridiagonal_thomas(a, b, c, y, x)
    dr_max = float(np.max(np.abs(x)))

    # Update arrays accordingly
    update_r_p_rho(r, x, p, rho, vol_old)

    # Check for shell crossing
    for i in range(Np1 - 1):
        if r[i + 1] - r[i] <= 0.0:
            return STATUS_SHELL_CROSSING, dr_max, -1.0

    he_res = compute_he_resid_norm(r, rho, p, m_tot)
    
    return STATUS_OK, dr_max, he_res

@njit(void(float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def compute_he_pressures(r, rho, p, m):
    """
    In-place hydrostatic-equilibrium pressure update for unaligned radial grids.

    This version does NOT compute residuals.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii per species.
    rho : ndarray, shape (N,)
        Shell densities per species.
    p : ndarray, shape (N,)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (N+1,)
        Enclosed-mass-like data used by interp_m_enc().
    """
    Np1 = r.shape[0]
    N = Np1 - 1
    quarter = 0.25

    # Backward sweep:
    # p[N-1] is treated as the outer boundary value and left unchanged
    for i in range(N - 2, -1, -1):
        rip1 = r[i + 1]
        p[i] = p[i + 1] + (
            (rho[i + 1] + rho[i]) *
            (r[i + 2] - r[i]) *
            m[i + 1] *
            (quarter / (rip1 * rip1))
        )

@njit(types.Tuple((float64, float64))(float64[:], float64[:], float64[:], float64[:]),fastmath=True, cache=True)
def compute_he_pressures_with_resid(r, rho, p, m):
    """
    In-place hydrostatic-equilibrium pressure update for unaligned radial grids.

    This version computes and returns the old and new HE residual norms.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii per species.
    rho : ndarray, shape (N,)
        Shell densities per species.
    p : ndarray, shape (N,)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (N+1,)
        Enclosed-mass data.

    Returns
    -------
    res_old : float
        HE residual of input arrays.
    res_new : float
        HE residual after in-place pressure update.
    """
    res_old = compute_he_resid_norm(r, rho, p, m)
    compute_he_pressures(r, rho, p, m)
    res_new = compute_he_resid_norm(r, rho, p, m)
    return res_old, res_new