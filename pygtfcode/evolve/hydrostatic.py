import numpy as np
import math
from numba import njit, float64, types, void, int64
from pygtfcode.util.calc import solve_tridiagonal_thomas

STATUS_OK = 0
STATUS_SHELL_CROSSING = 1
_TINY64 = np.finfo(np.float64).tiny

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

@njit(types.void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=False)
def build_tridiag_system_log(r, rho, p, m_tot, a, b, c, y):
    """
    Fill preallocated arrays ``a``, ``b``, ``c``, and ``y`` with the
    tridiagonal system A·x = y for interior radial corrections in the
    logarithmic form of the HE equation.

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
    a : ndarray
        Preallocated output array of length n - 1 for the subdiagonal
        entries, ``a[i] = A[i, i-1]``.
    b : ndarray
        Preallocated output array of length n - 1 for the main diagonal
        entries, ``b[i] = A[i, i]``.
    c : ndarray
        Preallocated output array of length n - 1 for the superdiagonal
        entries, ``c[i] = A[i, i+1]``.
    y : ndarray
        Preallocated output array of length n - 1 for the right-hand side.

    Notes
    -----
    This function performs in-place updates only and returns nothing.

    The output arrays must already be allocated with length ``n - 1``,
    where ``n = len(rho) = len(p)``.
    """
    tiny = _TINY64
    n_out = p.shape[0] - 1

    # Nothing to do if there are no interior unknowns.
    if n_out <= 0:
        return

    # ------------------------------------------------------------------
    # Row 0 is *not* assembled using the general formula, because we
    # overwrite it with the special boundary condition anyway.
    #
    # The main loop below handles only i >= 1.
    # ------------------------------------------------------------------
    if n_out > 1:
        # Initialize the sliding window for the first interior row that
        # is actually assembled by the general formula, namely i = 1:
        #
        #   left  edge -> r[1]
        #   center edge -> r[2]
        #   right edge -> r[3]
        #
        # We also cache powers and logs that can be rolled forward from
        # one iteration to the next to reduce repeated work.
        rL = r[1]
        rC = r[2]
        rR = r[3]

        rL3 = rL * rL * rL
        rC2 = rC * rC
        rC3 = rC2 * rC
        rR3 = rR * rR * rR

        log_rL = math.log(rL)
        log_rC = math.log(rC)
        log_rR = math.log(rR)

        log_pL = math.log(p[1])
        log_pR = math.log(p[2])

        for i in range(1, n_out):
            # Geometry factors for the current 3-point stencil.
            denomL = rC3 - rL3
            denomR = rR3 - rC3
            inv_denomL = 1.0 / denomL
            inv_denomR = 1.0 / denomR

            rL3rL3 = rL3 * inv_denomL
            rR3rC3 = rR3 * inv_denomR
            rC3rC3 = rC3 * inv_denomR
            rC3rL3 = rC3 * inv_denomL
            rC2rC3 = rC2 * inv_denomR
            rC2rL3 = rC2 * inv_denomL

            # Local thermodynamic state.
            pL = p[i]
            pR = p[i + 1]
            rhoL = rho[i]
            rhoR = rho[i + 1]

            dlnp = log_pR - log_pL
            dlnr = 0.5 * (log_rR - log_rL)

            sp = pL + pR
            sr = rhoL + rhoR

            # Floors to avoid divide-by-zero/inf.
            if abs(sp) < tiny:
                sp = math.copysign(tiny, sp)
            if abs(dlnr) < tiny:
                dlnr = math.copysign(tiny, dlnr)

            inv_sp = 1.0 / sp
            inv_dlnr = 1.0 / dlnr
            inv_dlnr2 = inv_dlnr * inv_dlnr

            dpdr = 0.5 * dlnp * inv_dlnr2
            srsp = sr * inv_sp

            m_edge = m_tot[i + 1]
            mr = m_edge / rC
            m_over_sp = m_edge * inv_sp
            mr_over_sp = mr * inv_sp

            afac = 5.0 * inv_dlnr + mr_over_sp * (5.0 * pL * srsp - 3.0 * rhoL)
            cfac = 5.0 * inv_dlnr + mr_over_sp * (3.0 * rhoR - 5.0 * pR * srsp)

            bfac3 = 5.0 * pR * srsp - 3.0 * rhoR
            bfac4 = 3.0 * rhoL - 5.0 * pL * srsp

            a[i] = dpdr - rL3rL3 * afac
            b[i] = (
                5.0 * inv_dlnr * (rC3rC3 + rC3rL3)
                - mr * srsp
                - m_over_sp * (rC2rC3 * bfac3 + rC2rL3 * bfac4)
            )
            c[i] = -dpdr - rR3rC3 * cfac
            y[i] = -mr * srsp - dlnp * inv_dlnr

            # Advance the sliding window:
            #
            # Old: (rL, rC, rR) = (r[i],   r[i+1], r[i+2])
            # New: (rL, rC, rR) = (r[i+1], r[i+2], r[i+3])
            #
            # The same rolling update is used for log(r) and log(p),
            # so each new iteration computes only one new log(r) and
            # one new log(p).
            if i + 1 < n_out:
                rL = rC
                rC = rR
                rR = r[i + 3]

                rL3 = rC3
                rC2 = rC * rC
                rC3 = rC2 * rC
                rR3 = rR * rR * rR

                log_rL = log_rC
                log_rC = log_rR
                log_rR = math.log(rR)

                log_pL = log_pR
                log_pR = math.log(p[i + 2])

    # ------------------------------------------------------------------
    # Enforce dp/dr = 0 for i = 1 exactly as in the original code.
    #
    # Since row 0 is always replaced by this boundary condition, we only
    # assemble it here once and never build the discarded general row.
    # ------------------------------------------------------------------
    rL = r[0]
    rC = r[1]
    rR = r[2]

    rC2 = rC * rC
    rC3 = rC2 * rC
    rR3 = rR * rR * rR
    rL3 = rL * rL * rL

    denomL = rC3 - rL3
    denomR = rR3 - rC3
    inv_denomL = 1.0 / denomL
    inv_denomR = 1.0 / denomR

    rR3rC3 = rR3 * inv_denomR
    rC3rC3 = rC3 * inv_denomR
    rC3rL3 = rC3 * inv_denomL

    a[0] = 0.0
    b[0] = 5.0 * (rC3rC3 + rC3rL3)
    c[0] = -5.0 * rR3rC3
    y[0] = -(math.log(p[1]) - math.log(p[0]))

@njit(types.void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=False)
def build_tridiag_system_log_OLD(r, rho, p, m_tot, a, b, c, y):
    """
    Fill preallocated arrays ``a``, ``b``, ``c``, and ``y`` with the
    tridiagonal system A·x = y for interior radial corrections in the
    logarithmic form of the HE equation.

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
    a : ndarray
        Preallocated output array of length n - 1 for the subdiagonal
        entries, ``a[i] = A[i, i-1]``.
    b : ndarray
        Preallocated output array of length n - 1 for the main diagonal
        entries, ``b[i] = A[i, i]``.
    c : ndarray
        Preallocated output array of length n - 1 for the superdiagonal
        entries, ``c[i] = A[i, i+1]``.
    y : ndarray
        Preallocated output array of length n - 1 for the right-hand side.

    Notes
    -----
    This function performs in-place updates only and returns nothing.

    The output arrays must already be allocated with length ``n - 1``,
    where ``n = len(rho) = len(p)``.
    """
    tiny = np.finfo(np.float64).tiny
    n_out = p.shape[0] - 1

    for i in range(n_out):
        # Local geometry
        rL = r[i]
        rC = r[i + 1]
        rR = r[i + 2]

        rC2 = rC * rC
        rC3 = rC2 * rC
        rR3 = rR * rR * rR
        rL3 = rL * rL * rL

        denomL = rC3 - rL3
        denomR = rR3 - rC3

        rL3rL3 = rL3 / denomL
        rR3rC3 = rR3 / denomR
        rC3rC3 = rC3 / denomR
        rC3rL3 = rC3 / denomL
        rC2rC3 = rC2 / denomR
        rC2rL3 = rC2 / denomL

        # Local thermodynamics
        pL = p[i]
        pR = p[i + 1]
        rhoL = rho[i]
        rhoR = rho[i + 1]

        dlnp = math.log(pR) - math.log(pL)

        # Match original handling of lnr[0] = lnr[1]
        if i == 0:
            dlnr = 0.5 * (math.log(rR) - math.log(rC))
        else:
            dlnr = 0.5 * (math.log(rR) - math.log(rL))

        sp = pL + pR
        sr = rhoL + rhoR

        # Floors to avoid divide-by-zero/inf
        if abs(sp) < tiny:
            sp = math.copysign(tiny, sp)
        if abs(dlnr) < tiny:
            dlnr = math.copysign(tiny, dlnr)

        inv_dlnr = 1.0 / dlnr
        dpdr = 0.5 * dlnp * inv_dlnr * inv_dlnr
        srsp = sr / sp
        mr = m_tot[i + 1] / rC

        afac = 5.0 * inv_dlnr + (mr / sp) * (5.0 * pL * srsp - 3.0 * rhoL)
        cfac = 5.0 * inv_dlnr + (mr / sp) * (3.0 * rhoR - 5.0 * pR * srsp)

        bfac3 = 5.0 * pR * srsp - 3.0 * rhoR
        bfac4 = 3.0 * rhoL - 5.0 * pL * srsp

        a[i] = dpdr - rL3rL3 * afac
        b[i] = (
            5.0 * inv_dlnr * (rC3rC3 + rC3rL3)
            - mr * srsp
            - (m_tot[i + 1] / sp) * (rC2rC3 * bfac3 + rC2rL3 * bfac4)
        )
        c[i] = -dpdr - rR3rC3 * cfac
        y[i] = -mr * srsp - dlnp * inv_dlnr

    # Enforce dp/dr = 0 for i = 1
    rL = r[0]
    rC = r[1]
    rR = r[2]

    rC2 = rC * rC
    rC3 = rC2 * rC
    rR3 = rR * rR * rR
    rL3 = rL * rL * rL

    denomL = rC3 - rL3
    denomR = rR3 - rC3

    rR3rC3 = rR3 / denomR
    rC3rC3 = rC3 / denomR
    rC3rL3 = rC3 / denomL

    a[0] = 0.0
    b[0] = 5.0 * (rC3rC3 + rC3rL3)
    c[0] = -5.0 * rR3rC3
    y[0] = -(math.log(p[1]) - math.log(p[0]))

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

@njit(types.Tuple((int64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64), cache=True, fastmath=True,)
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
    build_tridiag_system_log(r, rho, p, m_tot, a, b, c, y)
    solve_tridiagonal_thomas(a, b, c, y, x)
    dr_max = float(np.max(np.abs(x)))

    # Update arrays accordingly
    update_r_p_rho(r, x, p, rho, vol_old)

    # Check for shell crossing
    for i in range(Np1 - 1):
        if r[i + 1] - r[i] <= 0.0:
            return STATUS_SHELL_CROSSING, dr_max
    
    return STATUS_OK, dr_max

@njit(types.Tuple((int64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64), cache=True, fastmath=True,)
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
    build_tridiag_system_log(r, rho, p, m_tot, a, b, c, y)
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