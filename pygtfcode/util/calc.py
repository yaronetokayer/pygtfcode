import numpy as np
import math
from numba import njit, void, float64, types

@njit(types.Tuple((float64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_core_r_m_v2(r, rmid, rho, v2, m):
    """
    Computes core radius, core mass, and core v2.

    Core radius is defined as r_c such that rho(r_c) = 0.5 * rho_0,
    where rho_0 is the central density.

    r_c is estimated by log-log interpolation in rho(r).
    m_c is estimated by taking the shell that contains r_c and using
    a constant-density-within-shell volume fraction.
    v2_c is estimated as a shell-mass-weighted average of v2 inside r_c.

    Arguments
    ---------
    r : ndarray, shape (N+1,)
        Shell edge radii.
    rmid : ndarray, shape (N,)
        Midpoint radii.
    rho : ndarray, shape (N,)
        Shell densities.
    v2 : ndarray, shape (N,)
        Shell square of 1D velocity dispersion.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.

    Returns
    -------
    r_c : float
        Core radius
    m_c : float
        Core mass
    v2_c : float
        Core v2
    """
    N = rmid.shape[0]

    rho0        = rho[0]
    halfrho0    = 0.5 * rho0
    loghalfrho0 = math.log(halfrho0)

    # Numerator for mass-weighted v2 average
    numv_full = 0.0

    log_r_prev      = math.log(rmid[0])
    log_rho_prev    = math.log(rho0)

    for j in range(1, N):
        rho_cur = rho[j]

        # Find the first shell outside r_c
        if rho_cur <= halfrho0:
            # Log-log interpolation for rc
            log_r_cur   = math.log(rmid[j])
            log_rho_cur = math.log(rho_cur)

            logslope    = (log_rho_cur - log_rho_prev) / (log_r_cur - log_r_prev)
            log_rc      = log_r_prev + (loghalfrho0 - log_rho_prev) / logslope
            r_c         = math.exp(log_rc)

            # Accumulate m and v2 until r_c
            if r_c <= r[j]:
                k = j - 1
                numv = numv_full
            else:
                k = j
                dm_prev = m[j] - m[j - 1]
                numv = numv_full + dm_prev * v2[j - 1]

            rk0 = r[k]
            rk1 = r[k + 1]
            mk0 = m[k]
            mk1 = m[k + 1]

            # Find m_c assuming constant densities
            if r_c <= rk0:
                frac = 0.0
            elif r_c >= rk1:
                frac = 1.0
            else:
                rc3 = r_c * r_c * r_c
                rk03 = rk0 * rk0 * rk0
                rk13 = rk1 * rk1 * rk1
                frac = (rc3 - rk03) / (rk13 - rk03)

            m_c = mk0 + frac * (mk1 - mk0)

            # Complete v2_c computation with last partial shell
            dm_partial = m_c - mk0
            numv += dm_partial * v2[k]

            if m_c > 0.0:
                v2_c = numv / m_c
            else:
                v2_c = v2[0]

            return r_c, m_c, v2_c

        # No crossing yet; shell j-1 is fully inside for future crossings
        dm_prev = m[j] - m[j - 1]
        numv_full += dm_prev * v2[j - 1]

        log_r_prev      = math.log(rmid[j])
        log_rho_prev    = math.log(rho_cur)

    # If no crossing is found, return outermost values.
    r_c = rmid[N - 1]
    m_c = m[N]

    dm_last = m[N] - m[N - 1]
    numv_full += dm_last * v2[N - 1]

    if m_c > 0.0:
        v2_c = numv_full / m_c
    else:
        v2_c = v2[0]

    return r_c, m_c, v2_c

@njit(types.Tuple((float64, float64))(float64[:], float64[:], float64[:], float64), fastmath=True, cache=True)
def calc_smfp_r_m(r, rho, m, sigma_m):
    """
    Computes SMFP radius and SMFP mass.

    SMFP radius is defined as r_smfp such that

        int_0^{r_smfp} rho(r) dr = 1 / sigma_m

    This is where the optical depth equals 1.
    The integral is estimated by simple Riemann sums, assuming rho is
    constant within each shell. If the threshold is crossed within a
    shell, r_smfp is found by linear interpolation in radius within that
    shell.

    m_smfp is estimated by taking the shell that contains r_smfp and using
    a constant-density-within-shell volume fraction.

    Arguments
    ---------
    r : ndarray, shape (N+1,)
        Shell edge radii.
    rho : ndarray, shape (N,)
        Shell densities.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.
    sigma_m : float
        Cross section per unit mass.

    Returns
    -------
    r_smfp : float
        SMFP radius
    m_smfp : float
        Enclosed mass at r_smfp
    """
    N = rho.shape[0]

    if sigma_m <= 0.0:
        return 0.0, 0.0

    tau_target = 1.0 / sigma_m
    tau_prev = 0.0

    r1 = r[0]
    for j in range(N):
        r0 = r1
        r1 = r[j + 1]
        dr = r1 - r0

        tau_cur = tau_prev + rho[j] * dr

        if tau_cur >= tau_target:
            if tau_cur > tau_prev:
                frac_r = (tau_target - tau_prev) / (tau_cur - tau_prev)
            else:
                frac_r = 0.0

            r_smfp = r0 + frac_r * dr

            m0 = m[j]
            m1 = m[j + 1]

            if r_smfp <= r0:
                frac_m = 0.0
            else:
                r_smfp3 = r_smfp * r_smfp * r_smfp
                r03 = r0 * r0 * r0
                r13 = r1 * r1 * r1
                frac_m = (r_smfp3 - r03) / (r13 - r03)

            m_smfp = m0 + frac_m * (m1 - m0)

            return r_smfp, m_smfp

        tau_prev = tau_cur

    return 0.0, 0.0

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