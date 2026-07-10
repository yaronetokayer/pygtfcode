import numpy as np
import math
from numba import njit, void, float64, types

GAMMA = 5.0 / 3.0

@njit(float64[:](float64[:], float64[:], types.int64), fastmath=True, cache=True)
def calc_balberg_zeta(m_c, v2_c, window): 
    """
    Computes zeta = dln(v2_c) / dln(m_c) + 1 using a local log-log fit.

    zeta is estimated from the local power-law relation between v2_c and m_c,
    so that locally v2_c is approximately proportional to

        m_c**(zeta - 1).

    At each point i, zeta[i] is computed by fitting

        ln(v2_c) = a + (zeta - 1) * ln(m_c)

    over a window of neighboring points.

    Arguments
    ---------
    m_c : ndarray, shape (N,)
        Core masses.
    v2_c : ndarray, shape (N,)
        Core square of 1D velocity dispersion.
    window : int
        Number of points used in each local fit. Must be odd.

    Returns
    -------
    zeta : ndarray, shape (N,)
        Local logarithmic slope dln(v2_c) / dln(m_c) + 1.
    """
    N = m_c.shape[0]

    zeta = np.empty(N, dtype=np.float64)

    # Precompute logs once.
    log_m = np.empty(N, dtype=np.float64)
    log_v2 = np.empty(N, dtype=np.float64)

    for i in range(N):
        log_m[i] = math.log(m_c[i])
        log_v2[i] = math.log(v2_c[i])

    half_window = window // 2

    # Minimum allowed local variance in log(m_c).
    # Since den = sum((x - xbar)**2), compare against count * min variance.
    min_logm_var = 1.0e-24

    for i in range(N):

        # Choose local fitting window centered on i.
        i0 = i - half_window
        i1 = i + half_window + 1

        # Shift the window back inside the valid index range near boundaries.
        if i0 < 0:
            i1 -= i0
            i0 = 0

        if i1 > N:
            i0 -= i1 - N
            i1 = N

        if i0 < 0:
            i0 = 0

        # First pass: compute mean log(m_c) and mean log(v2_c).
        xbar = 0.0
        ybar = 0.0
        count = 0

        for j in range(i0, i1):
            xbar += log_m[j]
            ybar += log_v2[j]
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            dx = log_m[j] - xbar
            dy = log_v2[j] - ybar

            num += dx * dy
            den += dx * dx

        if den > min_logm_var * count:
            zeta[i] = num / den + 1.0
        else:
            zeta[i] = 1.0

    return zeta
 
@njit(float64[:](float64[:], float64[:], types.int64), fastmath=True, cache=True)
def calc_dlnmc_dlnvc(m_c, v2_c, window): 
    """
    Computes zeta = dln(m_c) / dln(v_c) using a local log-log fit.

    zeta is estimated as the local power-law slope between m_c and v_c,
    so that locally m_c is approximately proportional to v_c**zeta.

    At each point i, zeta[i] is computed by fitting

        ln(m_c) = a + 0.5 * zeta * ln(v2_c) (the 0.5 factor is to account for the fact that v2_c is v_c squared)

    over a window of neighboring points.

    Arguments
    ---------
    m_c : ndarray, shape (N,)
        Core masses.
    v2_c : ndarray, shape (N,)
        Core square of 1D velocity dispersion.
    window : int
        Number of points used in each local fit.
        If even, the effective window size is window + 1.

    Returns
    -------
    zeta : ndarray, shape (N,)
        Local logarithmic slope dln(m_c) / dln(v_c).
    """
    N = m_c.shape[0]

    zeta = np.empty(N, dtype=np.float64)

    # Precompute logs once.
    log_m = np.empty(N, dtype=np.float64)
    log_v2 = np.empty(N, dtype=np.float64)

    for i in range(N):
        log_m[i] = math.log(m_c[i])
        log_v2[i] = math.log(v2_c[i])

    half_window = window // 2

    # Minimum allowed local variance in log(v2_c).
    # Since den = sum((x - xbar)**2), compare against count * min variance.
    min_logv2_var = 1.0e-24

    for i in range(N):

        # Choose local fitting window centered on i.
        i0 = i - half_window
        i1 = i + half_window + 1

        # Shift the window back inside the valid index range near boundaries.
        if i0 < 0:
            i1 -= i0
            i0 = 0

        if i1 > N:
            i0 -= i1 - N
            i1 = N

        if i0 < 0:
            i0 = 0

        # First pass: compute mean log(v2_c) and mean log(m_c).
        xbar = 0.0
        ybar = 0.0
        count = 0

        for j in range(i0, i1):
            xbar += log_v2[j]
            ybar += log_m[j]
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            dx = log_v2[j] - xbar
            dy = log_m[j] - ybar

            num += dx * dy
            den += dx * dx

        if den > min_logv2_var * count:
            zeta[i] = 2.0 * num / den
        else:
            zeta[i] = 0.0

    return zeta

@njit(float64[:](float64[:], float64[:], types.int64), fastmath=True, cache=True)
def calc_dlnrhoc_dlnvc(rho_c, v2_c, window): 
    """
    Computes zeta = dln(rho_c) / dln(v_c) using a local log-log fit.

    zeta is estimated as the local power-law slope between rho_c and v_c,
    so that locally rho_c is approximately proportional to v_c**zeta.

    At each point i, zeta[i] is computed by fitting

        ln(rho_c) = a + 0.5 * zeta * ln(v2_c)

    over a window of neighboring points.

    Arguments
    ---------
    rho_c : ndarray, shape (N,)
        Core densities.
    v2_c : ndarray, shape (N,)
        Core square of 1D velocity dispersion.
    window : int
        Number of points used in each local fit.
        If even, then the effective window size is window + 1.

    Returns
    -------
    zeta : ndarray, shape (N,)
        Local logarithmic slope dln(rho_c) / dln(v_c).
    """
    N = rho_c.shape[0]

    zeta = np.empty(N, dtype=np.float64)

    # Precompute logs once.
    log_rho = np.empty(N, dtype=np.float64)
    log_v2 = np.empty(N, dtype=np.float64)

    for i in range(N):
        log_rho[i] = math.log(rho_c[i])
        log_v2[i] = math.log(v2_c[i])

    half_window = window // 2

    # Minimum allowed local variance in log(v2_c).
    # Since den = sum((x - xbar)**2), compare against count * min variance.
    min_logv2_var = 1.0e-24

    for i in range(N):

        # Choose local fitting window centered on i.
        i0 = i - half_window
        i1 = i + half_window + 1

        # Shift the window back inside the valid index range near boundaries.
        if i0 < 0:
            i1 -= i0
            i0 = 0

        if i1 > N:
            i0 -= i1 - N
            i1 = N

        if i0 < 0:
            i0 = 0

        # First pass: compute mean log(v2_c) and mean log(rho_c).
        xbar = 0.0
        ybar = 0.0
        count = 0

        for j in range(i0, i1):
            xbar += log_v2[j]
            ybar += log_rho[j]
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            dx = log_v2[j] - xbar
            dy = log_rho[j] - ybar

            num += dx * dy
            den += dx * dx

        if den > min_logv2_var * count:
            zeta[i] = 2.0 * num / den
        else:
            zeta[i] = 0.0

    return zeta

@njit(float64[:](float64[:], float64[:], types.int64), fastmath=True, cache=True)
def calc_dlnmc_dlnrhoc(m_c, rho_c, window): 
    """
    Computes zeta = dln(m_c) / dln(rho_c) using a local log-log fit.

    zeta is estimated as the local power-law slope between m_c and rho_c,
    so that locally m_c is approximately proportional to rho_c**zeta.

    At each point i, zeta[i] is computed by fitting

        ln(m_c) = a + zeta * ln(rho_c)

    over a window of neighboring points.

    Arguments
    ---------
    m_c : ndarray, shape (N,)
        Core masses.
    rho_c : ndarray, shape (N,)
        Core densities.
    window : int
        Number of points used in each local fit.
        If even, then the effective window size is window + 1.

    Returns
    -------
    zeta : ndarray, shape (N,)
        Local logarithmic slope dln(m_c) / dln(rho_c).
    """
    N = m_c.shape[0]

    zeta = np.empty(N, dtype=np.float64)

    # Precompute logs once.
    log_m = np.empty(N, dtype=np.float64)
    log_rho = np.empty(N, dtype=np.float64)

    for i in range(N):
        log_m[i] = math.log(m_c[i])
        log_rho[i] = math.log(rho_c[i])

    half_window = window // 2

    # Minimum allowed local variance in log(rho_c).
    # Since den = sum((x - xbar)**2), compare against count * min variance.
    min_logrho_var = 1.0e-24

    for i in range(N):

        # Choose local fitting window centered on i.
        i0 = i - half_window
        i1 = i + half_window + 1

        # Shift the window back inside the valid index range near boundaries.
        if i0 < 0:
            i1 -= i0
            i0 = 0

        if i1 > N:
            i0 -= i1 - N
            i1 = N

        if i0 < 0:
            i0 = 0

        # First pass: compute mean log(rho_c) and mean log(m_c).
        xbar = 0.0
        ybar = 0.0
        count = 0

        for j in range(i0, i1):
            xbar += log_rho[j]
            ybar += log_m[j]
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            dx = log_rho[j] - xbar
            dy = log_m[j] - ybar

            num += dx * dy
            den += dx * dx

        if den > min_logrho_var * count:
            zeta[i] = num / den
        else:
            zeta[i] = 0.0

    return zeta

@njit(types.Tuple((float64[:], float64[:]))(float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_s_dsdr(v2, rho, rmid):
    """
    s = ln(v^3 / rho) = 1.5 ln(v2) - ln(rho)
    dsdr via finite differences on possibly nonuniform rmid.
    """
    n = v2.size
    s = np.empty(n, dtype=np.float64)
    dsdr = np.empty(n, dtype=np.float64)

    for i in range(n):
        s[i] = 1.5 * np.log(v2[i]) - np.log(rho[i])

    if n == 1:
        dsdr[0] = 0.0
        return s, dsdr

    dsdr[0] = np.nan
    dsdr[1] = np.nan

    for i in range(2, n - 1):
        dsdr[i] = (s[i + 1] - s[i - 1]) / (rmid[i + 1] - rmid[i - 1])

    dsdr[n - 1] = (s[n - 1] - s[n - 2]) / (rmid[n - 1] - rmid[n - 2])

    return s, dsdr

@njit(float64[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_sc1(v2, rho, rmid):
    """
    Schwarzschild criterion:

        SC1 = ((rho/(gamma*p)) * dp/dr) / (drho/dr)

    SC1 > 1 implies stability against convection.
    """
    n = v2.size

    p = np.empty(n, dtype=np.float64)
    dpdr = np.empty(n, dtype=np.float64)
    drhodr = np.empty(n, dtype=np.float64)
    sc1 = np.empty(n, dtype=np.float64)

    # p = rho * v2
    for i in range(n):
        p[i] = rho[i] * v2[i]

    if n == 1:
        sc1[0] = np.nan
        return sc1

    # one-sided lower boundary
    dr = rmid[1] - rmid[0]
    if dr == 0.0:
        dpdr[0] = np.nan
        drhodr[0] = np.nan
    else:
        dpdr[0] = (p[1] - p[0]) / dr
        drhodr[0] = (rho[1] - rho[0]) / dr

    # centered interior
    for i in range(1, n - 1):
        dr = rmid[i + 1] - rmid[i - 1]
        if dr == 0.0:
            dpdr[i] = np.nan
            drhodr[i] = np.nan
        else:
            dpdr[i] = (p[i + 1] - p[i - 1]) / dr
            drhodr[i] = (rho[i + 1] - rho[i - 1]) / dr

    # one-sided upper boundary
    dr = rmid[n - 1] - rmid[n - 2]
    if dr == 0.0:
        dpdr[n - 1] = np.nan
        drhodr[n - 1] = np.nan
    else:
        dpdr[n - 1] = (p[n - 1] - p[n - 2]) / dr
        drhodr[n - 1] = (rho[n - 1] - rho[n - 2]) / dr

    for i in range(n):
        denom = GAMMA * p[i] * drhodr[i]

        if denom == 0.0:
            sc1[i] = np.nan
        else:
            sc1[i] = rho[i] * dpdr[i] / denom

    return sc1

@njit(float64[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_sc2(v2, rho, rmid):
    """
    Schwarzschild criterion #2:

        SC2 = ((1 - 1/gamma) * (v2/p) * abs(dp/dr)) / abs(dv2/dr)

    Since p = rho * v2,

        v2/p = 1/rho

    so

        SC2 = ((1 - 1/gamma) * abs(dp/dr)) / (rho * abs(dv2/dr))

    SC2 > 1 implies stability against convection.
    """
    n = v2.size

    p = np.empty(n, dtype=np.float64)
    dpdr = np.empty(n, dtype=np.float64)
    dv2dr = np.empty(n, dtype=np.float64)
    sc2 = np.empty(n, dtype=np.float64)

    for i in range(n):
        p[i] = rho[i] * v2[i]

    if n == 1:
        sc2[0] = np.nan
        return sc2

    sc2[0] = np.nan
    sc2[1] = np.nan

    if n == 2:
        return sc2

    # centered interior
    for i in range(2, n - 1):
        dr = rmid[i + 1] - rmid[i - 1]
        if dr == 0.0:
            dpdr[i] = np.nan
            dv2dr[i] = np.nan
        else:
            dpdr[i] = (p[i + 1] - p[i - 1]) / dr
            dv2dr[i] = (v2[i + 1] - v2[i - 1]) / dr

    # one-sided upper boundary
    dr = rmid[n - 1] - rmid[n - 2]
    if dr == 0.0:
        dpdr[n - 1] = np.nan
        dv2dr[n - 1] = np.nan
    else:
        dpdr[n - 1] = (p[n - 1] - p[n - 2]) / dr
        dv2dr[n - 1] = (v2[n - 1] - v2[n - 2]) / dr

    prefac = 1.0 - 1.0 / GAMMA

    for i in range(2, n):
        denom = rho[i] * np.abs(dv2dr[i])

        if denom == 0.0:
            sc2[i] = np.nan
        else:
            sc2[i] = prefac * np.abs(dpdr[i]) / denom

    return sc2

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def calc_dlogrho_dlogp(v2, rho):
    """
    Return

        dlog(rho) / dlog(P)

    using P = rho * v2.

    This is computed as

        (dlog(rho)/dr) / (dlog(P)/dr)
      = (P/rho) * (drho/dr) / (dP/dr)
      = v2 * (drho/dr) / (dP/dr)

    using one-sided boundary and centered interior stencil style.

    The first two cells are set to NaN because they fluctuate.
    """
    n = v2.size
    out = np.empty(n, dtype=np.float64)

    if n == 0:
        return out

    out[0] = np.nan

    if n == 1:
        return out

    out[1] = np.nan

    if n == 2:
        return out

    # centered interior
    for i in range(2, n - 1):
        out[i] = (
            v2[i]
            * (rho[i + 1] - rho[i - 1])
            / (rho[i + 1] * v2[i + 1] - rho[i - 1] * v2[i - 1])
        )

    # one-sided upper boundary
    out[n - 1] = (
        v2[n - 1]
        * (rho[n - 1] - rho[n - 2])
        / (rho[n - 1] * v2[n - 1] - rho[n - 2] * v2[n - 2])
    )

    return out

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def calc_dlnrho_dlnr(rho, rmid):
    """
    Compute dln(rho)/dln(r) via finite differences on possibly nonuniform rmid.
    """
    n = rho.size
    dlnrho_dlnr = np.empty(n, dtype=np.float64)

    dlnrho_dlnr[0] = np.nan

    dlnrho_dlnr[1] = (
        math.log(rho[2]) - math.log(rho[1])
    ) / (
        math.log(rmid[2]) - math.log(rmid[1])
    )

    for i in range(2, n - 1):
        dlnrho_dlnr[i] = (
            math.log(rho[i + 1]) - math.log(rho[i - 1])
        ) / (
            math.log(rmid[i + 1]) - math.log(rmid[i - 1])
        )

    dlnrho_dlnr[n - 1] = (
        math.log(rho[n - 1]) - math.log(rho[n - 2])
    ) / (
        math.log(rmid[n - 1]) - math.log(rmid[n - 2])
    )

    return dlnrho_dlnr

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def calc_dlnv_dlnr(v2, rmid):
    """
    Compute dln(v)/dln(r) via finite differences on possibly nonuniform rmid.

    Input is v2 = v^2, so dln(v)/dln(r) = 0.5 * dln(v2)/dln(r).

    Index 0 is set to nan.
    Index 1 uses a forward one-sided finite difference.
    Interior points use centered finite differences.
    Final index uses a backward one-sided finite difference.
    """
    n = v2.size
    dlnv_dlnr = np.empty(n, dtype=np.float64)

    if n == 1:
        dlnv_dlnr[0] = np.nan
        return dlnv_dlnr

    dlnv_dlnr[0] = np.nan

    dlnv_dlnr[1] = 0.5 * (
        math.log(v2[2]) - math.log(v2[1])
    ) / (
        math.log(rmid[2]) - math.log(rmid[1])
    )

    for i in range(2, n - 1):
        dlnv_dlnr[i] = 0.5 * (
            math.log(v2[i + 1]) - math.log(v2[i - 1])
        ) / (
            math.log(rmid[i + 1]) - math.log(rmid[i - 1])
        )

    dlnv_dlnr[n - 1] = 0.5 * (
        math.log(v2[n - 1]) - math.log(v2[n - 2])
    ) / (
        math.log(rmid[n - 1]) - math.log(rmid[n - 2])
    )

    return dlnv_dlnr
