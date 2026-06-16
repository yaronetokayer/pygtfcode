import numpy as np
import math
from numba import njit, void, float64, types

@njit(types.Tuple((float64, float64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_core_r_rho_m_v2(r, rmid, rho, v2, m):
    """
    Computes core radius, core average density, core mass, and core v2.

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
    rho_c : float
        Mean density in the core
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

            if r_c > 0.0:
                rho_c = 3.0 * m_c / (r_c * r_c * r_c)
            else:
                rho_c = rho[0]

            return r_c, rho_c, m_c, v2_c

        # No crossing yet; shell j-1 is fully inside for future crossings
        dm_prev = m[j] - m[j - 1]
        numv_full += dm_prev * v2[j - 1]

        log_r_prev      = math.log(rmid[j])
        log_rho_prev    = math.log(rho_cur)

    # If no crossing is found, return outermost values.
    r_c = r[N]
    m_c = m[N]

    dm_last = m[N] - m[N - 1]
    numv_full += dm_last * v2[N - 1]

    if m_c > 0.0:
        v2_c = numv_full / m_c
    else:
        v2_c = v2[0]

    if r_c > 0.0:
        rho_c = 3.0 * m_c / (r_c * r_c * r_c)
    else:
        rho_c = rho[0]

    return r_c, rho_c, m_c, v2_c

@njit(types.Tuple((float64, float64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_rm2_rho_m_v2(r, rmid, rho, v2, m):
    """
    Computes r_m2, average density inside r_m2, enclosed mass, and core v2.

    r_m2 is defined such that

        d(ln rho) / d(ln r) = -2.

    r_m2 is estimated by finding the first place where the log-log
    density slope crosses -2. Slopes are measured between adjacent rmid
    values and assigned to the geometric midpoint of each interval.
    The crossing location is then found by linear interpolation of the
    slope in log-radius.

    m_m2 is estimated by taking the shell that contains r_m2 and using
    a constant-density-within-shell volume fraction.
    v2_m2 is estimated as a shell-mass-weighted average of v2 inside r_m2.
    """
    N = rmid.shape[0]

    target = -2.0

    numv_full = 0.0

    log_r_im2   = math.log(rmid[0])
    log_rho_im2 = math.log(rho[0])

    log_r_im1   = math.log(rmid[1])
    log_rho_im1 = math.log(rho[1])

    slope_prev = (log_rho_im1 - log_rho_im2) / (log_r_im1 - log_r_im2)
    log_rslope_prev = 0.5 * (log_r_im2 + log_r_im1)

    # Shell 0 is fully inside for any later crossing
    dm_prev = m[1] - m[0]
    numv_full += dm_prev * v2[0]

    for j in range(2, N):
        log_r_cur   = math.log(rmid[j])
        log_rho_cur = math.log(rho[j])

        slope_cur = (log_rho_cur - log_rho_im1) / (log_r_cur - log_r_im1)
        log_rslope_cur = 0.5 * (log_r_im1 + log_r_cur)

        # Find the first place where the slope crosses -2
        if slope_prev > target and slope_cur <= target:
            dslope = slope_cur - slope_prev

            if dslope != 0.0:
                log_r_m2 = log_rslope_prev + (target - slope_prev) * (
                    log_rslope_cur - log_rslope_prev
                ) / dslope
                r_m2 = math.exp(log_r_m2)
            else:
                r_m2 = math.exp(log_rslope_cur)

            # Accumulate m and v2 until r_m2
            if r_m2 <= r[j]:
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

            # Find m_m2 assuming constant densities
            if r_m2 <= rk0:
                frac = 0.0
            elif r_m2 >= rk1:
                frac = 1.0
            else:
                rm23 = r_m2 * r_m2 * r_m2
                rk03 = rk0 * rk0 * rk0
                rk13 = rk1 * rk1 * rk1
                frac = (rm23 - rk03) / (rk13 - rk03)

            m_m2 = mk0 + frac * (mk1 - mk0)

            # Complete v2_m2 computation with last partial shell
            dm_partial = m_m2 - mk0
            numv += dm_partial * v2[k]

            if m_m2 > 0.0:
                v2_m2 = numv / m_m2
            else:
                v2_m2 = v2[0]

            if r_m2 > 0.0:
                rho_m2 = 3.0 * m_m2 / (r_m2 * r_m2 * r_m2)
            else:
                rho_m2 = rho[0]

            return r_m2, rho_m2, m_m2, v2_m2

        # No crossing yet; shell j-1 is fully inside for future crossings
        dm_prev = m[j] - m[j - 1]
        numv_full += dm_prev * v2[j - 1]

        log_r_im1 = log_r_cur
        log_rho_im1 = log_rho_cur
        slope_prev = slope_cur
        log_rslope_prev = log_rslope_cur

    # If no crossing is found, return outermost values.
    r_m2 = r[N]
    m_m2 = m[N]

    dm_last = m[N] - m[N - 1]
    numv_full += dm_last * v2[N - 1]

    if m_m2 > 0.0:
        v2_m2 = numv_full / m_m2
    else:
        v2_m2 = v2[0]

    if r_m2 > 0.0:
        rho_m2 = 3.0 * m_m2 / (r_m2 * r_m2 * r_m2)
    else:
        rho_m2 = rho[0]

    return r_m2, rho_m2, m_m2, v2_m2

@njit(types.Tuple((float64, float64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_smfp_r_rho_m_v2(r, rmid, kn, rho, v2, m):
    """
    Computes SMFP radius, SMFP average density, SMFP mass, and SMFP v2.

    SMFP radius is defined as r_smfp where Kn crosses 1.0.

    r_smfp is estimated by log-log interpolation in Kn(r).
    m_smfp is estimated by taking the shell that contains r_smfp and using
    a constant-density-within-shell volume fraction.
    v2_smfp is estimated as a shell-mass-weighted average of v2 inside r_smfp.

    Arguments
    ---------
    r : ndarray, shape (N+1,)
        Shell edge radii.
    rmid : ndarray, shape (N,)
        Midpoint radii.
    kn : ndarray, shape (N,)
        Knudsen number at rmid.
    rho : ndarray, shape (N,)
        Shell densities.
    v2 : ndarray, shape (N,)
        Shell square of 1D velocity dispersion.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.

    Returns
    -------
    r_smfp : float
        SMFP radius
    rho_smfp : float
        Average density within r_smfp
    m_smfp : float
        Enclosed mass at r_smfp
    v2_smfp : float
        Mass-weighted v2 within r_smfp
    """
    N = rmid.shape[0]

    target_kn = 1.0
    log_target_kn = 0.0

    numv_full = 0.0

    log_r_prev  = math.log(rmid[0])
    log_kn_prev = math.log(kn[0])

    for j in range(1, N):
        kn_cur = kn[j]

        # Find the first shell where Kn crosses 1.0
        if (kn_cur - target_kn) * (kn[j - 1] - target_kn) <= 0.0:
            # Log-log interpolation for r_smfp
            log_r_cur  = math.log(rmid[j])
            log_kn_cur = math.log(kn_cur)

            logslope = (log_kn_cur - log_kn_prev) / (log_r_cur - log_r_prev)

            if logslope != 0.0:
                log_r_smfp = log_r_prev + (log_target_kn - log_kn_prev) / logslope
                r_smfp = math.exp(log_r_smfp)
            else:
                r_smfp = rmid[j - 1]

            # Accumulate m and v2 until r_smfp
            if r_smfp <= r[j]:
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

            # Find m_smfp assuming constant density in shell k
            if r_smfp <= rk0:
                frac = 0.0
            elif r_smfp >= rk1:
                frac = 1.0
            else:
                rsmfp3 = r_smfp * r_smfp * r_smfp
                rk03 = rk0 * rk0 * rk0
                rk13 = rk1 * rk1 * rk1
                frac = (rsmfp3 - rk03) / (rk13 - rk03)

            m_smfp = mk0 + frac * (mk1 - mk0)

            # Complete v2_smfp computation with last partial shell
            dm_partial = m_smfp - mk0
            numv += dm_partial * v2[k]

            if m_smfp > 0.0:
                v2_smfp = numv / m_smfp
            else:
                v2_smfp = v2[0]

            if r_smfp > 0.0:
                rho_smfp = 3.0 * m_smfp / (r_smfp * r_smfp * r_smfp)
            else:
                rho_smfp = rho[0]

            return r_smfp, rho_smfp, m_smfp, v2_smfp

        # No crossing yet; shell j-1 is fully inside for future crossings
        dm_prev = m[j] - m[j - 1]
        numv_full += dm_prev * v2[j - 1]

        log_r_prev  = math.log(rmid[j])
        log_kn_prev = math.log(kn_cur)

    # If no crossing is found, return outermost values.
    r_smfp = r[N]
    m_smfp = m[N]

    dm_last = m[N] - m[N - 1]
    numv_full += dm_last * v2[N - 1]

    if m_smfp > 0.0:
        v2_smfp = numv_full / m_smfp
    else:
        v2_smfp = v2[0]

    if r_smfp > 0.0:
        rho_smfp = 3.0 * m_smfp / (r_smfp * r_smfp * r_smfp)
    else:
        rho_smfp = rho[0]

    return r_smfp, rho_smfp, m_smfp, v2_smfp

@njit(types.Tuple((float64, float64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calc_mintheta_r_rho_m_v2(r, rmid, rho, v2, m, Theta):
    """
    Computes core radius, core average density, core mass, and core v2.

    Core radius is defined as the radius where Theta is minimum.
    Theta is defined at rmid.

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
    Theta : ndarray, shape (N,)
        Local cooling-to-sound-crossing time ratio.

    Returns
    -------
    r_c : float
        Radius of min Theta.
    rho_c : float
        Mean density within r_c.
    m_c : float
        Mass within r_c.
    v2_c : float
        v2 within r_c.
    """
    N = rmid.shape[0]

    # Find index of minimum Theta
    k = 0
    theta_min = Theta[0]
    for j in range(1, N):
        if Theta[j] < theta_min:
            theta_min = Theta[j]
            k = j

    r_c = rmid[k]

    # Mass inside r_c, assuming constant density within shell k
    rk0 = r[k]
    rk1 = r[k + 1]
    mk0 = m[k]
    mk1 = m[k + 1]

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

    # Mass-weighted v2 inside r_c
    numv = 0.0

    for j in range(k):
        dm = m[j + 1] - m[j]
        numv += dm * v2[j]

    dm_partial = m_c - mk0
    numv += dm_partial * v2[k]

    if m_c > 0.0:
        v2_c = numv / m_c
    else:
        v2_c = v2[0]

    if r_c > 0.0:
        rho_c = 3.0 * m_c / (r_c * r_c * r_c)
    else:
        rho_c = rho[0]

    return r_c, rho_c, m_c, v2_c

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

    half_window = window // 2

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
            xbar += math.log(m_c[j])
            ybar += math.log(v2_c[j])
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            x = math.log(m_c[j])
            y = math.log(v2_c[j])

            dx = x - xbar
            dy = y - ybar

            num += dx * dy
            den += dx * dx

        if den > 0.0:
            zeta[i] = num / den + 1.0
        else:
            zeta[i] = 1.0

    return zeta

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True)
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

    half_window = window // 2

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
            xbar += math.log(v2_c[j])
            ybar += math.log(m_c[j])
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            x = math.log(v2_c[j])
            y = math.log(m_c[j])

            dx = x - xbar
            dy = y - ybar

            num += dx * dy
            den += dx * dx

        if den > 0.0:
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

    half_window = window // 2

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
            xbar += math.log(v2_c[j])
            ybar += math.log(rho_c[j])
            count += 1

        xbar /= count
        ybar /= count

        # Second pass: compute least-squares slope in log-log space.
        num = 0.0
        den = 0.0

        for j in range(i0, i1):
            x = math.log(v2_c[j])
            y = math.log(rho_c[j])

            dx = x - xbar
            dy = y - ybar

            num += dx * dy
            den += dx * dx

        if den > 0.0:
            zeta[i] = 2.0 * num / den
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

    dsdr[0] = (s[1] - s[0]) / (rmid[1] - rmid[0])

    for i in range(1, n - 1):
        dsdr[i] = (s[i + 1] - s[i - 1]) / (rmid[i + 1] - rmid[i - 1])

    dsdr[n - 1] = (s[n - 1] - s[n - 2]) / (rmid[n - 1] - rmid[n - 2])

    return s, dsdr

GAMMA = 5.0 / 3.0

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
        sc1[0] = 0.0
        return sc1

    # one-sided boundaries
    dpdr[0] = (p[1] - p[0]) / (rmid[1] - rmid[0])
    drhodr[0] = (rho[1] - rho[0]) / (rmid[1] - rmid[0])

    # centered interior
    for i in range(1, n - 1):
        dpdr[i] = (p[i + 1] - p[i - 1]) / (rmid[i + 1] - rmid[i - 1])
        drhodr[i] = (rho[i + 1] - rho[i - 1]) / (rmid[i + 1] - rmid[i - 1])

    # one-sided boundaries
    dpdr[n - 1] = (p[n - 1] - p[n - 2]) / (rmid[n - 1] - rmid[n - 2])
    drhodr[n - 1] = (rho[n - 1] - rho[n - 2]) / (rmid[n - 1] - rmid[n - 2])

    for i in range(n):
        sc1[i] = rho[i] * dpdr[i] / (GAMMA * p[i] * drhodr[i])

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
        sc2[0] = 0.0
        return sc2

    dpdr[0] = (p[1] - p[0]) / (rmid[1] - rmid[0])
    dv2dr[0] = (v2[1] - v2[0]) / (rmid[1] - rmid[0])

    for i in range(1, n - 1):
        dpdr[i] = (p[i + 1] - p[i - 1]) / (rmid[i + 1] - rmid[i - 1])
        dv2dr[i] = (v2[i + 1] - v2[i - 1]) / (rmid[i + 1] - rmid[i - 1])

    dpdr[n - 1] = (p[n - 1] - p[n - 2]) / (rmid[n - 1] - rmid[n - 2])
    dv2dr[n - 1] = (v2[n - 1] - v2[n - 2]) / (rmid[n - 1] - rmid[n - 2])

    prefac = 1.0 - 1.0 / GAMMA

    for i in range(n):
        sc2[i] = prefac * np.abs(dpdr[i]) / (rho[i] * np.abs(dv2dr[i]))

    return sc2

# @njit(types.Tuple((float64, float64, float64, float64))(float64[:], float64[:], float64[:], float64[:], float64), fastmath=True, cache=True)
# def calc_smfp_r_rho_m_v2(r, rho, v2, m, sigma_m):
#     """
#     Computes SMFP radius, SMFP mass, and SMFP v2.

#     SMFP radius is defined as r_smfp such that

#         int_0^{r_smfp} rho(r) dr = 1 / sigma_m

#     This is where the optical depth equals 1.
#     The integral is estimated by simple Riemann sums, assuming rho is
#     constant within each shell. If the threshold is crossed within a
#     shell, r_smfp is found by linear interpolation in radius within that
#     shell.

#     m_smfp is estimated by taking the shell that contains r_smfp and using
#     a constant-density-within-shell volume fraction.
#     v2_smfp is estimated as a shell-mass-weighted average of v2 inside r_smfp.

#     Arguments
#     ---------
#     r : ndarray, shape (N+1,)
#         Shell edge radii.
#     rho : ndarray, shape (N,)
#         Shell densities.
#     v2 : ndarray, shape (N,)
#         Shell square of 1D velocity dispersion.
#     m : ndarray, shape (N+1,)
#         Enclosed mass at shell edges.
#     sigma_m : float
#         Cross section per unit mass.

#     Returns
#     -------
#     r_smfp : float
#         SMFP radius
#     rho_smfp : float
#         Average density within r_smfp
#     m_smfp : float
#         Enclosed mass at r_smfp
#     v2_smfp : float
#         Mass-weighted v2 within r_smfp
#     """
#     N = rho.shape[0]

#     if sigma_m <= 0.0:
#         return 0.0, rho[0], 0.0, v2[0]

#     tau_target = 1.0 / sigma_m
#     tau_prev = 0.0

#     numv_full = 0.0 # Numerator for v2 computation

#     r1 = r[0]
#     for j in range(N):
#         r0 = r1
#         r1 = r[j + 1]
#         dr = r1 - r0

#         tau_cur = tau_prev + rho[j] * dr

#         if tau_cur >= tau_target:
#             if tau_cur > tau_prev:
#                 frac_r = (tau_target - tau_prev) / (tau_cur - tau_prev)
#             else:
#                 frac_r = 0.0

#             r_smfp = r0 + frac_r * dr

#             m0 = m[j]
#             m1 = m[j + 1]

#             if r_smfp <= r0:
#                 frac_m = 0.0
#             else:
#                 r_smfp3 = r_smfp * r_smfp * r_smfp
#                 r03 = r0 * r0 * r0
#                 r13 = r1 * r1 * r1
#                 frac_m = (r_smfp3 - r03) / (r13 - r03)

#             m_smfp = m0 + frac_m * (m1 - m0)

#             if r_smfp > 0.0:
#                 rho_smfp = 3.0 * m_smfp / (r_smfp * r_smfp * r_smfp)
#             else:
#                 rho_smfp = rho[0]

#             dm_partial = m_smfp - m0
#             numv = numv_full + dm_partial * v2[j]

#             if m_smfp > 0.0:
#                 v2_smfp = numv / m_smfp
#             else:
#                 v2_smfp = v2[0]

#             return r_smfp, rho_smfp, m_smfp, v2_smfp
        
#         dm = m[j + 1] - m[j]
#         numv_full += dm * v2[j]

#         tau_prev = tau_cur

#     return 0.0, rho[0], 0.0, v2[0]