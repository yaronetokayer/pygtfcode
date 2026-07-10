"""
Helpers to compute quantities related to the core
"""

import math
from numba import njit, float64, types
from pygtfcode.util.interpolate import interp_pl_to_r

### CORE DEFINITIONS

@njit(float64(float64[:], float64[:], float64[:]), fastmath=True, cache=True,)
def calc_core_r(r, rmid, rho):
    """
    Find the radius at which rho falls to half its central value.

    The crossing is calculated by assuming rho is a local power law
    between adjacent shell midpoint radii.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii.
    rmid : ndarray, shape (N,)
        Shell midpoint radii.
    rho : ndarray, shape (N,)
        Density evaluated at shell midpoints.

    Returns
    -------
    r_c : float
        Radius at which rho(r_c) = 0.5 * rho[0]. If no crossing is
        found, the outermost shell-edge radius is returned.
    """
    N = rho.shape[0]

    rho_target = 0.5 * rho[0]
    log_rho_target = math.log(rho_target)

    log_r_prev = math.log(rmid[0])
    log_rho_prev = math.log(rho[0])

    for j in range(1, N):
        if rho[j] <= rho_target:
            log_r_cur = math.log(rmid[j])
            log_rho_cur = math.log(rho[j])

            slope = (
                (log_rho_cur - log_rho_prev)
                / (log_r_cur - log_r_prev)
            )

            log_r_c = (
                log_r_prev
                + (log_rho_target - log_rho_prev) / slope
            )

            return math.exp(log_r_c)

        log_r_prev = math.log(rmid[j])
        log_rho_prev = math.log(rho[j])

    return r[N]

@njit(float64(float64[:], float64[:], float64[:], float64,), fastmath=True, cache=True,)
def calc_r_mn(r, rmid, rho, n):
    """
    Compute the radius r_mn where the logarithmic density slope first
    crosses -n:

        d(ln rho) / d(ln r) = -n.

    Density slopes are calculated between adjacent shell-midpoint
    values and assigned to the geometric midpoint of each interval.
    The crossing radius is found by linearly interpolating the slope
    in log-radius.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii.
    rmid : ndarray, shape (N,)
        Shell midpoint radii.
    rho : ndarray, shape (N,)
        Density evaluated at the shell midpoints. Values must be
        strictly positive.
    n : float
        Positive magnitude of the target logarithmic slope.

    Returns
    -------
    r_mn : float
        First radius where the density slope crosses from above -n
        to at or below -n. If no crossing is found, the outermost
        shell-edge radius is returned.
    """
    N = rmid.shape[0]

    if N < 2:
        return r[N]

    target = -n

    log_r_prev = math.log(rmid[0])
    log_rho_prev = math.log(rho[0])

    log_r_cur = math.log(rmid[1])
    log_rho_cur = math.log(rho[1])

    slope_prev = (
        (log_rho_cur - log_rho_prev)
        / (log_r_cur - log_r_prev)
    )
    log_rslope_prev = 0.5 * (log_r_prev + log_r_cur)

    for j in range(2, N):
        log_r_next = math.log(rmid[j])
        log_rho_next = math.log(rho[j])

        slope_cur = (
            (log_rho_next - log_rho_cur)
            / (log_r_next - log_r_cur)
        )
        log_rslope_cur = 0.5 * (log_r_cur + log_r_next)

        # Find the first downward crossing of the target slope.
        if slope_prev > target and slope_cur <= target:
            dslope = slope_cur - slope_prev

            if dslope != 0.0:
                log_r_mn = log_rslope_prev + (
                    (target - slope_prev)
                    * (log_rslope_cur - log_rslope_prev)
                    / dslope
                )
            else:
                log_r_mn = log_rslope_cur

            return math.exp(log_r_mn)

        log_r_cur = log_r_next
        log_rho_cur = log_rho_next
        slope_prev = slope_cur
        log_rslope_prev = log_rslope_cur

    return r[N]

@njit(float64(float64[:], float64[:], float64[:],), fastmath=True, cache=True,)
def calc_r_smfp(r, rmid, kn):
    """
    Compute the SMFP radius, defined as the first radius where the
    Knudsen number crosses unity:

        Kn(r_smfp) = 1.

    The crossing radius is estimated by assuming Kn is a local power
    law between adjacent shell midpoint radii, equivalent to linear
    interpolation in log(Kn) versus log(r).

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii. Used to return the outermost edge if no
        crossing is found.

    rmid : ndarray, shape (N,)
        Shell midpoint radii.

    kn : ndarray, shape (N,)
        Knudsen number evaluated at the shell midpoints. Values must
        be strictly positive.

    Returns
    -------
    r_smfp : float
        First radius where Kn crosses 1. If no crossing is found,
        the outermost shell-edge radius is returned.
    """
    N = rmid.shape[0]

    if N < 2:
        return r[N]

    log_r_prev = math.log(rmid[0])
    log_kn_prev = math.log(kn[0])

    for j in range(1, N):
        kn_prev = kn[j - 1]
        kn_cur = kn[j]

        # Find the first crossing of Kn = 1 in either direction.
        if (kn_prev - 1.0) * (kn_cur - 1.0) <= 0.0:
            log_r_cur = math.log(rmid[j])
            log_kn_cur = math.log(kn_cur)

            dlog_kn = log_kn_cur - log_kn_prev

            if dlog_kn != 0.0:
                # log(Kn_target) = log(1) = 0.
                log_r_smfp = (
                    log_r_prev
                    - log_kn_prev
                    * (log_r_cur - log_r_prev)
                    / dlog_kn
                )

                return math.exp(log_r_smfp)

            # Both points have the same Kn value. This can only define
            # a crossing unambiguously when both values equal one.
            return rmid[j - 1]

        log_r_prev = math.log(rmid[j])
        log_kn_prev = math.log(kn_cur)

    return r[N]

### CORE AVERAGES

@njit(float64(float64[:], float64[:], float64[:], float64,), fastmath=True, cache=True,)
def calc_logmean_within_r(r, m, q, r_max):
    """
    Compute the mass-weighted geometric mean of q inside r_max.

    The returned quantity is

        exp[sum(dm * log(q)) / sum(dm)].

    q is treated as constant within each shell. If r_max lies within
    a shell, the enclosed mass at r_max is found with
    interp_pl_to_r(r, m, r_max).

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.
    q : ndarray, shape (N,)
        Shell quantity to average. Every included value with nonzero
        shell mass must be strictly positive.

    r_max : float
        Outer radius of the averaging region.

    Returns
    -------
    q_mean : float
        Mass-weighted geometric mean of q.
    """
    N = q.shape[0]

    if r_max <= r[0]:
        return q[0]

    weighted_sum = 0.0
    included_mass = 0.0

    for k in range(N):
        # Entire shell lies inside r_max.
        if r_max >= r[k + 1]:
            dm = m[k + 1] - m[k]

        # r_max lies inside this shell.
        elif r_max > r[k]:
            m_at_rmax = interp_pl_to_r(r, m, r_max)
            dm = m_at_rmax - m[k]

        # No part of this or any subsequent shell is included.
        else:
            break

        if dm > 0.0:
            if q[k] <= 0.0:
                raise ValueError(
                    "The logarithmic mean requires q > 0."
                )

            weighted_sum += dm * math.log(q[k])
            included_mass += dm

        # Stop after including the partial outer shell.
        if r_max < r[k + 1]:
            break

    if included_mass > 0.0:
        return math.exp(weighted_sum / included_mass)

    return q[0]

@njit(float64(float64[:], float64[:], float64[:], float64,), fastmath=True, cache=True,)
def calc_mean_within_r(r, m, q, r_max):
    """
    Compute the arithmetic mass-weighted mean of q inside r_max.

    The returned quantity is

        sum(dm * q) / sum(dm).

    q is treated as constant within each shell. If r_max lies within
    a shell, the enclosed mass at r_max is found with
    interp_pl_to_r(r, m, r_max).

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.
    q : ndarray, shape (N,)
        Shell quantity to average.
    r_max : float
        Outer radius of the averaging region.

    Returns
    -------
    q_mean : float
        Arithmetic mass-weighted mean of q.
    """
    N = q.shape[0]

    if r_max <= r[0]:
        return q[0]

    weighted_sum = 0.0
    included_mass = 0.0

    for k in range(N):
        # Entire shell lies inside r_max.
        if r_max >= r[k + 1]:
            dm = m[k + 1] - m[k]

        # r_max lies inside this shell.
        elif r_max > r[k]:
            m_at_rmax = interp_pl_to_r(r, m, r_max)
            dm = m_at_rmax - m[k]

        # No part of this or any subsequent shell is included.
        else:
            break

        if dm > 0.0:
            weighted_sum += dm * q[k]
            included_mass += dm

        # Stop after including the partial outer shell.
        if r_max < r[k + 1]:
            break

    if included_mass > 0.0:
        return weighted_sum / included_mass

    return q[0]

### COMBINATIONS FOR ACCESSIBILITY

@njit(types.Tuple((float64, float64, float64, float64, float64,))(float64[:], float64[:], float64[:], float64[:], float64[:],), fastmath=True, cache=True,)
def calc_core_r_rho_m_v2(r, rmid, rho, v2, m):
    """
    Compute core properties using the reusable interpolation and
    averaging functions.
    """
    r_c = calc_core_r(r, rmid, rho)

    # m is an edge quantity, so use r as its coordinate array.
    m_c = interp_pl_to_r(r, m, r_c)

    # v2 is a shell quantity, but its average is weighted by shell
    # masses, so r and m define the integration shells.
    v2_c = calc_mean_within_r(r, m, v2, r_c)

    if r_c > 0.0:
        rho_c = 3.0 * m_c / (r_c * r_c * r_c)
    else:
        rho_c = rho[0]

    if v2_c > 0.0:
        crossing_time = r_c / math.sqrt(v2_c)
    else:
        crossing_time = 0.0

    return r_c, rho_c, m_c, v2_c, crossing_time

@njit(types.Tuple((float64, float64, float64, float64,))(float64[:], float64[:], float64[:], float64[:], float64[:], float64,), fastmath=True, cache=True,)
def calc_rmn_rho_m_v2(r, rmid, rho, v2, m, n):
    """
    Compute r_mn, average density inside r_mn, enclosed mass, and
    mass-weighted mean v2.

    r_mn is defined by

        d(ln rho) / d(ln r) = -n.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii.
    rmid : ndarray, shape (N,)
        Shell midpoint radii.
    rho : ndarray, shape (N,)
        Density evaluated at shell midpoints.
    v2 : ndarray, shape (N,)
        Shell square of the 1D velocity dispersion.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.
    n : float
        Positive magnitude of the target logarithmic density slope.

    Returns
    -------
    r_mn : float
        Radius where the logarithmic density slope first crosses -n.
    rho_mn : float
        Average density inside r_mn.
    m_mn : float
        Enclosed mass at r_mn.
    v2_mn : float
        Mass-weighted arithmetic mean of v2 inside r_mn.
    """
    r_mn = calc_r_mn(r, rmid, rho, n)

    m_mn = interp_pl_to_r(r, m, r_mn)

    v2_mn = calc_mean_within_r(r, m, v2, r_mn,)

    if r_mn > 0.0:
        rho_mn = 3.0 * m_mn / (r_mn * r_mn * r_mn)
    else:
        rho_mn = rho[0]

    return r_mn, rho_mn, m_mn, v2_mn

@njit(types.Tuple((float64, float64, float64, float64,))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],), fastmath=True, cache=True,)
def calc_smfp_r_rho_m_v2(r, rmid, kn, rho, v2, m):
    """
    Compute the SMFP radius, average density inside it, enclosed mass,
    and mass-weighted mean v2.

    The SMFP radius is defined by

        Kn(r_smfp) = 1.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Shell-edge radii.
    rmid : ndarray, shape (N,)
        Shell midpoint radii.
    kn : ndarray, shape (N,)
        Knudsen number evaluated at shell midpoints.
    rho : ndarray, shape (N,)
        Density evaluated at shell midpoints.
    v2 : ndarray, shape (N,)
        Shell square of the 1D velocity dispersion.
    m : ndarray, shape (N+1,)
        Enclosed mass at shell edges.

    Returns
    -------
    r_smfp : float
        First radius where Kn crosses unity.
    rho_smfp : float
        Average density inside r_smfp.
    m_smfp : float
        Enclosed mass at r_smfp.
    v2_smfp : float
        Mass-weighted arithmetic mean of v2 inside r_smfp.
    """
    r_smfp = calc_r_smfp(r, rmid, kn)

    m_smfp = interp_pl_to_r(r, m, r_smfp)

    v2_smfp = calc_mean_within_r(r, m, v2, r_smfp,)

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
