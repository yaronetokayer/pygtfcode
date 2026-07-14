"""
Helpers to for derived quantities in the integration loop
"""

import numpy as np
import math
from numba import njit, void, float64, types
from pygtfcode.util.interpolate import interp_linear_to_interfaces, interp_pl_to_r
from pygtfcode.util.calc_core import calc_core_r, calc_mean_within_r

@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def low_kn_boost(kn_c, kn_threshold, boost, width):
    """
    Smooth multiplicative boost to eps_du in the low-Knudsen regime.

    Returns ~1 when kn_c >> kn_threshold,
    ~boost when kn_c << kn_threshold.
    """
    x = np.log10(kn_c / kn_threshold)
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

@njit(types.Tuple((float64[::1], float64[::1], float64[::1],))(float64[::1], float64[::1], float64[::1], float64, float64, float64, float64, float64,), cache=True, fastmath=True,)
def calc_kappa_cell(v2, rho, rmid, a_param, b_param, c_param, sigma_m, alph,):
    """
    Compute the cell-centered LMFP, SMFP, and interpolated conductivities, shape (N,)

        kappa_LMFP = (3/2) * v * r^2 * (c * rho * v^2)

        kappa_SMFP = (3/2) * v * r^2 * b / (a * sigma_m^2)

        kappa = ( kappa_LMFP^(-alpha) + kappa_SMFP^(-alpha) )^(-1/alpha).

    The returned conductivity is positive.

    Parameters
    ----------
    v2 : ndarray, shape (N,)
        One-dimensional velocity dispersion squared at cell centers.
    rho : ndarray, shape (N,)
        Mass density at cell centers.
    rmid : ndarray, shape (N,)
        Radial coordinate of cell centers.
    a_param : float
        Coefficient a in the SMFP conductivity.
    b_param : float
        Coefficient b in the SMFP conductivity.
    c_param : float
        Coefficient c in the LMFP conductivity.
    sigma_m : float
        Self-interaction cross section per unit mass.
    alph : float
        Positive interpolation parameter.

    Returns
    -------
    kappa_lmfp : ndarray, shape (N,)
        Cell-centered LMFP conductivity.
    kappa_smfp : ndarray, shape (N,)
        Cell-centered SMFP conductivity.
    kappa : ndarray, shape (N,)
        Interpolated cell-centered conductivity.
    """
    n = v2.size

    k_l = np.empty(n, dtype=np.float64)
    k_s = np.empty(n, dtype=np.float64)
    k   = np.empty(n, dtype=np.float64)

    # Quantities that are constant across all cells.
    smfp_fac = 1.5 * b_param / (
        a_param * sigma_m * sigma_m
    )
    lmfp_fac = 1.5 * c_param
    neg_alph = -alph
    neg_inv_alph = -1.0 / alph

    for i in range(n):
        v2_i = v2[i]
        r_i = rmid[i]

        # common geometric/velocity factor: v * r^2
        vr2 = math.sqrt(v2_i) * r_i * r_i

        k_l_i = lmfp_fac * vr2 * rho[i] * v2_i
        k_s_i = smfp_fac * vr2

        k_l[i] = k_l_i
        k_s[i] = k_s_i

        k[i] = (
            k_l_i**neg_alph
            + k_s_i**neg_alph
        )**neg_inv_alph

    return k_l, k_s, k

@njit(types.Tuple((float64[::1], float64[::1], float64[::1],))(float64[::1], float64[::1], float64[::1], float64, float64, float64, float64, float64,), cache=True, fastmath=True,)
def calc_kappa_edge(v2, rho, r, a_param, b_param, c_param, sigma_m, alph,):
    """
    Compute the edge LMFP, SMFP, and interpolated conductivities, shape (N,), but with
    the boundary condition of zero at the edge.

        kappa_LMFP = (3/2) * v * r^2 * (c * rho * v^2)

        kappa_SMFP = (3/2) * v * r^2 * b / (a * sigma_m^2)

        kappa = ( kappa_LMFP^(-alpha) + kappa_SMFP^(-alpha) )^(-1/alpha).

    The returned conductivity is positive.

    Parameters
    ----------
    v2 : ndarray, shape (N,)
        One-dimensional velocity dispersion squared at cell centers.
    rho : ndarray, shape (N,)
        Mass density at cell centers.
    r : ndarray, shape (N+1,)
        Radial coordinate of cell edges.
    a_param : float
        Coefficient a in the SMFP conductivity.
    b_param : float
        Coefficient b in the SMFP conductivity.
    c_param : float
        Coefficient c in the LMFP conductivity.
    sigma_m : float
        Self-interaction cross section per unit mass.
    alph : float
        Positive interpolation parameter.

    Returns
    -------
    kappa_lmfp : ndarray, shape (N,)
        Edge LMFP conductivity.
    kappa_smfp : ndarray, shape (N,)
        Edge SMFP conductivity.
    kappa : ndarray, shape (N,)
        Edge conductivity.
    """
    n = v2.size

    k_l = np.empty(n, dtype=np.float64)
    k_s = np.empty(n, dtype=np.float64)
    k   = np.empty(n, dtype=np.float64)

    rho_int = interp_linear_to_interfaces(r, rho) # Shape (N-1,)
    v2_int  = interp_linear_to_interfaces(r, v2)

    # Quantities that are constant across all cells.
    smfp_fac = 1.5 * b_param / (
        a_param * sigma_m * sigma_m
    )
    lmfp_fac = 1.5 * c_param
    neg_alph = -alph
    neg_inv_alph = -1.0 / alph

    for i in range(n-1):
        v2_i    = v2_int[i]
        r_i     = r[i+1]

        # common geometric/velocity factor: v * r^2
        vr2 = math.sqrt(v2_i) * r_i * r_i

        k_l_i = lmfp_fac * vr2 * rho_int[i] * v2_i
        k_s_i = smfp_fac * vr2

        k_l[i] = k_l_i
        k_s[i] = k_s_i

        k[i] = (
            k_l_i**neg_alph
            + k_s_i**neg_alph
        )**neg_inv_alph

    # Nan for last edge, boundary condition
    k_l[n-1] = np.nan; k_s[n-1] = np.nan; k[n-1] = np.nan

    return k_l, k_s, k

@njit(float64(float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64, float64, float64, float64, float64, float64,), cache=True, fastmath=True)
def calc_core_lum_dt(r, rmid, rho, v2, m, alpha, a, b, c, sigma_m, eps,):
    """
    Estimate the timestep that limits fractional core-energy loss.

    Returns
    -------
    dt : float
        math.inf:
            The luminosity constraint is inactive, for example because
            there is no resolved core interface or the gradient is zero.

        0.0:
            The evolved state contains nonfinite or nonphysical quantities.
            Returning zero is conservative for a timestep limiter.

    Raises
    ------
    ValueError
        For invalid static configuration or incompatible array shapes.
    """
    N = rmid.size

    # Compute core radius
    r_c = calc_core_r(r, rmid, rho)

    if not math.isfinite(r_c) or r_c <= 0.0:
        return math.inf

    r_first = r[0]
    r_last = r[N]
    rmid_first = rmid[0]
    rmid_last = rmid[N - 1]

    # Interpolation and the centered gradient both require r_c to lie inside
    # their domains. At rmid[-1], there is no interval to the right.
    if (
        r_c < r_first
        or r_c > r_last
        or r_c < rmid_first
        or r_c >= rmid_last
    ):
        return math.inf

    # Find the interval rmid[i - 1] <= r_c < rmid[i].
    # Return the first index i for which x[i] > value
    lo = 0; hi = N
    while lo < hi:
        mid = (lo + hi) >> 1
        if rmid[mid] <= r_c:
            lo = mid+1
        else:
            hi = mid

    i = lo

    if i <= 0 or i >= N:
        return math.inf

    dv2dr = (v2[i] - v2[i - 1]) / (rmid[i] - rmid[i - 1])

    grad_abs = abs(dv2dr)

    # No temperature gradient means no interface luminosity.
    if grad_abs == 0.0:
        return math.inf

    m_c = interp_pl_to_r(r, m, r_c)
    v2_c = calc_mean_within_r(r, m, v2, r_c)
    v2_int = interp_pl_to_r(rmid, v2, r_c)
    rho_int = interp_pl_to_r(rmid, rho, r_c)

    # Compute 1/kappa
    smfp_term = (a / b) * sigma_m * sigma_m
    lmfp_term = (1.0 / c) / rho_int / v2_int

    if smfp_term >= lmfp_term:
        scale = smfp_term
        ratio = lmfp_term / smfp_term
    else:
        scale = lmfp_term
        ratio = smfp_term / lmfp_term

    inv_kappa = scale * math.pow(1.0 + math.pow(ratio, alpha), 1.0 / alpha,)


    # The factors of 1.5 in E_c and lum_int cancel exactly:
    #
    # dt = eps * m_c * v2_c * (1/kappa)
    #      --------------------------------
    #      sqrt(v2_int) * r_c**2 * |dv2dr|
    numerator = eps * m_c * v2_c * inv_kappa
    denominator = math.sqrt(v2_int) * r_c * r_c * grad_abs

    return numerator / denominator