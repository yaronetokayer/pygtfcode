"""
Helpers for velocity-dependent cross-section moments and conductivity.

The Kp fit follows Eq. (12) of the supplied van den Bosch (2026) note,
with epsilon = 1e-12. Its argument is the one-dimensional velocity
dispersion divided by w, not the relative speed of a particle pair.
Individual moments retain the finite-epsilon normalization of the fit.
"""

import math
import numpy as np
from numba import njit

# Rows correspond to p = 3, 5, 7, 9; columns contain p0, p1, p2, p3.
COEFF = np.array([
    [8.0,  0.0339848, 0.37, 0.63],
    [24.0, 0.251115,  0.41, 0.71],
    [48.0, 0.682602,  0.42, 0.74],
    [80.0, 1.32953,   0.43, 0.76],
])


@njit(cache=True)
def softplus(z):
    """
    Compute ln(1 + exp(z)) without overflow for finite z.

    Parameters
    ----------
    z : float
        Dimensionless logarithmic argument.

    Returns
    -------
    value : float
        Softplus of z, evaluated using a stable exponential argument.
    """
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


@njit(cache=True)
def kp_log_slope(x, p):
    """
    Compute the logarithm of Kp and its logarithmic velocity slope.

    Parameters
    ----------
    x : float
        Ratio v / w, where v is the one-dimensional velocity dispersion.
        Must be finite and nonnegative.
    p : int
        Moment index. Supported values are 3, 5, 7, and 9.

    Returns
    -------
    log_kp : float
        Natural logarithm of the fitted moment Kp(x).
    slope : float
        dln(Kp) / dln(x). At x = 0, returns the limiting value zero.

    Notes
    -----
    The fit uses s = x^4 + 1e-12 and is evaluated in logarithmic form.
    The finite epsilon makes Kp(0) slightly different from one; no
    renormalization is applied to the individual moments.
    """
    if p not in (3, 5, 7, 9) or x < 0 or not math.isfinite(x):
        raise ValueError('Require finite x>=0 and p in (3,5,7,9)')

    p0, p1, p2, p3 = COEFF[(p - 3) // 2]

    # Stable logarithm of s = x^4 + epsilon, including x = 0.
    log_x4 = 4 * math.log(x) if x > 0 else -math.inf
    log_eps = math.log(1e-12)
    log_s = max(log_x4, log_eps) + math.log1p(
        math.exp(-abs(log_x4 - log_eps))
    )

    z = p2 * (log_s + math.log(p1))
    log_one_plus_exp_z = softplus(z)
    log_a = log_s + math.log(p0 * p2 / 1.5) - math.log(log_one_plus_exp_z)
    y = p3 * log_a
    log_kp = -softplus(y) / p3

    # Differentiate the same fit with respect to ln(x).
    logistic = math.exp(-softplus(-y))
    dlog_a_dlog_s = 1 - p2 * math.exp(-softplus(-z)) / log_one_plus_exp_z
    dlog_s_dlog_x = 4 * math.exp(log_x4 - log_s)
    slope = -logistic * dlog_a_dlog_s * dlog_s_dlog_x

    return log_kp, slope


@njit(cache=True)
def factors(T, what, order=2):
    """
    Compute LMFP and SMFP transport factors and their temperature slopes.

    Parameters
    ----------
    T : float
        Positive dimensionless one-dimensional velocity dispersion squared,
        corresponding to state.v2.
    what : float
        Positive dimensionless velocity scale w / v_s (char.w_char).
        Positive infinity selects the exact constant-cross-section branch.
    order : int, optional
        SMFP approximation order, either 1 or 2. Defaults to 2.
        Order 1 uses K5; order 2 uses the normalized K5, K7, K9 combination.

    Returns
    -------
    k_lmfp : float
        LMFP transport factor K5.
    k_smfp : float
        SMFP transport factor for the selected order.
    slope_lmfp : float
        dln(k_lmfp) / dln(T) at fixed what.
    slope_smfp : float
        dln(k_smfp) / dln(T) at fixed what.

    Notes
    -----
    Order 2 multiplies the raw second-order moment combination by 45/44.
    This preserves the chosen constant-limit coefficient b, rather than
    the absolute normalization of the raw second-order conductivity.
    The infinite-what branch returns (1, 1, 0, 0) before evaluating moments
    or checking order; input validation belongs to the parameter layer.
    """
    if math.isinf(what):
        return 1.0, 1.0, 0.0, 0.0

    x = math.sqrt(T) / what
    log_k5, slope_k5 = kp_log_slope(x, 5)
    k5 = math.exp(log_k5)
    slope_k5 *= 0.5

    if order == 1:
        return k5, k5, slope_k5, slope_k5
    if order != 2:
        raise ValueError('order must be 1 or 2')

    log_k7, slope_k7 = kp_log_slope(x, 7)
    log_k9, slope_k9 = kp_log_slope(x, 9)
    slope_k7 *= 0.5
    slope_k9 *= 0.5

    # Factor out K5 to avoid squaring small moments.
    ratio_k7 = math.exp(log_k7 - log_k5)
    ratio_k9 = math.exp(log_k9 - log_k5)
    numerator = 28 + 80 * ratio_k9 - 64 * ratio_k7 * ratio_k7
    denominator = 77 - 112 * ratio_k7 + 80 * ratio_k9

    # Derivatives of the factored numerator and denominator with respect to ln(T).
    dnumerator = (
        80 * ratio_k9 * (slope_k9 - slope_k5)
        - 128 * ratio_k7 * ratio_k7 * (slope_k7 - slope_k5)
    )
    ddenominator = (
        -112 * ratio_k7 * (slope_k7 - slope_k5)
        + 80 * ratio_k9 * (slope_k9 - slope_k5)
    )
    k_smfp = k5 * (numerator / denominator) * (45 / 44)
    slope_smfp = slope_k5 + dnumerator / numerator - ddenominator / denominator

    return k5, k_smfp, slope_k5, slope_smfp


@njit(cache=True)
def conductivity(T, rho, sigmahat, what, alph, a, b, c, order=2):
    """
    Compute the temperature-flux coefficient and its logarithmic slope.

        S = (a / b) * sigmahat^2 * k_smfp
        L = 1 / (c * rho * T * k_lmfp)
        D = (S^alph + L^alph)^(1 / alph)
        k = sqrt(T) / D

    Parameters
    ----------
    T : float
        Positive dimensionless one-dimensional velocity dispersion squared.
    rho : float
        Positive dimensionless mass density.
    sigmahat : float
        Positive cross-section amplitude in characteristic units,
        corresponding to char.sigma_m_0_char.
    what : float
        Positive velocity scale in characteristic units (char.w_char).
        Positive infinity selects constant-cross-section factors.
    alph : float
        Positive interpolation parameter between LMFP and SMFP transport.
    a : float
        Positive collision-rate coefficient.
    b : float
        Positive normalization coefficient for SMFP conductivity.
    c : float
        Positive normalization coefficient for LMFP conductivity.
    order : int, optional
        SMFP approximation order, either 1 or 2. Defaults to 2.
        See factors for the second-order normalization convention.

    Returns
    -------
    k : float
        Dimensionless coefficient sqrt(T) / D. Geometric factors and
        physical-unit conversions are not included.
    slope : float
        dln(k) / dln(T), holding density, amplitude, velocity scale, and
        transport parameters fixed. Includes derivatives of both factors.
    """
    k_lmfp, k_smfp, slope_lmfp, slope_smfp = factors(T, what, order)
    log_smfp = math.log(a / b) + 2 * math.log(sigmahat) + math.log(k_smfp)
    log_lmfp = -math.log(c * rho * T * k_lmfp)

    # Resistance weights and interpolated denominator, evaluated in log space.
    z = alph * (log_smfp - log_lmfp)
    weight_smfp = math.exp(-softplus(-z))
    weight_lmfp = 1 - weight_smfp
    log_d = max(log_smfp, log_lmfp) + math.log1p(
        math.exp(-alph * abs(log_smfp - log_lmfp))
    ) / alph

    k = math.exp(0.5 * math.log(T) - log_d)
    slope = 0.5 - weight_smfp * slope_smfp + weight_lmfp * (1 + slope_lmfp)

    return k, slope
