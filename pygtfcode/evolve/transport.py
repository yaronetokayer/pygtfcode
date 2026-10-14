import numpy as np
import math
from numba import njit, float64, types, void
from pygtfcode.util.interpolate import interp_linear_to_interfaces
from pygtfcode.util.calc_linalg import solve_tridiagonal_thomas
from pygtfcode.util.calc_kp import conductivity

_TINY64 = np.finfo(np.float64).tiny

@njit(void(float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, types.int64, float64[:], float64[:], float64[:], float64[:]), cache=True)
def build_tridiag_system(r, m, rho_int, v2, dt, a_param, b_param, c_param, alph, sigma_m_0_char, w_char, smfp_order, a, b, c, d,):
    """
    Build tridiagonal system for implicit conduction update in v2:

        a_i dv2_{i-1} + b_i dv2_i + c_i dv2_{i+1} = d_i

    Requires at least three cells. Supports positive alph and finite or
    infinite w_char. Linearizes the temperature-dependent flux once at
    the current state; timestep retries are not nonlinear Newton iterations.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii.
    m : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rho_int : ndarray, shape (N-1,)
        Interface density values.
    v2 : ndarray, shape (N,)
        Cell-centered one-dimensional velocity dispersion squared.
    dt : float
        Timestep.
    a_param, b_param, c_param : float
        Conductivity parameters.
    alph : float
        Coefficient for interpolation scheme between lmfp and smfp regimes.
        kappa = ( kappa_smfp^-alph + kappa_lmfp^-alph )^(-1/alph)
    sigma_m_0_char : float
        cross section normalization per unit mass in characteristic cross section units.
    w_char : float
        velocity scale w in characteristic velocity units.
    smfp_order : int
        Chapman-Enskog order for the normalized SMFP factor.
        See Outmezguine et al. (2023), Appendix B. Options are 1 and 2.
    a, b, c, d : ndarray, shape (N,)
        Output tridiagonal coefficients and RHS, filled in place.
    """
    n           = v2.shape[0] # Number of cells with unknown dv2 values
    sqrt2       = math.sqrt(2.0)

    ### First cell ###
    rL          = r[0]
    rR          = r[1]
    rRR         = r[2]

    drcR        = 0.5 * ( rRR - rL )
    rR2         = rR * rR
    coefR       = rR2 / drcR

    v2C         = v2[0]
    v2R         = v2[1]
    dvR         = v2R - v2C
    T_face      = 0.5 * ( v2C + v2R )

    k_face, slope_face = conductivity(
        T_face,
        rho_int[0],
        sigma_m_0_char,
        w_char,
        alph,
        a_param,
        b_param,
        c_param,
        smfp_order,
    )

    termR = sqrt2 * k_face
    commonR = termR * dvR * slope_face / (2.0 * T_face)
    fluxR = coefR * termR * dvR

    a[0] = 0.0
    b[0] = coefR * (commonR - termR) - sqrt2 * m[1] / dt
    c[0] = coefR * (commonR + termR)
    d[0] = -fluxR

    ### INTERIOR CELLS ###
    # Initialize sliding window
    rL      = rR
    rR      = rRR
    rRR     = r[3]

    drcR    = 0.5 * ( rRR - rL )
    rR2     = rR * rR
    coefL   = coefR
    coefR   = rR2 / drcR

    v2C     = v2R
    v2R     = v2[2]
    dvR     = v2R - v2C
    T_face  = 0.5 * ( v2C + v2R )

    k_face, slope_face = conductivity(
        T_face,
        rho_int[1],
        sigma_m_0_char,
        w_char,
        alph,
        a_param,
        b_param,
        c_param,
        smfp_order,
    )

    termL   = termR
    termR = sqrt2 * k_face
    commonL = commonR
    commonR = termR * dvR * slope_face / (2.0 * T_face)
    fluxL   = fluxR
    fluxR = coefR * termR * dvR

    for j in range(1, n - 1):
        a[j] = -coefL * (commonL - termL)
        b[j] = (
            coefR * (commonR - termR)
            - coefL * (commonL + termL)
            - sqrt2 * (m[j + 1] - m[j]) / dt
        )
        c[j] = coefR * (commonR + termR)
        d[j] = fluxL - fluxR

        # Advance sliding window
        if j < n - 2:
            rL      = rR
            rR      = rRR
            rRR     = r[j + 3]

            drcR    = 0.5 * ( rRR - rL )
            rR2     = rR * rR
            coefL   = coefR
            coefR   = rR2 / drcR

            v2C     = v2R
            v2R     = v2[j + 2]
            dvR     = v2R - v2C
            T_face  = 0.5 * ( v2C + v2R )

            k_face, slope_face = conductivity(
                T_face,
                rho_int[j + 1],
                sigma_m_0_char,
                w_char,
                alph,
                a_param,
                b_param,
                c_param,
                smfp_order,
            )

            termL   = termR
            termR = sqrt2 * k_face
            commonL = commonR
            commonR = termR * dvR * slope_face / (2.0 * T_face)
            fluxL   = fluxR
            fluxR = coefR * termR * dvR

    ### Last cell ###
    a[n - 1] = - coefR * ( commonR - termR)
    b[n - 1] = - coefR * ( commonR + termR ) - sqrt2 * ( m[n] - m[n - 1] ) / dt
    c[n - 1] = 0.0
    d[n - 1] = fluxR

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, types.int64, float64), cache=True)
def conduct_implicit_nolim(v2, rho, r, m, dv2, rho_int, dt, a_param, b_param, c_param, sigma_m_0, w_char, smfp_order, alph,):
    """
    Implicit conduction step on v2.

    w_char is w / v_s; smfp_order selects the normalized first- or
    second-order SMFP factor. Use w_char = inf for constant scattering.

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.

    No fractional-change limiter: the caller is responsible for choosing dt.
    Returns (du_max, dt, 0). This wrapper is not used by the default driver.
    """
    N = v2.shape[0]
    du_max = 0.0

    a   = np.empty(N, dtype=np.float64)
    b   = np.empty(N, dtype=np.float64)
    c   = np.empty(N, dtype=np.float64)
    d   = np.empty(N, dtype=np.float64)
    gam = np.empty(N, dtype=np.float64)

    interp_linear_to_interfaces(r, rho, rho_int)

    build_tridiag_system(r, m, rho_int, v2, dt, a_param, b_param, c_param, alph, sigma_m_0, w_char, smfp_order, a, b, c, d,)
    solve_tridiagonal_thomas(a, b, c, d, dv2, gam)

    tiny = _TINY64
    for i in range(N):
        dv2i    = dv2[i]
        v2i     = v2[i]

        denom = v2i if v2i > tiny else tiny
        rat = abs(dv2i) / denom
        if rat > du_max:
            du_max = rat
        v2[i] += dv2i

    return du_max, dt, 0

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, types.int64, float64), cache=True)
def conduct_implicit_tcool_nolim(v2, rho, r, m, dv2, t_cool, rho_int, dt, a_param, b_param, c_param, sigma_m_0, w_char, smfp_order, alph,):
    """
    Implicit conduction step on v2.

    w_char is w / v_s; smfp_order selects the normalized first- or
    second-order SMFP factor. Use w_char = inf for constant scattering.

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.
    t_cool is updated in-place.

    No du/u limiter: accepts the supplied dt and returns du_iter = 0.
    """
    N = v2.shape[0]
    du_max = 0.0

    a   = np.empty(N, dtype=np.float64)
    b   = np.empty(N, dtype=np.float64)
    c   = np.empty(N, dtype=np.float64)
    d   = np.empty(N, dtype=np.float64)
    gam = np.empty(N, dtype=np.float64)

    interp_linear_to_interfaces(r, rho, rho_int)

    build_tridiag_system(r, m, rho_int, v2, dt, a_param, b_param, c_param, alph, sigma_m_0, w_char, smfp_order, a, b, c, d,)
    solve_tridiagonal_thomas(a, b, c, d, dv2, gam)

    tiny = _TINY64
    for i in range(N):
        v2_old = v2[i]
        dv2i = dv2[i]

        denom = v2_old if v2_old > tiny else tiny

        rat = abs(dv2i) / denom
        if rat > du_max:
            du_max = rat

        if abs(dv2i) > tiny:
            t_cool[i] = denom * dt / abs(dv2i)
        else:
            t_cool[i] = np.inf

        v2[i] = v2_old + dv2i

    return du_max, dt, 0

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, types.int64, float64), cache=True)
def conduct_implicit_Theta_nolim(v2, rho, r, m, dv2, Th, rho_int, dt, a_param, b_param, c_param, sigma_m_0, w_char, smfp_order, alph,):
    """
    Implicit conduction step on v2.

    w_char is w / v_s; smfp_order selects the normalized first- or
    second-order SMFP factor. Use w_char = inf for constant scattering.

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place. Also updates Theta in place.

    Theta = (v2 / |dv2dt|) / (a_param * sigma_m_0 * dr / sqrt(v2)).
    Both times are in t_s units; the crossing speed is the 1D dispersion.

    No fractional-change limiter: the caller is responsible for choosing dt.
    Returns (du_max, dt, 0). This wrapper is not used by the default driver.
    """
    N = v2.shape[0]
    du_max = 0.0

    a   = np.empty(N, dtype=np.float64)
    b   = np.empty(N, dtype=np.float64)
    c   = np.empty(N, dtype=np.float64)
    d   = np.empty(N, dtype=np.float64)
    gam = np.empty(N, dtype=np.float64)

    interp_linear_to_interfaces(r, rho, rho_int)

    build_tridiag_system(r, m, rho_int, v2, dt, a_param, b_param, c_param, alph, sigma_m_0, w_char, smfp_order, a, b, c, d,)
    solve_tridiagonal_thomas(a, b, c, d, dv2, gam)

    tiny = _TINY64
    rR = r[0]
    for i in range(N):
        dv2i    = dv2[i]
        absdv2i = abs(dv2i)
        v2i     = v2[i]

        rL = rR
        rR = r[i + 1]
        dr = rR - rL

        denom = v2i if v2i > tiny else tiny
        rat = absdv2i / denom
        if rat > du_max:
            du_max = rat

        if absdv2i > tiny and dr > tiny and v2i > tiny:
            Th[i] = v2i * math.sqrt(v2i) * dt / (a_param * sigma_m_0 * dr * absdv2i)
        else:
            Th[i] = np.inf

        v2[i] += dv2i

    return du_max, dt, 0

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, types.int64, float64, float64, types.int64), cache=True)
def conduct_implicit_dulim(v2, rho, r, m, dv2, rho_int, dt, a_param, b_param, c_param, sigma_m_0, w_char, smfp_order, alph, eps_du, max_iter,):
    """
    Implicit conduction step on v2.

    w_char is w / v_s; smfp_order selects the normalized first- or
    second-order SMFP factor. Use w_char = inf for constant scattering.
    Repeatedly solves the implicit system with a trial dt until the
    maximum absolute fractional change satisfies

        max_i |dv2_i| / v2_i <= eps_du

    Then updates v2 in place and returns

        (du_max, dt_used, du_iter)

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.
    """
    N = v2.shape[0]

    a   = np.empty(N, dtype=np.float64)
    b   = np.empty(N, dtype=np.float64)
    c   = np.empty(N, dtype=np.float64)
    d   = np.empty(N, dtype=np.float64)
    gam = np.empty(N, dtype=np.float64)

    interp_linear_to_interfaces(r, rho, rho_int)

    tiny = _TINY64
    safety = 0.95

    dt_trial = dt
    du_max = math.inf

    for j in range(max_iter):

        build_tridiag_system(r, m, rho_int, v2, dt_trial, a_param, b_param, c_param, alph, sigma_m_0, w_char, smfp_order, a, b, c, d,)
        solve_tridiagonal_thomas(a, b, c, d, dv2, gam)

        du_max = 0.0
        for i in range(N):
            v2i     = v2[i]

            denom = v2i if v2i > tiny else tiny
            rat = abs(dv2[i]) / denom
            # NaN comparisons are false: explicitly reject invalid solutions.
            if not math.isfinite(rat) or v2i + dv2[i] <= 0.0:
                du_max = math.inf
                break

            if rat > du_max:
                du_max = rat
        
        if du_max <= eps_du:
            for i in range(N):
                v2[i] += dv2[i]
            return du_max, dt_trial, j

        # Saturated implicit responses must still make progress on rejection.
        dt_trial *= (min(0.5, safety * eps_du / du_max) if math.isfinite(du_max) else 0.5)

    return du_max, dt_trial, -1

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, types.int64, float64, float64, types.int64), cache=True)
def conduct_implicit_tcool_dulim(v2, rho, r, m, dv2, t_cool, rho_int, dt, a_param, b_param, c_param, sigma_m_0, w_char, smfp_order, alph, eps_du, max_iter,):
    """
    Implicit conduction step on v2.

    w_char is w / v_s; smfp_order selects the normalized first- or
    second-order SMFP factor. Use w_char = inf for constant scattering.
    Repeatedly solves the implicit system with a trial dt until the
    maximum absolute fractional change satisfies

        max_i |dv2_i| / v2_i <= eps_du

    Then updates v2 in place and returns

        (du_max, dt_used, du_iter)

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.
    """
    N = v2.shape[0]

    a   = np.empty(N, dtype=np.float64)
    b   = np.empty(N, dtype=np.float64)
    c   = np.empty(N, dtype=np.float64)
    d   = np.empty(N, dtype=np.float64)
    gam = np.empty(N, dtype=np.float64)

    interp_linear_to_interfaces(r, rho, rho_int)

    tiny = _TINY64
    safety = 0.95

    dt_trial = dt
    du_max = math.inf

    for j in range(max_iter):

        build_tridiag_system(r, m, rho_int, v2, dt_trial, a_param, b_param, c_param, alph, sigma_m_0, w_char, smfp_order, a, b, c, d,)
        solve_tridiagonal_thomas(a, b, c, d, dv2, gam)

        du_max = 0.0
        for i in range(N):
            v2i     = v2[i]

            denom = v2i if v2i > tiny else tiny
            rat = abs(dv2[i]) / denom
            # NaN comparisons are false: explicitly reject invalid solutions.
            if not math.isfinite(rat) or v2i + dv2[i] <= 0.0:
                du_max = math.inf
                break

            if rat > du_max:
                du_max = rat
        
        if du_max <= eps_du:
            for i in range(N):
                v2_old = v2[i]
                dv2i = dv2[i]

                denom = v2_old if v2_old > tiny else tiny
                if abs(dv2i) > tiny:
                    t_cool[i] = denom * dt_trial / abs(dv2i)
                else:
                    t_cool[i] = np.inf

                v2[i] = v2_old + dv2i
            return du_max, dt_trial, j

        # Saturated implicit responses must still make progress on rejection.
        dt_trial *= (min(0.5, safety * eps_du / du_max) if math.isfinite(du_max) else 0.5)

    return du_max, dt_trial, -1

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, types.int64, float64, float64, types.int64), cache=True)
def conduct_implicit_Theta_dulim(v2, rho, r, m, dv2, Th, rho_int, dt, a_param, b_param, c_param, sigma_m_0, w_char, smfp_order, alph, eps_du, max_iter,):
    """
    Implicit conduction step on v2.

    w_char is w / v_s; smfp_order selects the normalized first- or
    second-order SMFP factor. Use w_char = inf for constant scattering.
    Repeatedly solves the implicit system with a trial dt until the
    maximum absolute fractional change satisfies

        max_i |dv2_i| / v2_i <= eps_du

    Then updates v2 in place and returns

        (du_max, dt_used, du_iter).

    Also updates Theta in place.

    Theta = (v2 / |dv2dt|) / (a_param * sigma_m_0 * dr / sqrt(v2)).
    Both times are in t_s units; the crossing speed is the 1D dispersion.

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.
    """
    N = v2.shape[0]

    a   = np.empty(N, dtype=np.float64)
    b   = np.empty(N, dtype=np.float64)
    c   = np.empty(N, dtype=np.float64)
    d   = np.empty(N, dtype=np.float64)
    gam = np.empty(N, dtype=np.float64)

    interp_linear_to_interfaces(r, rho, rho_int)

    tiny = _TINY64
    safety = 0.95

    dt_trial = dt
    du_max = math.inf

    for j in range(max_iter):

        build_tridiag_system(r, m, rho_int, v2, dt_trial, a_param, b_param, c_param, alph, sigma_m_0, w_char, smfp_order, a, b, c, d,)
        solve_tridiagonal_thomas(a, b, c, d, dv2, gam)

        du_max = 0.0
        for i in range(N):
            v2i     = v2[i]

            denom = v2i if v2i > tiny else tiny
            rat = abs(dv2[i]) / denom
            # NaN comparisons are false: explicitly reject invalid solutions.
            if not math.isfinite(rat) or v2i + dv2[i] <= 0.0:
                du_max = math.inf
                break

            if rat > du_max:
                du_max = rat
        
        if du_max <= eps_du:
            rR = r[0]
            for i in range(N):
                rL = rR
                rR = r[i + 1]
                dr = rR - rL

                v2i     = v2[i]
                dv2i    = dv2[i]
                absdv2i = abs(dv2i)

                if absdv2i > tiny and dr > tiny and v2i > tiny:
                    Th[i] = v2i * math.sqrt(v2i) * dt_trial / (a_param * sigma_m_0 * dr * absdv2i)
                else:
                    Th[i] = np.inf

                v2[i] += dv2i
            return du_max, dt_trial, j

        # Bound reduction even when the implicit response has saturated.
        fac = (min(0.5, safety * eps_du / du_max) if math.isfinite(du_max) else 0.5)
        dt_trial *= fac

    return du_max, dt_trial, -1
