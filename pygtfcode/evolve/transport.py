import numpy as np
import math
from numba import njit, float64, boolean, types, void
from pygtfcode.util.interpolate import interp_linear_to_interfaces
from pygtfcode.util.calc import solve_tridiagonal_thomas

_TINY64 = np.finfo(np.float64).tiny

# cored = (init.profile == 'abg') and (float(init.gamma) < 1.0) # Leftover from integration loop when we used this function

@njit(void(float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:], boolean), cache=True, fastmath=True)
def compute_luminosities(a, b, c, sigma_m, r, v2, rho, lum, cored): 
    """ 
    Compute luminosity of each shell interface based on temperature gradient and conductivity.
    e.g, Eq. (2) in Nishikawa et al. 2020.

    Arguments
    ----------
    a : float
        Constant 'a' in the conductivity formula.
    b : float
        Constant 'b' in the conductivity formula.
    c : float
        Constant 'c' in the conductivity formula.
    sigma_m : float
        Interaction cross section in dimensionless units.
    r : ndarray (N+1,)
        Radial grid points, including cell edges.
    v2 : ndarray (N,)
        Velocity dispersion squared for each cell.
    rho : ndarray (N,)
        Density for each cell.
    lum : ndarray (N+1,)
        Luminosities at each shell boundary. Updated in-place.
    cored : bool
        Whether the system has a central core (i.e., ABG with gamma < 1.0).
    """
    Np1 = r.shape[0]
    N = Np1 - 1

    # Boundary conditions
    lum[0] = 0.0
    lum[N] = 0.0

    # Compute rho and v2 at interfaces (N-1,)
    v2int  = interp_linear_to_interfaces(r, v2)
    rhoint = interp_linear_to_interfaces(r, rho)

    smfp_term = (a / b) * sigma_m**2

    for i in range(N-1):
        # One sided difference for cored profiles (i.e., ABG with gamma < 1)
        if i == 0 and cored:
                dTdr = 0.5 * ( v2[1] - v2[0] ) / ( r[2] - r[1] ) # Need 0.5 to account for fact that this is not two-sided central difference in denom
        else:
            dTdr = ( v2[i+1] - v2[i] ) / ( r[i+2] - r[i] )

        coeff = -3.0 * np.sqrt(v2int[i]) * r[i+1]**2
        lmfp_term = 1.0 / ( c * rhoint[i] * v2int[i] )

        lum[i + 1] = coeff * dTdr / ( smfp_term + lmfp_term )

@njit(types.Tuple((float64, float64))(float64[:], float64[:], float64[:], float64[:], float64, float64), cache=True, fastmath=True)
def conduct_heat(v2, m, lum, dv2dt, dt_prop, eps_du) -> tuple[float, float]:
    """
    Conduct heat and adjust internal energies accordingly.
    Ignores PdV work and assumes fixed density.
    Updates internal energy and recomputes pressure.
    Updates dt_prop if necessary based on max relative change in u.

    Arguments
    ---------
    v2 : np.ndarray (N,)
        Sqaure of 1D velocity dispersion. u = 1.5*v2.
    m : np.ndarray (N+1,)
        Enclosed mass array
    lum : np.ndarray (N+1,)
        Array of luminosities from compute_luminosities (length = len(state.r))
    dv2dt : np.ndarray (N,)
        Preallocated working array for storage of dv2dt
    dt_prop : float
        Current timestep duration
    eps_du : float
        Maximum allowed relative change in u for convergence

    Returns
    -------
    dumax : float
        Max relative change in u.
    dt_prop : float
        Modified timestep.
    """
    Np1 = m.shape[0]
    N = Np1 - 1

    for i in range(N):
        dm = m[i+1] - m[i]
        dv2dt[i] = -(2.0 / 3.0) * (lum[i+1] - lum[i]) / dm

    # Find maximum relative proposed change
    floor = _TINY64
    dv2max = 0.0
    for i in range(N):
        dv2 = dv2dt[i] * dt_prop

        denom = abs(v2[i])
        if denom < floor:
            denom = floor

        rat = abs(dv2) / denom
        if rat > dv2max:
            dv2max = rat

    # Adaptive limiter
    if dv2max > eps_du:
        scale = 0.95 * (eps_du / dv2max)
        dv2max *= scale
        dt_eff = dt_prop * scale
    else:
        dt_eff = dt_prop

    # Apply update in place
    for i in range(N):
        v2[i] += dv2dt[i] * dt_eff

    return float(dv2max), float(dt_eff)

### IMPLICIT SCHEME

@njit(void(float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system(r, m, rho_int, v2, Csmfp, Clmfp, dt, alph, a, b, c, d,):
    """
    Build tridiagonal system for implicit conduction update in v2:

        a_i dv2_{i-1} + b_i dv2_i + c_i dv2_{i+1} = d_i

    Assumes n > 3.  Can take arbitrary alpha, see below.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii.
    m : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rho_int : ndarray, shape (N-1,)
        Interface values of rhok.
    v2 : ndarray, shape (N,)
        Cell-centered v2 for one species.
    Csmfp, Clmfp : float
        Conductivity coefficients.
    dt : float
        Timestep.
    alph : float
        Coefficient for interpolation scheme between lmfp and smfp regimes.
        kappa = ( kappa_smfp^-alph + kappa_lmfp^-alph )^(-1/alph)
    a, b, c, d : ndarray, shape (N,)
        Output tridiagonal coefficients and RHS, filled in place.
    """
    n           = v2.shape[0] # Number of cells with unknown dv2 values
    sqrt2       = math.sqrt(2.0)
    two_Clmfp   = 2.0 * Clmfp
    inv_alph    = 1.0 / alph

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
    svR         = v2C + v2R
    sqrt_svR    = math.sqrt(svR)
    svR32       = sqrt_svR * svR

    rhofacR         = ( two_Clmfp / rho_int[0] )**alph
    denombaseR      = ( Csmfp * svR )**alph + rhofacR
    inv_denombaseR  = 1.0 / denombaseR
    inv_denomR1     = inv_denombaseR**inv_alph
    inv_denomR2     = inv_denomR1 * inv_denombaseR

    tmpR        = sqrt_svR * dvR
    commonR     = tmpR * ( 0.5 * inv_denomR1 + rhofacR * inv_denomR2 )
    termR       = svR32 * inv_denomR1
    fluxR       = coefR * svR32 * dvR * inv_denomR1

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

    dvR         = v2R - v2C
    svR         = v2C + v2R
    sqrt_svR    = math.sqrt(svR)
    svR32       = sqrt_svR * svR

    rhofacR         = ( two_Clmfp / rho_int[1] )**alph
    denombaseR      = ( Csmfp * svR )**alph + rhofacR
    inv_denombaseR  = 1.0 / denombaseR
    inv_denomR1     = inv_denombaseR**inv_alph
    inv_denomR2     = inv_denomR1 * inv_denombaseR
    
    commonL = commonR
    tmpR    = sqrt_svR * dvR
    commonR = tmpR * (0.5 * inv_denomR1 + rhofacR * inv_denomR2 )
    termL   = termR
    termR   = svR32 * inv_denomR1
    fluxL   = fluxR
    fluxR   = coefR * svR32 * dvR * inv_denomR1

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

            dvR         = v2R - v2C
            svR         = v2C + v2R
            sqrt_svR    = math.sqrt(svR)
            svR32       = sqrt_svR * svR

            rhofacR         = ( two_Clmfp / rho_int[j + 1] )**alph
            denombaseR      = ( Csmfp * svR )**alph + rhofacR
            inv_denombaseR  = 1.0 / denombaseR
            inv_denomR1     = inv_denombaseR**inv_alph
            inv_denomR2     = inv_denomR1 * inv_denombaseR

            commonL     = commonR
            tmpR        = sqrt_svR * dvR
            commonR     = tmpR * ( 0.5 * inv_denomR1 + rhofacR * inv_denomR2 )
            termL       = termR
            termR       = svR32 * inv_denomR1
            fluxL       = fluxR
            fluxR       = coefR * svR32 * dvR * inv_denomR1

    ### Last cell ###
    a[n - 1] = - coefR * ( commonR - termR)
    b[n - 1] = - coefR * ( commonR + termR ) - sqrt2 * ( m[n] - m[n - 1] ) / dt
    c[n - 1] = 0.0
    d[n - 1] = fluxR

@njit(void(float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system_ALPH1(r, m, rho_int, v2, Csmfp, Clmfp, dt, a, b, c, d,):
    """
    Build tridiagonal system for implicit conduction update in v2:

        a_i dv2_{i-1} + b_i dv2_i + c_i dv2_{i+1} = d_i

    Assumes n > 3.  This is for alpha=1.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii.
    m : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rho_int : ndarray, shape (N-1,)
        Interface values of rhok.
    v2 : ndarray, shape (N,)
        Cell-centered v2 for one species.
    Csmfp, Clmfp : float
        Conductivity coefficients.
    dt : float
        Timestep.
    a, b, c, d : ndarray, shape (N,)
        Output tridiagonal coefficients and RHS, filled in place.
    """
    n = v2.shape[0] # Number of cells with unknown dv2 values
    sqrt2 = math.sqrt(2.0)
    two_Clmfp = 2.0 * Clmfp

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
    svR         = v2C + v2R
    sqrt_svR    = math.sqrt(svR)
    svR32       = sqrt_svR * svR

    inv_rhoR    = 1.0 / rho_int[0]
    denomR      = Csmfp * svR + two_Clmfp * inv_rhoR
    inv_denomR  = 1.0 / denomR
    inv_denomR2 = inv_denomR * inv_denomR

    tmpR        = sqrt_svR * dvR
    commonR     = tmpR * (0.5 * inv_denomR + two_Clmfp * inv_denomR2 * inv_rhoR)
    termR       = svR32 * inv_denomR
    fluxR       = coefR * svR32 * dvR * inv_denomR

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

    dvR = v2R - v2C
    svR = v2C + v2R
    sqrt_svR = math.sqrt(svR)
    svR32 = sqrt_svR * svR

    inv_rhoR = 1.0 / rho_int[1]
    denomR = Csmfp * svR + two_Clmfp * inv_rhoR
    inv_denomR = 1.0 / denomR
    inv_denomR2 = inv_denomR * inv_denomR
    
    commonL = commonR
    tmpR = sqrt_svR * dvR
    commonR = tmpR * (0.5 * inv_denomR + two_Clmfp * inv_denomR2 * inv_rhoR)
    termL = termR
    termR = svR32 * inv_denomR
    fluxL = fluxR
    fluxR = coefR * svR32 * dvR * inv_denomR

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

            dvR         = v2R - v2C
            svR         = v2C + v2R
            sqrt_svR    = math.sqrt(svR)
            svR32       = sqrt_svR * svR

            inv_rhoR    = 1.0 / rho_int[j + 1]
            denomR      = Csmfp * svR + two_Clmfp * inv_rhoR
            inv_denomR  = 1.0 / denomR
            inv_denomR2 = inv_denomR * inv_denomR

            commonL     = commonR
            tmpR        = sqrt_svR * dvR
            commonR     = tmpR * (0.5 * inv_denomR + two_Clmfp * inv_denomR2 * inv_rhoR)
            termL       = termR
            termR       = svR32 * inv_denomR
            fluxL       = fluxR
            fluxR       = coefR * svR32 * dvR * inv_denomR

    ### Last cell ###
    a[n - 1] = - coefR * ( commonR - termR)
    b[n - 1] = - coefR * ( commonR + termR ) - sqrt2 * ( m[n] - m[n - 1] ) / dt
    c[n - 1] = 0.0
    d[n - 1] = fluxR

@njit(void(float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system_VEC(r, m, rho_int, v2, Csmfp, Clmfp, dt, a, b, c, d,):
    """
    Build tridiagonal system for implicit conduction update in v2:

        a_i dv2_{i-1} + b_i dv2_i + c_i dv2_{i+1} = d_i

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii.
    m : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rho_int : ndarray, shape (N-1,)
        Interface values of rhok.
    v2 : ndarray, shape (N,)
        Cell-centered v2 for one species.
    Csmfp, Clmfp : float
        Conductivity coefficients.
    dt : float
        Timestep.
    a, b, c, d : ndarray, shape (N,)
        Output tridiagonal coefficients and RHS, filled in place.
    """
    N = v2.shape[0]

    drc = 0.5 * (r[2:] - r[:-2])     # (N-1,)
    dv  = v2[1:] - v2[:-1]           # (N-1,)
    sv  = v2[1:] + v2[:-1]           # (N-1,)
    r2  = r[1:-1] ** 2               # (N-1,)
    dm  = m[1:] - m[:-1]             # (N,)

    denom = Csmfp * sv + 2.0 * Clmfp / rho_int
    sqrt_sv = np.sqrt(sv)
    base  = (r2 / drc) * (sv * sqrt_sv)
    flux0 = base * dv / denom

    common = (r2 / drc) * (
        0.5 * sqrt_sv * dv / denom
        + 2.0 * Clmfp * sqrt_sv * dv / (denom * denom * rho_int)
    )

    left_coef  = common - base / denom
    right_coef = common + base / denom

    # interior rows
    a[1:-1] = -left_coef[:-1]
    b[1:-1] = left_coef[1:] - right_coef[:-1] - math.sqrt(2.0) * dm[1:-1] / dt
    c[1:-1] = right_coef[1:]
    d[1:-1] = flux0[:-1] - flux0[1:]

    # inner boundary
    a[0] = 0.0
    b[0] = left_coef[0] - math.sqrt(2.0) * dm[0] / dt
    c[0] = right_coef[0]
    d[0] = -flux0[0]

    # outer boundary
    a[-1] = -left_coef[-1]
    b[-1] = -right_coef[-1] - math.sqrt(2.0) * dm[-1] / dt
    c[-1] = 0.0
    d[-1] = flux0[-1]

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64), cache=True, fastmath=True)
def conduct_implicit_nolim(v2, rho, r, m, dv2, dt, a_param, b_param, c_param, sigma_m, alph,):
    """
    Implicit conduction step on v2.

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.

    No limit on maximum fractional change - assumes we are being limited by 
    relaxation time criterion
    """
    N = v2.shape[0]
    du_max = 0.0

    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    d = np.empty(N, dtype=np.float64)

    Csmfp = a_param * sigma_m**2 / b_param
    Clmfp = 1.0 / c_param

    rho_int = interp_linear_to_interfaces(r, rho)

    build_tridiag_system(r, m, rho_int, v2, Csmfp, Clmfp, dt, alph, a, b, c, d,)
    solve_tridiagonal_thomas(a, b, c, d, dv2)

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

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64), cache=True, fastmath=True)
def conduct_implicit_Theta_nolim(v2, rho, r, m, dv2, Th, dt, a_param, b_param, c_param, sigma_m, alph,):
    """
    Implicit conduction step on v2.

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place. Also updates Theta in place.

    Theta = (v2 / |dv2dt|)/(dr / sqrt(v2)), local cooling-to-sound-crossing time ratio

    No limit on maximum fractional change - assumes we are being limited by 
    relaxation time criterion
    """
    N = v2.shape[0]
    du_max = 0.0

    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    d = np.empty(N, dtype=np.float64)

    Csmfp = a_param * sigma_m**2 / b_param
    Clmfp = 1.0 / c_param

    rho_int = interp_linear_to_interfaces(r, rho)

    build_tridiag_system(r, m, rho_int, v2, Csmfp, Clmfp, dt, alph, a, b, c, d,)
    solve_tridiagonal_thomas(a, b, c, d, dv2)

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
            Th[i] = v2i * math.sqrt(v2i) * dt / (dr * absdv2i)
        else:
            Th[i] = np.inf

        v2[i] += dv2i

    return du_max, dt, 0

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64 , float64, types.int64), cache=True, fastmath=True)
def conduct_implicit_dulim(v2, rho, r, m, dv2, dt, a_param, b_param, c_param, sigma_m, alph, eps_du, max_iter,):
    """
    Implicit conduction step on v2.
    Repeatedly solves the implicit system with a trial dt until the
    maximum absolute fractional change satisfies

        max_i |dv2_i| / v2_i <= eps_du

    Then updates v2 in place and returns

        (du_max, dt_used)

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.
    """
    N = v2.shape[0]

    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    d = np.empty(N, dtype=np.float64)

    Csmfp = a_param * sigma_m**2 / b_param
    Clmfp = 1.0 / c_param

    rho_int = interp_linear_to_interfaces(r, rho)

    tiny = _TINY64
    safety = 0.95

    dt_trial = dt

    for j in range(max_iter):

        build_tridiag_system(r, m, rho_int, v2, Csmfp, Clmfp, dt_trial, alph, a, b, c, d,)
        solve_tridiagonal_thomas(a, b, c, d, dv2)

        du_max = 0.0
        for i in range(N):
            v2i     = v2[i]

            denom = v2i if v2i > tiny else tiny
            rat = abs(dv2[i]) / denom

            if rat > du_max:
                du_max = rat
        
        if du_max <= eps_du:
            for i in range(N):
                v2[i] += dv2[i]
            return du_max, dt_trial, j

        dt_trial *= safety * eps_du / du_max

    return du_max, dt_trial, -1

@njit(types.Tuple((float64, float64, types.int64))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64 , float64, types.int64), cache=True, fastmath=True)
def conduct_implicit_Theta_dulim(v2, rho, r, m, dv2, Th, dt, a_param, b_param, c_param, sigma_m, alph, eps_du, max_iter,):
    """
    Implicit conduction step on v2.
    Repeatedly solves the implicit system with a trial dt until the
    maximum absolute fractional change satisfies

        max_i |dv2_i| / v2_i <= eps_du

    Then updates v2 in place and returns

        (du_max, dt_used).

    Also updates Theta in place.

    Theta = (v2 / |dv2dt|)/(dr / sqrt(v2)), local cooling-to-sound-crossing time ratio

    The tridiagonal system is defined by:
        a_i dv2_i-1 + b_i dv2_i + c_i dv2_i+1 = d_i
    
    v2 is updated in-place.
    """
    N = v2.shape[0]

    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    d = np.empty(N, dtype=np.float64)

    Csmfp = a_param * sigma_m**2 / b_param
    Clmfp = 1.0 / c_param

    rho_int = interp_linear_to_interfaces(r, rho)

    tiny = _TINY64
    safety = 0.95

    dt_trial = dt

    for j in range(max_iter):

        build_tridiag_system(r, m, rho_int, v2, Csmfp, Clmfp, dt_trial, alph, a, b, c, d,)
        solve_tridiagonal_thomas(a, b, c, d, dv2)

        du_max = 0.0
        for i in range(N):
            v2i     = v2[i]

            denom = v2i if v2i > tiny else tiny
            rat = abs(dv2[i]) / denom

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
                    Th[i] = v2i * math.sqrt(v2i) * dt_trial / (dr * absdv2i)
                else:
                    Th[i] = np.inf

                v2[i] += dv2i
            return du_max, dt_trial, j

        dt_trial *= safety * eps_du / du_max

    return du_max, dt_trial, -1