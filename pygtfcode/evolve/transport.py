import numpy as np
from numba import njit, float64, boolean, types, void
from pygtfcode.util.interpolate import interp_linear_to_interfaces

@njit(void(float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:], boolean), cache=True, fastmath=True)
def compute_luminosities(a, b, c, sigma_m, r, v2, rho, lum, cored): # In place version
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
    floor = 1e-40
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

    ###

    dudt = -( lum[1:] - lum[:-1] ) / ( m[1:] - m[:-1] )
    du = dudt * dt_prop

    tiny = np.finfo(np.float64).tiny
    abs_u = np.abs(u)
    abs_u[abs_u < tiny] = tiny
    dumax = np.max(np.abs(du) / abs_u)

    if dumax > eps_du:
        scale = 0.95 * (eps_du / dumax)
        dt_eff = dt_prop * scale
        dumax *= scale
        du *= scale
    else:
        dt_eff = dt_prop

    u_new = u + du
    p_new = ( 2.0 / 3.0 ) * rho * u_new

    return p_new, float(dumax), float(dt_eff)

### IMPLICIT SCHEME

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64), cache=True, fastmath=True)
def build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt):
    """
    Construct tridiagonal coefficients: a_i du_i-1 + b_i du_i + c_i du_i+1 = d_i
    a, b, c, and d are updated in place

    Arguments
    ---------
    a : ndarray, shape (N,)
        Subdiagonal coefficients (multiply du_{i-1}) for interior nodes j=1..N.
    b : ndarray, shape (N,)
        Main diagonal coefficients (multiply du_i) for interior nodes.
    c : ndarray, shape (N,)
        Superdiagonal coefficients (multiply d_{u+1}) for interior nodes.
    d : ndarray, shape (N,)
        Right-hand side vector for the interior nodes.
    rk : ndarray, shape (N+1,)
        Edge radii.
    mk : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rhok_int : ndarray, shape (N-1,)
        Densities interpolated to shell edges
    uk : ndarray, shape (N,)
        Specific internal energy.
    pref : float
        prefactor for species k
    dt : float
        timestep
    """
    drc     = 0.5 * (rk[2:] - rk[:-2])      # (N-1,)
    delu    = uk[1:] - uk[:-1]              # (N-1,)
    su      = uk[1:] + uk[:-1]              # (N-1,)
    sqrt2   = 1.41421356237309

    # Interior cells
    facL    = rhok_int[:-1] * rk[1:-2]**2 / drc[:-1]
    facR    = rhok_int[1:] * rk[2:-1]**2 / drc[1:]
    su12L   = 1 / np.sqrt(su[:-1])
    su12R   = 1 / np.sqrt(su[1:])
    dusu32L = 0.5 * delu[:-1] / su[:-1]**(3.0/2.0)
    dusu32R = 0.5 * delu[1:] / su[1:]**(3.0/2.0)

    a[1:-1] = facL * ( su12L + dusu32L )
    b[1:-1] = -1 * (
        facR * ( su12R + dusu32R )
        + facL * ( su12L - dusu32L )
        + ( ( mk[2:-1] - mk[1:-2] ) / ( sqrt2 * pref * dt ) )
    )
    c[1:-1] = facR * ( su12R - dusu32R )
    d[1:-1] = (
        facL * delu[:-1] / np.sqrt(su[:-1])
        - facR * delu[1:] / np.sqrt(su[1:])
    )

    # i = 1
    a[0] = 0.0
    b[0] = -1 *  (
        su12L[0] + dusu32L[0]
        + ( mk[1] * drc[0] / ( rhok_int[0] * rk[1]**2 * pref * sqrt2 * dt ) )
    )
    c[0] = su12L[0] - dusu32L[0]
    d[0] = - delu[0] / np.sqrt(su[0])

    # i = N
    a[-1] = su12R[-1] + dusu32R[-1]
    b[-1] = (
        dusu32R[-1] - su12R[-1]
        - ( 
            (mk[-1] - mk[-2]) * drc[-1] 
            / ( rhok_int[-1] * rk[-2]**2 * pref * sqrt2 * dt )
        )
    )
    c[-1] = 0.0
    d[-1] = delu[-1] / np.sqrt(su[-1])

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64, float64[:], float64[:, :], float64[:, :], float64), cache=True, fastmath=True)
def conduct_implicit(v2, rho, r, m, c2, mrat, lnL, du_trial, dt,):
    """
    Implicit intra-species conduction step on v2.
    Use a fixed dt - no timestep limiting in this step - we find that the hex step limits in almost all cases.

    For each species, solve a tridiagonal system for du, but only commit the
    update once the limiter is satisfied.

    The tridiagonal system is defined by:
        a_i du_i-1 + b_i du_i + c_i du_i+1 = d_i
    
    v2 is updated in-place. u = 1.5 * v2
    """
    s, N = v2.shape

    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    d = np.empty(N, dtype=np.float64)

    for k in range(s):
        rk       = r[k]
        mk       = m[k]
        rhok     = rho[k]
        uk       = 1.5 * v2[k]
        rhok_int = interp_linear_to_interfaces(rk, rhok)

        pref = c2 * (mrat[k] * lnL[k, k])

        build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt)
        solve_tridiagonal_thomas(a, b, c, d, du_trial[k])

    for k in range(s):
        for i in range(N):
            v2[k, i] += (2.0 / 3.0) * du_trial[k, i]