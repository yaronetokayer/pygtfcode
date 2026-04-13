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
                dTdr = ( v2[1] - v2[0] ) / ( r[2] - r[1] )
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
