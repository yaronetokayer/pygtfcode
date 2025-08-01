import numpy as np
from numba import njit

@njit
def compute_luminosities(a, b, c, sigma_m, r, v2, p, cored) -> np.ndarray:
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
    r : ndarray
        Radial grid points, including cell edges (length = ngrid + 1).
    v2 : ndarray
        Velocity dispersion squared for each cell (length = ngrid).
    p : ndarray
        Pressure for each cell (length = ngrid).
    cored : bool
        Whether the system has a central core (i.e., ABG with gamma < 1.0).

    Returns
    -------
    lum : ndarray
        Luminosities at each shell boundary (same length as r).
    """
    lum = np.empty_like(r)

    # Compute temperature gradient and midpoints using cell-centered values
    dTdr = ( v2[1:] - v2[:-1] ) / ( r[2:] - r[:-2] )
    vmed = np.sqrt( 0.5 * ( v2[1:] + v2[:-1] ) )
    pmed = 0.5 * ( p[1:] + p[:-1] )

    # One sided difference for cored profiles (i.e., ABG with gamma < 1)
    if cored:
        dTdr[0] = (v2[1] - v2[0]) / (r[2] - r[1])
        vmed[0] = np.sqrt(v2[0])
        pmed[0] = p[0]

    fac1 = -3.0 * vmed * r[1:-1]**2
    fac2 = (a / b) * sigma_m**2 + ( (1.0 / c) / pmed )

    lum[1:-1] = (fac1 / fac2) * dTdr

    # Boundary conditions
    lum[0] = 0.0
    lum[-1] = 0.0

    return lum

@njit
def conduct_heat(m, u, rho, lum, dt) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Conduct heat and adjust internal energies accordingly.
    Ignores PdV work and assumes fixed density.
    Updates internal energy and recomputes pressure.

    Arguments
    ---------
    m : np.ndarray
        Enclosed mass array
    u : np.ndarray
        Internal energy array
    rho : np.ndarray
        Density array
    lum : np.ndarray
        Array of luminosities from compute_luminosities (length = len(state.r))
    dt : float
        Current timestep duration

    Returns
    -------
    u : np.ndarray
        Updated internal energy array.
    p : np.ndarray
        Updated pressure array.
    dumax : float
        Max relative change in u
    """

    dudt = -( lum[1:] - lum[:-1] ) / ( m[1:] - m[:-1] )
    du = dudt * dt

    u += du
    p = ( 2 / 3 ) * rho * u

    # Track max relative change in u for timestep control
    dumax = np.max(np.abs(du / u))

    return p, dumax