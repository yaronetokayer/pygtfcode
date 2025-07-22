import numpy as np

def compute_luminosities(state):
    """ 
    Compute luminosity of each shell interface based on temperature gradient and conductivity.
    e.g, Eq. (2) in Nishikawa et al. 2020.

    Parameters
    ----------
    state : State
        The current simulation state.

    Returns
    -------
    lum : np.ndarray
        Array of luminosities evaluated at each radial grid point (same length as state.r).
    """
    lum = np.empty_like(state.r)

    a = state.config.sim.a
    b = state.config.sim.b
    c = state.config.sim.c
    sigma_m_sq = state.char.sigma_m_char**2

    # Compute temperature gradient and midpoints using cell-centered values
    dTdr = ( state.v2[1:] - state.v2[:-1] ) / ( state.r[2:] - state.r[:-2] )

    # One sided difference for cored profiles (i.e., ABG with gamma < 1)
    if state.config.init.profile == 'abg' and state.config.init.gamma < 1.0:
        dTdr[0] = (state.v2[1] - state.v2[0]) / (state.r[2] - state.r[1])

    vmed = np.sqrt(0.5 * state.v2[1:] + state.v2[:-1] )
    pmed = 0.5 * ( state.p[1:] + state.p[:-1] )

    fac1 = -3.0 * vmed * state.r[1:-1]**2
    fac2 = (a / b) * sigma_m_sq + (1.0 / c) / pmed

    lum[1:-1] = (fac1 / fac2) * dTdr

    # Boundary conditions
    lum[0] = 0.0
    lum[-1] = 0.0

    return lum

def conduct_heat(state, lum):
    """
    Conduct heat and adjust internal energies accordingly.
    Ignores PdV work and assumes fixed density.
    Updates internal energy and recomputes pressure.

    Parameters
    ----------
    state : State
        The current simulation state.
    lum : np.ndarray
        Array of luminosities from compute_luminosities (length = len(state.r))

    Returns
    -------
    u : np.ndarray
        Updated internal energy array.
    p : np.ndarray
        Updated pressure array.
    """

    dudt = -( lum[1:] - lum[:-1] ) / (state.m[1:] - state.m[: -1])
    du = dudt * state.dt

    u = state.u + du
    p = ( 2 / 3 ) * state.rho * u

    # Track max relative change in u for timestep control
    dumax = np.max(np.abs(du / u))

    return u, p, dumax