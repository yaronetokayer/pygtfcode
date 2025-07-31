import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from numba import njit
import numpy as np
from scipy.integrate import quad

@njit
def df(e):
    """
    Full (untruncated) distribution function f(e) for the NFW profile.

    Parameters
    ----------
    e : float or ndarray
        Energy variable (0 < e < 1)

    Returns
    -------
    float or ndarray
        Value of the distribution function.
    """
    F0 = 9.1968e-2
    q = -2.7419
    p1, p2, p3, p4 = 0.3620, -0.5639, -0.0859, -0.4912

    e2 = 1.0 - e
    pp = p1 * e + p2 * e**2 + p3 * e**3 + p4 * e**4

    fac1 = e**1.5 / e2**2.5
    fac2 = (-np.log(e) / e2)**q
    fac3 = np.exp(pp)

    return F0 * fac1 * fac2 * fac3

@njit
def integrand_for_rho(z, phi, Zt, Ft):
    if z > phi:
        return 0.0
    e = z + Zt
    return (df(e) - Ft) * np.sqrt(2.0 * (phi - z))

def rho(phi, config):
    """
    Compute density rho(phi) using Eddington-like inversion of the DF.

    Parameters
    ----------
    phi : float
        Gravitational potential at a radius.
    Zt : float
        Truncation energy parameter.

    Returns
    -------
    rho : float
        Density corresponding to the given potential.
    """

    Zt = config.init.Zt
    Ft = df(Zt)

    epsabs = config.prec.epsabs
    epsrel = config.prec.epsrel

    result, _ = quad(integrand_for_rho, 0.0, phi, args=(phi, Zt, Ft), epsabs=epsabs, epsrel=epsrel)
    
    return 4.0 * np.pi * result

def generate_rho_lookup(config, n_points=10000, phi_min=1e-7):
    """
    Generate an interpolated function for rho(phi) for use in later functions.

    Parameters
    ----------
    config : Config
        Simulation configuration containing init and prec parameters.
    phi_max : float
        Maximum potential value to consider.
    n_points : int
        Number of points used to generate the interpolated function.

    Returns
    -------
    rho_interp : interp1d
        Interpolated function for rho(phi).
    """
    if config.io.chatter:
        print("Generating lookup for rho(phi)...")

    phi_max = 1.0 - config.init.Zt - 1e-4

    # Precompute rho over a grid of phi
    phi_grid = np.linspace(phi_min, phi_max, n_points)  # choose phi_max ~ initial phi0
    rho_grid = np.array([rho(phi, config) for phi in phi_grid])
    rho_interp = interp1d(phi_grid, rho_grid, bounds_error=False, fill_value=0.0)

    return rho_interp

def df_trunc(e, Zt, Ft):
    """
    Truncated distribution function f_trunc(e) = f(e + Zt) - Ft.

    Parameters
    ----------
    e : float or ndarray
        Energy variable for integration, in [0, phi]
    Zt : float
        Energy shift defining the truncation threshold.
    Ft : float
        Distribution function floor used in truncation.

    Returns
    -------
    float or ndarray
        Truncated distribution function.
    """
    return df(e + Zt) - Ft

@njit
def potential(r, Zt):
    """
    Gravitational potential for an NFW profile.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    Zt : float
        Truncation energy parameter.

    Returns
    -------
    phi : float or ndarray
        Potential at radius r (dimensionless).
    """
    return np.log(1.0 + r) / r - Zt

def integrate_potential(config, rho_interp):
    """
    Computes the potential profile for a truncated NFW halo by integrating outward
    until the potential drops to zero. Also determines the outer truncation radius.

    Parameters
    ----------
    config : Config
        Simulation configuration.
    rho_interp : interp1d
        Interpolated function for rho(phi).

    Returns
    -------
    rcut : float
        Truncation radius (in units of r_s).
    rmax_new : float
        Updated value of grid.rmax (ensures rmax < rcut).
    pot_interp : interp1d
        Interpolated potential function.
    rad : ndarray
        Radial grid points where potential was computed.
    pot_vals : ndarray
        Corresponding potential values at those radial points.
    """
    chatter = config.io.chatter
    init = config.init

    if chatter:
        print("Computing potential profile for truncated NFW halo...")
    r_min = config.grid.rmin / 2
    eps = 1e-6  # A small number used for finite differences
    Zt = init.Zt
    deltaP = -50 * init.deltaP
    Nstep = 10

    # Step 1: Compute initial log-derivative of potential
    r_lo = (1.0 - eps) * r_min
    r_hi = (1.0 + eps) * r_min
    pot_lo = potential(r_lo, Zt)
    pot_hi = potential(r_hi, Zt)
    pot_init = potential(r_min, Zt)
    dpot_dlogr_init = (pot_hi - pot_lo) / (r_hi - r_lo)

    # Step 2: Prepare initial values for ODE integration, assuming linearity
    y = [pot_init, dpot_dlogr_init]  # y = [phi, dphi/dlogr]
    dr = deltaP / y[1]
    r1 = r_min
    r2 = r1 + dr

    rad = [r1]
    pot_vals = [y[0]]

    r_last_print = 0.0

    # Step 3: Integrate until potential crosses zero
    while y[0] > 0.0:
        # Only print if r has changed by at least 0.5 since last print
        if chatter:
            if len(rad) == 1 or abs(r1 - r_last_print) >= 1.0:
                print(f"\rIntegrating Poisson equation outward: r = {r1:.6f}, phi = {y[0]:.6f}", end='', flush=True)
                r_last_print = r1
        step_size = (r2 - r1) / Nstep

        def dphi_dr(r, y):
            phi, dphi_dr = y
            Q = rho_interp(phi)
            return [dphi_dr, -Q - 2 * dphi_dr / r]
        
        sol = solve_ivp(
            dphi_dr,
            (r1, r2),
            y,
            method='RK45',
            t_eval=[r2],
            max_step=step_size,
            rtol=1e-5,
            atol=1e-8
        )

        y = sol.y[:, -1]
        r1 = r2
        rad.append(r2)
        pot_vals.append(y[0])

        dr = deltaP / y[1]
        if dr < 0:
            raise RuntimeError("dr became negative during integration.")

        dr = min(dr, 0.01)
        r2 = r1 + dr

    if chatter:
        print(f"\rIntegrating Poisson equation outward: r = {r1:.6f}, phi = {y[0]:.6f}")
        # print("") # Finalize output line

    # Step 4: truncate and return values
    rcut = r1
    rmax_new = min(config.grid.rmax, 0.99 * rcut)

    pot_interp = interp1d(rad, pot_vals, kind='cubic', fill_value=0.0, bounds_error=False)

    return rcut, rmax_new, pot_interp, np.array(rad), np.array(pot_vals)

def _density_times_r2_trunc(r, state):
    """
    Integrand for computing enclosed mass in truncated NFW profile:
    rho(r) * r^2, where rho is determined either analytically (inner region)
    or from interpolation of the potential and evaluation of the DF (outer region).

    Parameters
    ----------
    r : float
        Radius at which to evaluate the integrand (in units of r_s).
    state : State
        The simulation state, which must include pot and rad arrays.

    Returns
    -------
    float
        Value of the integrand rho(r) * r^2.
    """
    if r < state.pot_rad[0]: # the interpolated potential is not defined below the first radial point
        density = 1.0 / (r * (1.0 + r)**2)
    else:
        pot = state.pot_interp(r)
        density = state.rho_interp(pot)

    return density * r**2

def menc_trunc(r, state, chatter=True):
    """
    Enclosed mass for a truncated NFW profile computed via numerical integration.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    state : State
        The simulation state, which must include config, pot_interp and rho_interp.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass in units of Mvir.
    """
    if not state.config.io.chatter:
        chatter = False
    
    r = np.atleast_1d(r)
    epsabs = state.config.prec.epsabs
    epsrel = state.config.prec.epsrel

    result = np.empty_like(r)

    for i, ri in enumerate(r):
        result[i], _ = quad(
            _density_times_r2_trunc,
            0.0,
            ri,
            args=(state,),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=200
        )
        if chatter:
            print(f"\rComputing Menc: r = {ri:.3f}, m = {result[i]:.3f}", end='', flush=True)
    if chatter:
        print("")  # Finalize output line

    return result if result.ndim > 0 and result.shape[0] > 1 else result[0]

def generate_sigr_integrand_lookup(state, n_points=1000):
    """
    Generate an interpolated function for the velocity dispersion squared
    as a function of radius for the truncated NFW profile.

    Parameters
    ----------
    state : State
        The simulation state object.
    n_points : int
        Number of points to use in the lookup table.

    Returns
    -------
    sigr_interp : interp1d
        Interpolated function for velocity dispersion squared.
    """
    if state.config.io.chatter:
        print("Generating lookup for v2 integrand...")
    
    rgrid = np.geomspace(state.config.grid.rmin / 2 - 1e-4, state.rcut, n_points)
    
    pot = state.pot_interp(rgrid)
    mask = rgrid < state.pot_rad[0]
    density = np.zeros_like(rgrid)
    density[mask] = 1 / (rgrid[mask] * (1.0 + rgrid[mask])**2)
    density[~mask] = state.rho_interp(pot[~mask])
    menc = menc_trunc(rgrid, state, chatter=False)
    integrand_vals = menc * density / rgrid**2
    
    f_interp = interp1d(rgrid, integrand_vals, bounds_error=False, fill_value=0.0)
    
    return f_interp

def sigr_trunc(r, state):
    """ 
    v^2 profile for truncated NFW halo.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    state : State
        The simulation state, which must include config, pot_interp and rho_interp.

    Returns
    -------
    float or ndarray
        Velocity dispersion squared at radius r.
    """
    r = np.atleast_1d(r)
    epsabs = state.config.prec.epsabs
    epsrel = state.config.prec.epsrel
    result = np.empty_like(r)

    integrand = generate_sigr_integrand_lookup(state)

    for i, ri in enumerate(r):
        if ri > state.rcut:
            result[i] = 0.0
            continue
        else:
            if ri < state.pot_rad[0]: # the interpolated potential is not defined below the first radial point
                density = 1.0 / (ri * (1.0 + ri)**2)
            else:
                pot = state.pot_interp(ri)
                density = state.rho_interp(pot)
            if ri < 1.0:
                local_epsabs = 1e-5
                local_epsrel = 1e-3
            else:
                local_epsabs = epsabs
                local_epsrel = epsrel
            integral, _ = quad(
                integrand,
                ri,
                state.rcut,
                epsabs=local_epsabs,
                epsrel=local_epsrel,
                limit=200
            )

            result[i] = integral / density
        if state.config.io.chatter:
            print(f"\rComputing v2: r = {ri:.3f}, v2 = {result[i]:.3f}", end='', flush=True)
    if state.config.io.chatter:
        print("")  # Finalize output line

    return result if result.ndim > 0 and result.shape[0] > 1 else result[0]