import numpy as np
from pygtfcode.parameters.char_params import CharParams
from pygtfcode.parameters.constants import Constants as const
from pygtfcode.profiles.nfw import fNFW
from pygtfcode.profiles.abg import chi

def set_param(config):
    """
    Compute and set characteristic physical quantities based on InitParams
    and store them in config.char.

    Parameters
    ----------
    config : Config
        The simulation configuration object.

    Side Effects
    ------------
    - Adds a CharParams object to config.char
    """
    
    init = config.init # Access the InitParams object from config
    sim = config.sim # Access the SimParams object from config

    char = CharParams() # Instantiate CharParams object

    rvir = 0.169 * (init.Mvir / 1.0e12)**(1/3)
    rvir = rvir * (const.Delta_vir / 178.0)**(1/3)
    rvir = rvir * (_xH(init.z, const) / (100 * const.xhubble))**(-2/3)
    rvir = rvir / const.xhubble

    Mvir = init.Mvir / const.xhubble

    vvir = np.sqrt(const.gee * Mvir / rvir)

    if init.profile != 'abg':
        char.fc = fNFW(init.cvir)
        char.m_s = Mvir / char.fc
        char.r_s = rvir / init.cvir
    else:
        char.chi = chi(config)
        char.m_s = Mvir / char.chi
        char.r_s = rvir / init.cvir * ( fNFW(init.cvir) / char.chi )**(1/3)

    char.rho_s = char.m_s / ( 4.0 * np.pi * char.r_s**3 )
    char.v0 = np.sqrt(const.gee * char.m_s / char.r_s)
    char.sigma0 = 4.0 * np.pi * char.r_s**2 / char.m_s
    v0_cgs = char.v0 * 1e5
    rho_s_cgs = char.rho_s * const.Msun_to_gram / const.Mpc_to_cm**3
    char.t0 = 1.0 / (sim.a * sim.sigma_m * v0_cgs * rho_s_cgs)

    config.char = char  # Store the CharParams object in config
    

def setup_grid(config):
    """
    Constructs the radial grid in log-space between rmin and rmax.

    Parameters
    ----------
    config : Config
        The simulation configuration object.

    Returns
    -------
    r : ndarray of shape (Ngrid + 1,)
        Radial Lagrangian grid points, with r[0] = 0 and the rest spaced
        logarithmically between rmin and rmax.
    """
    rmin = config.grid.rmin
    rmax = config.grid.rmax
    Ngrid = config.grid.Ngrid

    xlgrmin = np.log10(rmin)
    xlgrmax = np.log10(rmax)

    r = np.empty(Ngrid + 1)
    r[0] = 0.0

    xlgr = np.linspace(xlgrmin, xlgrmax, Ngrid)
    r[1:] = 10**xlgr

    return r

def initialize_grid(r, config):
    """
    Computes initial physical quantities on the radial grid.

    Returns
    -------
    dict of ndarray
        Keys: 'M', 'rho', 'P', 'u', 'v2'
    """
    ...

def _xH(z, const):
    """
    Returns H(z) in units of km/s/Mpc using cosmological parameters
    defined in config.constants or config.init.

    Parameters
    ----------
    z : float
        Redshift.

    const : Constants
        Configuration object containing cosmological parameters.

    Returns
    -------
    H_z : float
        Hubble parameter at redshift z [km/s/Mpc].
    """
    
    omega_lambda = 1 - const.Omega_m
    xH_0 = 100 * const.xhubble  # H_0 in km/s/Mpc

    fac = omega_lambda + (1.0 - omega_lambda - const.Omega_m) * (1.0 + z)**2 + const.Omega_m * (1.0 + z)**3

    return xH_0 * np.sqrt(fac)


# Early on, we will call this:

# from pygtfcode.runtime.initialize import setup_grid, initialize_grid
# from pygtfcode.runtime.runtime_state import RuntimeState

# r = setup_grid(config)
# if the initial profile is truncated_nfw, we will also need to compute the potential grid
# if config.init.profile == "truncated_nfw":
#     rad, pot = integrate_potential(config)
# fields = initialize_grid(r, config)

# state = RuntimeState(r=r, **fields)

# class Simulator:
#     """
#     Placeholder for the main simulation class.

#     Eventually responsible for:
#     - Time integration loop
#     - Updating RuntimeState
#     - Handling outputs and stopping criteria
#     """

#     def __init__(self, config):
#         self.config = config

#     def run(self):
#         print("Simulation not yet implemented.")