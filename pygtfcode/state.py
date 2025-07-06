import numpy as np
from pygtfcode.parameters.char_params import CharParams
from pygtfcode.parameters.constants import Constants as const
from pygtfcode.profiles.nfw import fNFW
from pygtfcode.profiles.abg import chi
import pprint

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

class State:
    """
    Holds characteristic scales, grid, physical variables, time tracking,
    and simulation diagnostics. Constructed from a Config object.
    """

    def __init__(self, config):
        self.config = config
        self.char = self._set_param()
        self.r = self._setup_grid()
        self._initialize_grid()

        self.t = 0.0  # Current time in simulation units

    def _set_param(self):
        """
        Compute and set characteristic physical quantities based on InitParams.
        """
        
        init = self.config.init # Access the InitParams object from config
        sim = self.config.sim # Access the SimParams object from config

        char = CharParams() # Instantiate CharParams object

        rvir = 0.169 * (init.Mvir / 1.0e12)**(1/3)
        rvir *= (const.Delta_vir / 178.0)**(1/3)
        rvir *= (_xH(init.z, const) / (100 * const.xhubble))**(-2/3)
        rvir /= const.xhubble

        Mvir = init.Mvir / const.xhubble

        # vvir = np.sqrt(const.gee * Mvir / rvir)

        if init.profile != 'abg':
            char.fc = fNFW(init.cvir)
            char.m_s = Mvir / char.fc
            char.r_s = rvir / init.cvir
        else:
            char.chi = chi(self.config)
            char.m_s = Mvir / char.chi
            char.r_s = rvir / init.cvir * ( fNFW(init.cvir) / char.chi )**(1/3)

        char.rho_s = char.m_s / ( 4.0 * np.pi * char.r_s**3 )
        char.v0 = np.sqrt(const.gee * char.m_s / char.r_s)
        char.sigma0 = 4.0 * np.pi * char.r_s**2 / char.m_s

        v0_cgs = char.v0 * 1e5
        rho_s_cgs = char.rho_s * const.Msun_to_gram / const.Mpc_to_cm**3
        char.t0 = 1.0 / (sim.a * sim.sigma_m * v0_cgs * rho_s_cgs)

        return char  # Store the CharParams object in config
    
    def _setup_grid(self):
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
        rmin = self.config.grid.rmin
        rmax = self.config.grid.rmax
        Ngrid = self.config.grid.Ngrid

        xlgrmin = np.log10(rmin)
        xlgrmax = np.log10(rmax)

        r = np.empty(Ngrid + 1)
        r[0] = 0.0
        r[1:] = 10**np.linspace(xlgrmin, xlgrmax, Ngrid)

        return r
    
    def _initialize_grid(self):
        """
        Computes initial physical quantities on the radial grid.

        Returns
        -------
        dict of ndarray
            Keys: 'M', 'rho', 'P', 'u', 'v2'
        """
        pass


    def __repr__(self):
        # Copy the __dict__ and omit the 'config' key
        filtered = {k: v for k, v in self.__dict__.items() if k != "config"}
        return f"{self.__class__.__name__}(\n{pprint.pformat(filtered, indent=2)}\n)"

