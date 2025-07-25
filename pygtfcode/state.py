import numpy as np
from pygtfcode.parameters.constants import Constants as const
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
    Omega_m = const.Omega_m
    omega_lambda = 1 - Omega_m
    xH_0 = 100 * const.xhubble  # H_0 in km/s/Mpc

    fac = omega_lambda + (1.0 - omega_lambda - Omega_m) * (1.0 + z)**2 + Omega_m * (1.0 + z)**3

    return xH_0 * np.sqrt(fac)

class State:
    """
    Holds characteristic scales, grid, physical variables, time tracking,
    and simulation diagnostics. Constructed from a Config object.
    """

    def __init__(self, config):
        from pygtfcode.io.write import make_dir, write_metadata

        self.config = config
        self.char = self._set_param()
        if self.config.init.profile == 'truncated_nfw':
            from pygtfcode.profiles.truncated_nfw import integrate_potential, generate_rho_lookup
            self.rho_interp = generate_rho_lookup(config)
            self.rcut, self.config.grid.rmax, self.pot_interp, self.pot_rad, self.pot = integrate_potential(config, self.rho_interp)
        self.r = self._setup_grid()
        self._initialize_grid()

        self.t = 0.0                        # Current time in simulation units
        self.step_count = 0                 # Global integration step counter (never reset)
        self.snapshot_index = 0             # Counts profile output snapshots
        self.dt = 1e-6                      # Initial time step (will be updated adaptively)
        self.du_max = config.prec.eps_du    # Initialize the max du to upper limit
        self.dr_max = config.prec.eps_dr    # Initialize the max dr to upper limit

        self.maxvel = np.sqrt(np.max(self.v2))
        self.minkn = np.min(self.kn)
        self.mintrelax = np.min(self.trelax)

        if config.io.chatter:
            print("State initialized.")

        make_dir(self)                      # Create the model directory if it doesn't exist
        write_metadata(self)                # Write model metadata to disk

    def _set_param(self):
        """
        Compute and set characteristic physical quantities based on InitParams.
        """
        from pygtfcode.parameters.char_params import CharParams
        from pygtfcode.profiles.nfw import fNFW
        from pygtfcode.profiles.abg import chi

        if self.config.io.chatter:
            print("Computing characteristic parameters for simulation...")
        init = self.config.init # Access the InitParams object from config
        sim = self.config.sim # Access the SimParams object from config

        char = CharParams() # Instantiate CharParams object

        rvir = 0.169 * (init.Mvir / 1.0e12)**(1/3)
        rvir *= (const.Delta_vir / 178.0)**(-1.0/3.0)
        rvir *= (_xH(init.z, const) / (100 * const.xhubble))**(-2/3)
        rvir /= const.xhubble

        Mvir = init.Mvir / const.xhubble

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
        sigma0 = 4.0 * np.pi * char.r_s**2 / char.m_s # In Mpc^2 / Msun^2
        char.sigma0 = sigma0 * const.Mpc_to_cm**2 / const.Msun_to_gram # In cm^2 / g

        v0_cgs = char.v0 * 1e5
        rho_s_cgs = char.rho_s * const.Msun_to_gram / const.Mpc_to_cm**3
        char.t0 = 1.0 / (sim.a * sim.sigma_m * v0_cgs * rho_s_cgs)
        char.sigma_m_char = sim.sigma_m / char.sigma0 # sigma_m in dimensionless form

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
        r : ndarray of shape (ngrid + 1,)
            Radial Lagrangian grid points, with r[0] = 0 and the rest spaced
            logarithmically between rmin and rmax.
        """
        if self.config.io.chatter:
            print("Setting up radial grid...")
        rmin = self.config.grid.rmin
        rmax = self.config.grid.rmax
        ngrid = self.config.grid.ngrid

        xlgrmin = np.log10(rmin)
        xlgrmax = np.log10(rmax)

        r = np.empty(ngrid + 1)
        r[0] = 0.0
        r[1:] = 10**np.linspace(xlgrmin, xlgrmax, ngrid)

        return r
    
    def _initialize_grid(self):
        """
        Computes initial physical quantities on the radial grid using the
        initial profile defined in config.

        Sets the following attributes:
            - m: Enclosed mass at r[i+1]
            - rho: Density in each shell (size ngrid)
            - p: Pressure in each shell
            - u: Internal energy in each shell
            - v2: Velocity dispersion squared in each shell
            - kn: Knudsen number in each shell
            - maxvel: maximum velocity dispersion
            - minkn: minimum Knudsen number
        """
        from pygtfcode.profiles.profile_routines import menc, sigr

        if self.config.io.chatter:
            print("Initializing profiles...")

        r = self.r
        r_mid = 0.5 * (r[1:] + r[:-1])          # Midpoint of each shell
        dr3 = r[1:]**3 - r[:-1]**3              # Volume difference per shell
    
        m_outer = menc(r[1:], self)             # m[i] at shell edges
        m = np.concatenate(([0.0], m_outer))
        dm = m_outer - m[:-1]

        v2 = sigr(r_mid, self)
        rho = 3.0 * dm / dr3
        p = rho * v2
        u = 1.5 * v2
        kn = 1.0 / (self.char.sigma_m_char * np.sqrt(p))
        trelax = 1.0 / (np.sqrt(v2) * rho)

        # Apply central smoothing if using regular NFW profile (imode = 1)
        # This helps reduce artificial gradients in innermost cell
        if self.config.init.profile == "nfw":
            r1 = r[1]
            rho_c_ideal = 1.0 / (r1 * (1.0 + r1)**2)
            rho[0] = 2.0 * rho_c_ideal - rho[1]

            dr_ratio = (r[2] - r[0]) / (r[3] - r[1])
            p[0] = p[1] - dr_ratio * (p[2] - p[1])

            v2[0] = p[0] / rho[0]
            u[0] = 1.5 * v2[0]

        self.m = m
        self.rmid = r_mid
        self.rho = rho
        self.p = p
        self.u = u
        self.v2 = v2
        self.kn = kn
        self.trelax = trelax

    def run(self, steps=None, time=None, rho_c=None):
        """
        Run the simulation until a halting criterion is met.
        User can set halting criteria to run for a specified duration.
        These are overridden by the halting criteria in self.config.

        Arguments
        ---------
        steps : int, optional
            Number of steps to advance the simulation
        time : float, optional
            Amount of time by which to advance the simulation
        rho_c: float, optional
            Max central denisty value to advance until
        """
        from pygtfcode.evolve.integrator import run_until_stop
        from pygtfcode.io.write import write_log_entry, write_profile_snapshot, write_time_evolution

        # Prepare kwargs for run_until_stop if any halting criteria are provided
        kwargs = {}
        if steps is not None:
            kwargs['steps'] = steps
        if time is not None:
            kwargs['time'] = time
        if rho_c is not None:
            kwargs['rho_c'] = rho_c

        # Write initial state to disk
        write_profile_snapshot(self)
        write_time_evolution(self)
        write_log_entry(self)

        # Integrate forward in time until a halting criterion is met
        run_until_stop(self, **kwargs)

        # Write final state to disk
        write_profile_snapshot(self)
        write_time_evolution(self)
        write_log_entry(self)

    def get_phys(self):
        """
        Method to print characteristic quantities in physical units
        """
        from pygtfcode.profiles.profile_routines import menc
        char = self.char
        init = self.config.init

        Mtot = menc(self.config.grid.rmax, self) * char.m_s
        rvir = 0.169 * (init.Mvir / 1.0e12)**(1/3)
        rvir *= (const.Delta_vir / 178.0)**(-1.0/3.0)
        rvir *= (_xH(init.z, const) / (100 * const.xhubble))**(-2/3)
        rvir /= const.xhubble
        vvir = np.sqrt(const.gee * init.Mvir / const.xhubble / rvir)

        params_dict = {
            'log[Mvir/Msun]'            : np.log10(init.Mvir / const.xhubble),
            'log[Mtot/Msun]'            : np.log10(Mtot),
            'Vvir [km/s]'               : vvir,
            'v_0 [km/s]'                : char.v0,
            'log[rho_s/(Msun/kpc^3)]'   : np.log10(char.rho_s * 1.0e-9),
            'r_s [kpc]'                 : char.r_s * 1.0e3,
            't_0 [Gyr]'                 : char.t0 * const.sec_to_Gyr
        }

        return params_dict

    def __repr__(self):
        # Copy the __dict__ and omit the 'config' key
        filtered = {k: v for k, v in self.__dict__.items() if k != "config"}
        return f"{self.__class__.__name__}(\n{pprint.pformat(filtered, indent=2)}\n)"

