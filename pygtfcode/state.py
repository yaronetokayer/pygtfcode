import numpy as np
import pprint
import warnings
from pathlib import Path

def _print_time(start, end, funcname):
    """
    Routine to print elapsed time in a readable way
    """
    elapsed = end - start

    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{int(days)}d")
    if hours:
        parts.append(f"{int(hours)}h")
    if minutes:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.2f}s")  # Always include seconds

    print(f"Total time for {funcname}:", "".join(parts))

class State:
    """
    Holds characteristic scales, grid, physical variables, time tracking,
    and simulation diagnostics. Constructed from a Config object.

    kn and mfp retain amplitude-reference meanings. kn_cond and mfp_cond
    describe the conductivity transition, not literal collision statistics.
    kn_c and kn_cond_c are their respective core logarithmic means.
    The timestep tolerance boost uses kn_cond_c.
    """

    def __init__(self, config, ic_filepath=None):

        self.config = config
        self.char = self._set_param()
        if ic_filepath is None:
            if self.config.init.profile == 'truncated_nfw': # Numerical integrations for non-analytic truncated NFW profile
                from pygtfcode.profiles.truncated_nfw import integrate_potential, generate_rho_lookup
                self.rho_interp = generate_rho_lookup(config)
                self.rcut, self.config.grid.rmax, self.pot_interp, self.pot_rad, self.pot = integrate_potential(config, self.rho_interp)

    @classmethod
    def from_config(cls, config, ic_filepath=None):
        """
        Create a State object from a Config object.

        Parameters
        ----------
        config : Config
            Configuration object containing simulation parameters.
        ic_filepath : str, optional
            If provided, loads initial conditions from this file

        Returns
        -------
        State
            A new State object initialized with the given configuration.
        """
        from pygtfcode.io.write import make_dir, write_metadata, write_profile_snapshot, write_char_params

        state = cls(config, ic_filepath=ic_filepath)
        state.reset(ic_filepath=ic_filepath)                                    # Initialize all state variables

        make_dir(state)                                  # Create the model directory if it doesn't exist
        write_char_params(state)                         # Write model characteristic parameters to disk
        write_metadata(state)                            # Write model metadata to disk
        write_profile_snapshot(state, initialize=True)   # Write initial snapshot to disk

        return state

    @classmethod
    def from_dir(cls, model_dir: str, snapshot: None | int = None):
        """
        Restart from a model directory (not implemented).

        This method currently raises RuntimeError. Use from_config with
        ic_filepath to start a new run from a saved profile; that resets time.

        Parameters
        ----------
        model_dir : str
            Path to the model directory containing simulation data.
        snapshot : int, optional
            Snapshot index to load. If None, loads the latest snapshot.

        Raises
        ------
        RuntimeError
            Restart support is not implemented.
        """
        raise RuntimeError("State.from_dir restart support is not implemented")

    @classmethod
    def make_ic_file(cls, config, ic_filepath):
        """
        Write initial conditions to a file.

        Parameters
        ----------
        config : Config
            Configuration object containing simulation parameters.
        ic_filepath : str
            Path to write IC file

        Returns
        -------
        None
            Writes an unrelaxed profile; from_config relaxes it when loaded.
        """
        from pygtfcode.io.write import write_profile_snapshot
        
        state = cls(config)
        config = state.config

        state.r = state._setup_grid()
        state._initialize_grid()
        state.dt = 1.0e-6

        write_profile_snapshot(state, initialize=True, ic_filename=ic_filepath)   # Write initial snapshot to disk

    def _set_param(self):
        """
        Compute and set characteristic physical quantities based on InitParams.
        """
        from pygtfcode.parameters.char_params import CharParams
        from pygtfcode.profiles.nfw import fNFW
        from pygtfcode.parameters.constants import Constants as const

        if self.config.io.chatter:
            print("Computing characteristic parameters for simulation...")
        init    = self.config.init # Access the InitParams object from config
        sim     = self.config.sim # Access the SimParams object from config
        cosmo   = self.config.cosmo

        char = CharParams() # Instantiate CharParams object

        # Ensure double precision
        Mvir  = float(init.Mvir)
        cvir  = float(init.cvir)

        rvir = 0.169 * (Mvir / 1.0e12)**(1.0/3.0)
        rvir *= (float(cosmo.Delta_vir) / 178.0)**(-1.0/3.0)
        rvir *= (cosmo.xH() / (100.0 * float(cosmo.xhubble)))**(-2.0/3.0)
        rvir /= float(cosmo.xhubble)

        Mvir_h = Mvir / float(cosmo.xhubble)
        char.fc = float(fNFW(cvir))
        char.r_s = rvir / cvir

        if init.profile != 'abg':
            char.m_s = Mvir_h / char.fc # M(<cvir) = m_s * fNFW(cvir).
            
        else:
            from pygtfcode.profiles.abg import chi
            char.chi = float(chi(self.config))
            char.m_s = Mvir_h / char.chi
            char.r_s *= ( char.fc / char.chi )**(1.0/3.0)

        char.rho_s = char.m_s / ( 4.0 * np.pi * char.r_s**3 )
        char.v_s = float(np.sqrt(const.gee * char.m_s / char.r_s))
        sigma_m_s = 4.0 * np.pi * char.r_s**2 / char.m_s # In Mpc^2 / Msun
        char.sigma_m_s = sigma_m_s * float(const.Mpc_to_cm)**2 / float(const.Msun_to_gram) # In cm^2 / g

        v_s_cgs = char.v_s * 1.0e5
        rho_s_cgs = char.rho_s * float(const.Msun_to_gram) / float(const.Mpc_to_cm)**3
        char.t_s = 1.0 / (float(sim.a) * float(sim.sigma_m_0) * v_s_cgs * rho_s_cgs)
        char.sigma_m_0_char = float(sim.sigma_m_0) / char.sigma_m_s # Cross-section amplitude in characteristic units
        char.w_char = float(sim.w) / char.v_s

        return char
    
    def _setup_grid(self):
        """
        Constructs the radial grid in log-space between rmin and rmax.

        Returns
        -------
        r : ndarray of shape (ngrid + 1,)
            Radial Lagrangian grid points, with r[0] = 0 and the rest spaced
            logarithmically between rmin and rmax.
        """
        if self.config.io.chatter:
            print("Setting up radial grid...")

        # Compute ngrid
        # drfrac = (q - 1) / sqrt(q), q = rR/rL for each cell
        rmin = float(self.config.grid.rmin)
        rmax = float(self.config.grid.rmax)
        drfrac_init = float(self.config.grid.drfrac_init)

        # x=sqrt(q)
        x = 0.5 * (drfrac_init + np.sqrt(drfrac_init**2 + 4.0))
        q_init = x**2

        log_span = np.log(rmax / rmin)
        log_q_init = np.log(q_init)

        # Number of log-spaced cells between rmin and rmax.
        nlog = int(np.ceil(log_span / log_q_init))
        nlog = max(nlog, 2)
        ngrid = nlog + 1

        log_q_actual = log_span / nlog
        q_actual = np.exp(log_q_actual)
        drfrac_actual = (q_actual - 1.0) / np.sqrt(q_actual)

        r = np.empty(ngrid + 1, dtype=np.float64)
        r[0] = 0.0
        r[1:] = np.exp(
            np.linspace(np.log(rmin), np.log(rmax), ngrid, dtype=np.float64)
        )

        self.n = int(ngrid)

        if self.config.io.chatter:
            print(f"\tRadial grid set up with {ngrid} cells and uniform dr/r = {drfrac_actual:.4e} outside the innermost bin.")

        return r
    
    def _initialize_grid(self):
        """
        Computes initial physical quantities on the radial grid using the
        initial profile defined in config.

        Sets the following attributes:
            - m: Enclosed mass at edges r[i], including m[0] = 0
            - rho: Density in each shell (size ngrid)
            - v2: Velocity dispersion squared in each shell
            - kn, mfp: Amplitude-reference transport scales in each shell
            - kn_c: Knudsen number of the core
        """
        from pygtfcode.profiles.profile_routines import menc, sigr
        from pygtfcode.util.calc_runtime import calc_ltemp

        if self.config.io.chatter:
            print("Initializing profiles...")

        r = self.r.astype(np.float64, copy=False)
        r_mid = 0.5 * (r[1:] + r[:-1])          # Midpoint of each shell
        dr3 = r[1:]**3 - r[:-1]**3              # Volume difference per shell

        m = np.zeros_like(r, dtype=np.float64)
        m[1:] = menc(r[1:], self)               # m[i] at shell edges

        # Truncated-halo mass relative to the parent NFW virial mass.
        if self.config.init.profile == 'truncated_nfw':
            from pygtfcode.profiles.nfw import fNFW
            # Calculate Mtot / M200
            mtot = menc(self.rcut, self, chatter=False)
            fc = fNFW(self.config.init.cvir)
            self.char.mtot_m200 = mtot/fc

        v2 = np.asarray(sigr(r_mid, self), dtype=np.float64)
        rho = 3.0 * ( m[1:] - m[:-1] ) / dr3
        kn = 1.0 / (self.char.sigma_m_0_char * np.sqrt(rho * v2))
        mfp = 1.0 / (self.char.sigma_m_0_char * rho)

        # Apply central smoothing for the regular NFW profile
        # This helps reduce artificial gradients in innermost cell
        if self.config.init.profile == "nfw":
            r1 = r[1]
            rho_c_ideal = 1.0 / (r1 * (1.0 + r1)**2)
            rho[0] = 2.0 * rho_c_ideal - rho[1]

            dr_ratio = (r[2] - r[0]) / (r[3] - r[1])
            p = rho * v2
            p[0] = p[1] - dr_ratio * (p[2] - p[1])

            v2[0] = p[0] / rho[0]

        # State arrays
        self.m      = m
        self.rmid   = r_mid
        self.rho    = rho
        self.v2     = v2

        # Derived quantities
        self.kn         = kn
        self.drfrac     = np.zeros_like(rho, dtype=np.float64)
        self.drfrac[0]  = np.nan
        self.drfrac[1:] = (r[2:]/r[1:-1] - 1.0) / np.sqrt(r[2:]/r[1:-1])
        self.ltemp      = np.zeros_like(rho, dtype=np.float64)
        calc_ltemp(self.ltemp, v2, r_mid)
        self.mfp        = np.asarray(mfp, dtype=np.float64)

        # No accepted conduction rate is available before the first step.
        self.t_cool = np.full_like(rho, np.inf, dtype=np.float64)
        self.t_dyn  = (self.config.sim.a * self.char.sigma_m_0_char / np.sqrt(rho)).astype(np.float64)

        self._update_transport_diagnostics()

    def _load_ic(self, ic_filepath):
        """
        Loads initial conditions from a snapshot file.

        Parameters
        ----------
        ic_filepath : str
            Path to the snapshot file containing initial conditions.
        """
        from pygtfcode.io.read import extract_snapshot_data
        from pygtfcode.util.calc_runtime import calc_ltemp

        if self.config.io.chatter:
            print(f"Loading initial conditions from {ic_filepath} ...")

        data = extract_snapshot_data(ic_filepath, add_time=False)

        # Check that the grid matches
        r_loaded = np.insert(10**data['log_r'].astype(np.float64), 0, 0.0)
        if r_loaded.shape != self.r.shape or not np.allclose(r_loaded, self.r, rtol=1e-5, atol=1e-8):
            warnings.warn("Radial grid in IC file does not match the grid defined by the current configuration.  Using IC file grid.", RuntimeWarning)
            self.r = r_loaded
            self.n = r_loaded.size - 1

        self.rmid   = 0.5 * (self.r[1:] + self.r[:-1]).astype(np.float64)
        self.m      = np.insert(data['m'].astype(np.float64), 0, 0.0)
        self.rho    = data['rho'].astype(np.float64)
        self.v2     = data['v2'].astype(np.float64)

        # Derived quantities
        self.kn         = np.asarray(1.0 / (self.char.sigma_m_0_char * np.sqrt(self.rho * self.v2)), dtype=np.float64)
        self.drfrac     = np.zeros_like(self.rho, dtype=np.float64)
        self.drfrac[0]  = np.nan
        self.drfrac[1:] = (self.r[2:]/self.r[1:-1] - 1.0) / np.sqrt(self.r[2:]/self.r[1:-1])
        self.ltemp      = np.zeros_like(self.rho, dtype=np.float64)
        calc_ltemp(self.ltemp, self.v2, self.rmid)
        self.mfp        = np.asarray( 1.0 / (self.char.sigma_m_0_char * self.rho), dtype=np.float64)


        self.t_cool = np.full_like(self.rho, np.inf, dtype=np.float64)
        self.t_dyn  = (self.config.sim.a * self.char.sigma_m_0_char / np.sqrt(self.rho)).astype(np.float64)

        self._update_transport_diagnostics()

    def _ensure_virial_equilibrium(self):
        """
        Fine-tunes initial profile to ensure hydrostatic equilibrium.
        Iteratively runs revirialize() until max |dr/r| < eps_dr.
        """
        from pygtfcode.evolve.hydrostatic import revirialize_w_he_resid, compute_he_pressures_with_resid, STATUS_SHELL_CROSSING
        from pygtfcode.util.calc_runtime import calc_ltemp
        chatter = self.config.io.chatter

        if chatter:
            print("Ensuring initial hydrostatic equilibrium...")

        r_new   = self.r.astype(np.float64,             copy=True)
        rho_new = self.rho.astype(np.float64,           copy=True)
        m       = self.m.astype(np.float64,             copy=False)
        p_new   = rho_new * self.v2.astype(np.float64,  copy=True)

        # Update pressure with backward sweep
        res_old, res_new = compute_he_pressures_with_resid(self.r, self.rho, p_new, m)
        if chatter:
            print(f"\tInitial pressure correction applied. HE residual improved {float(res_old):.3e} -> {float(res_new):.3e}.")

        # Iteratively revirialize to achieve necessary precision
        # Preallocate arrays
        Np1 = r_new.shape[0]
        n_int = Np1 - 2
        a  = np.empty(n_int, dtype=np.float64)
        b  = np.empty(n_int, dtype=np.float64)
        c  = np.empty(n_int, dtype=np.float64)
        y  = np.empty(n_int, dtype=np.float64)
        x  = np.empty(n_int, dtype=np.float64)
        vol_old = np.empty(Np1 - 1, dtype=np.float64)

        eps_dr = float(self.config.prec.eps_dr)

        i = 0
        while True:
            i += 1
            status, dr_max_new, he_res = revirialize_w_he_resid(r_new, rho_new, p_new, m,
                                                                 a, b, c, y, x, vol_old)
            
            if status == STATUS_SHELL_CROSSING:
                raise RuntimeError(f"Initial revir iter {i}: Shell crossing!")

            if dr_max_new < eps_dr:
                break

            if i >= 100:
                raise RuntimeError("Failed to achieve hydrostatic equilibrium in 100 iterations.")
            
        v2_new = p_new / rho_new

        self.r = r_new
        self.rho = rho_new
        self.v2 = v2_new

        self.rmid[:]    = 0.5 * (r_new[1:] + r_new[:-1])
        self._update_transport_diagnostics()
        calc_ltemp(self.ltemp, self.v2, self.rmid)
        self.t_dyn[:]   = self.config.sim.a * self.char.sigma_m_0_char / np.sqrt(self.rho)
        self.drfrac[0] = np.nan
        self.drfrac[1:] = np.diff(self.r[1:]) / np.sqrt(self.r[1:-1] * self.r[2:])

        if chatter:
            print(f"Hydrostatic equilibrium achieved in {i} iterations. Max |dr/r| = {dr_max_new:.2e}.  HE res {he_res}.")

    def _update_transport_diagnostics(self):
        """Refresh both diagnostic conventions on the current cell grid.

        kn_c remains the amplitude-reference core mean for compatibility.
        kn_cond_c is the conductivity-effective core mean used by the
        timestep tolerance boost.
        """
        from pygtfcode.util.calc_runtime import calc_transport_scales
        from pygtfcode.util.calc_core import calc_core_r, calc_logmean_within_r

        for name in ('kn', 'mfp', 'kn_cond', 'mfp_cond'):
            if not hasattr(self, name) or getattr(self, name).size != self.rho.size:
                setattr(self, name, np.empty_like(self.rho))
        calc_transport_scales(
            self.v2, self.rho, float(self.char.sigma_m_0_char),
            float(self.char.w_char), self.config.sim.smfp_order,
            self.kn, self.mfp, self.kn_cond, self.mfp_cond,
        )
        r_c = calc_core_r(self.r, self.rmid, self.rho)
        self.kn_c = calc_logmean_within_r(self.r, self.m, self.kn, r_c)
        self.kn_cond_c = calc_logmean_within_r(self.r, self.m, self.kn_cond, r_c)

    def reset(self, ic_filepath=None):
        """
        Resets initial state
        """
        config = self.config

        self.r = self._setup_grid()
        if ic_filepath is not None:
            # Check if filepath exists
            if not Path(ic_filepath).is_file():
                print(f"IC file {ic_filepath} not found. Creating IC file at that location...")
                self.make_ic_file(config, ic_filepath=ic_filepath)
            self._load_ic(ic_filepath)
        else:
            self._initialize_grid()
        self._ensure_virial_equilibrium()

        self.t = 0.0                        # Current time in simulation units
        self.step_count = 0                 # Global integration step counter (never reset)
        self.snapshot_index = 0             # Counts profile output snapshots
        self.dt = 1e-6                      # Initial time step (will be updated adaptively)
        self.du_max = 0.0                   # Max du of most recent step (used for adaptive time stepping)

        # Recompute both conventions after initialization and relaxation.
        self._update_transport_diagnostics()

        self.n_iter_du          = 0
        self.n_iter_dr          = 0

        self.dt_cum             = 0.0
        self.dr_max_cum         = 0.0
        self.du_max_cum         = 0.0
        self.du_limit_cum = 0.0
        self.log_steps = 0
        self.n_split = 0
        self.n_merge = 0

        if config.io.chatter:
            print("State initialized.")

    def run(self, steps=None, stoptime=None, rho_c=None):
        """
        Run the simulation until a halting criterion is met.
        User can set halting criteria to run for a specified duration.
        The first satisfied user or configuration criterion stops the run.
        The configured density limit is checked only after t > 50 (in t_s).

        Arguments 
        ---------
        steps : int, optional
            Number of steps to advance the simulation
        stoptime : float, optional
            Amount of simulation time by which to advance the simulation
        rho_c: float, optional
            Maximum innermost-cell density (rho[0]/rho_s) to advance until
        """
        from pygtfcode.evolve.integrator import run_until_stop
        from pygtfcode.io.write import write_log_entry, write_profile_snapshot, write_time_evolution
        from time import time as _now

        start = _now()
        start_step = self.step_count

        # Prepare kwargs for run_until_stop if any halting criteria are provided
        kwargs = {}
        if steps is not None:
            kwargs['steps'] = steps
        if stoptime is not None:
            kwargs['stoptime'] = stoptime
        if rho_c is not None:
            kwargs['rho_c'] = rho_c

        # Write initial state to disk 
        write_profile_snapshot(self)
        write_time_evolution(self)
        write_log_entry(self, start_step)

        # Integrate forward in time until a halting criterion is met
        run_until_stop(self, start_step, **kwargs)

        # Write final state to disk
        write_profile_snapshot(self)
        write_time_evolution(self, last=True)
        write_log_entry(self, start_step)

        end = _now()
        _print_time(start, end, funcname="run()")
        
    def get_phys(self):
        """
        Return a dictionary of characteristic quantities in physical units
        """
        from pygtfcode.parameters.constants import Constants as const

        char = self.char
        init = self.config.init
        cosmo = self.config.cosmo

        Mtot = self.m[-1] * char.m_s
        rvir = 0.169 * (init.Mvir / 1.0e12)**(1/3)
        rvir *= (cosmo.Delta_vir / 178.0)**(-1.0/3.0)
        rvir *= (cosmo.xH() / (100 * cosmo.xhubble))**(-2/3)
        rvir /= cosmo.xhubble
        vvir = np.sqrt(const.gee * init.Mvir / cosmo.xhubble / rvir)

        params_dict = {
            'log[Mvir/Msun]'            : np.log10(init.Mvir / cosmo.xhubble),
            'log[Mtot/Msun]'            : np.log10(Mtot),
            'Vvir [km/s]'               : vvir,
            'v_s [km/s]'                : char.v_s,
            'log[rho_s/(Msun/kpc^3)]'   : np.log10(char.rho_s * 1.0e-9),
            'r_s [kpc]'                 : char.r_s * 1.0e3,
            't_s [Gyr]'                 : char.t_s * const.sec_to_Gyr
        }

        return params_dict

    def plot_time_evolution(self, **kwargs):
        """
        Plot any time-evolution quantity vs. time for the simulation represented by
        the State object

        Arguments
        ---------
        quantity : str, optional
            Key from the time_evolution.txt file to plot on the y-axis.
            Default is 'rho0'.
            Any time_evolution.txt column, including rho0, kn_cond_c,
            x_c/x_m2, K_L_c/K_S_c and K_L_m2/K_S_m2.
        ylabel : str, optional
            Custom y-axis label. Defaults to quantity.
        logy : bool, optional
            Use logarithmic scale on y-axis. Default is True.
        filepath : str, optional
            If specified, saves the figure to this path.
        show : bool, optional
            If True, show the plot even if saving.  Default is False.
        grid : bool, optional
            If True, shows grid on axis
        """
        from pygtfcode.plot.time_evolution import plot_time_evolution

        return plot_time_evolution(self, **kwargs)

    def plot_snapshots(self, **kwargs):
        """
        Method to plot up to three profiles at specified points in time for the simulation represented by
        the State object

        Arguments
        ---------
        snapshots : int or list of int, optional
            Snapshot indices to plot, default is the current state
        profiles : str or list of str, optional
            Profiles from plot.snapshot.VALID_PROFILES, including rho, v2,
            kn_cond, mfp_cond, x, K_L, and K_S.
        filepath : str, optional
            If provided, save the plot to this file.
        show : bool, optional
            If True, show the plot even if saving.  Default is False.
        grid : bool, optional
            If True, shows grid on axes
        """
        from pygtfcode.plot.snapshot import plot_snapshots

        snapshots = kwargs.pop('snapshots', -1)
        return plot_snapshots(self, snapshots=snapshots, **kwargs)
        
    def make_movie(self, **kwargs):
        """
        Animate profiles with the deluxe renderer for the simulation represented by
        the State object

        Arguments
        ---------
        parallel : bool, optional
            Render in parallel by default; False uses serial rendering.
        insets : False, str, list, or None, optional
            False disables all insets; None uses rho0 in the first panel.
            Lists specify a history column or None for each panel.
        filepath : str, optional
            Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_{profiles}.mp4'
        profiles : str or list of str, optional
            Profiles from plot.snapshot.VALID_PROFILES, including rho, v2,
            kn_cond, mfp_cond, x, K_L, and K_S.
        grid : bool, optional
            If True, shows grid on axes
        fps : int, optional
            Frames per second for the output movie. Default is 20

        Returns
        -------
        None
            Saves the movie as an MP4 file in the model directory.
        """
        from pygtfcode.plot.snapshot import make_movie

        make_movie(self, **kwargs)

    def resize_state_arrays(self):
        n = self.n

        self.rmid   = np.empty(n,   dtype=np.float64)
        self.kn     = np.empty(n,   dtype=np.float64)
        self.kn_cond = np.empty(n, dtype=np.float64)
        self.mfp_cond = np.empty(n, dtype=np.float64)
        self.ltemp  = np.empty(n,   dtype=np.float64)
        self.mfp    = np.empty(n,   dtype=np.float64)
        self.t_dyn  = np.empty(n,   dtype=np.float64)
        self.drfrac = np.empty(n,   dtype=np.float64)
        self.t_cool = np.full(n, np.inf, dtype=np.float64)

        self.rmid[:] = 0.5 * (self.r[1:] + self.r[:-1])
        self._update_transport_diagnostics()
        from pygtfcode.util.calc_runtime import calc_ltemp
        calc_ltemp(self.ltemp, self.v2, self.rmid)
        self.drfrac[0] = np.nan
        self.drfrac[1:] = np.diff(self.r[1:]) / np.sqrt(self.r[1:-1] * self.r[2:])
        self.t_dyn[:] = self.config.sim.a * self.char.sigma_m_0_char / np.sqrt(self.rho)

    def __repr__(self):
        # Copy the __dict__ and omit the 'config' key
        filtered = {k: v for k, v in self.__dict__.items() if k != "config"}
        return f"{self.__class__.__name__}(\n{pprint.pformat(filtered, indent=2)}\n)"
