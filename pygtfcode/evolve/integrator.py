import numpy as np
from pygtfcode.io.write import write_profile_snapshot, write_log_entry, write_time_evolution
from pygtfcode.evolve.transport import compute_luminosities, conduct_heat
from pygtfcode.evolve.hydrostatic import revirialize, compute_mass, STATUS_SHELL_CROSSING

def run_until_stop(state, start_step, **kwargs):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    ##################
    ### Set locals ###
    ##################

    # User halting criteria
    steps = kwargs.get('steps', None)
    time_limit = kwargs.get('stoptime', None)
    rho_c_limit = kwargs.get('rho_c', None)
    step_i = state.step_count if steps is not None else None
    time_i = state.t if time_limit is not None else None

    # Locals for speed + type hardening
    config = state.config
    io = config.io
    sim = config.sim
    chatter = bool(io.chatter)
    t_halt = float(sim.t_halt)
    rho0_last_prof = float(state.rho[0])
    rho0_last_tevol = float(state.rho[0])
    rho_c_halt = float(sim.rho_c_halt)
    drho_prof = float(io.drho_prof)
    drho_tevol = float(io.drho_tevol)
    nlog = int(io.nlog)
    nupdate = int(io.nupdate)

    # Preallocate working arrays for main loop
    Np1         = state.r.shape[0]
    n_int       = Np1 - 2
    lum         = np.zeros(Np1,     dtype=np.float64)
    a_alloc     = np.empty(n_int,   dtype=np.float64)
    b_alloc     = np.empty(n_int,   dtype=np.float64)
    c_alloc     = np.empty(n_int,   dtype=np.float64)
    y_alloc     = np.empty(n_int,   dtype=np.float64)
    x_alloc     = np.empty(n_int,   dtype=np.float64)
    work        = np.empty(Np1 - 1, dtype=np.float64)

    #################
    ### Main loop ###
    #################

    while state.t < t_halt:

        ###########################
        ### 1. Integrate system ###
        ###########################

        # Increment counter
        state.step_count += 1
        step_count = state.step_count

        # Compute relaxation-time-limited dt
        # In low-Kn regime, only set by conduction in conduction routine
        if state.minkn > 0.1:
            dt_prop = config.prec.eps_dt * state.mintrelax
        else:
            dt_prop = 1.0

        # Integrate time step
        integrate_time_step(state, config, dt_prop, step_count, lum, a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work, Np1)

        if step_count % nupdate == 0:
            print(f"Completed step {step_count}", end='\r', flush=True)

        rho0 = state.rho[0]

        ###########################
        ### 2. Halting criteria ###
        ###########################

        # Check halting criteria
        if rho0 > rho_c_halt:
            if state.t > 50:
                if chatter:
                    print("Simulation halted: central density exceeds halting value")
                break
        if np.isnan(rho0):
            if chatter:
                print("Simulation halted: central density is nan")
            break

        # User halting criteria
        if (
            (steps is not None and step_count - step_i >= steps)
            or (time_limit is not None and state.t - time_i >= time_limit)
            or (rho_c_limit is not None and rho0 >= rho_c_limit)
        ):
            if chatter:
                print("Simulation halted: user stopping condition reached")
            break

        #########################
        ### 3. Output to disk ###
        #########################

        # Check I/O criteria
        # Write profile to disk
        drho_for_prof = np.abs(rho0 - rho0_last_prof) / rho0_last_prof
        if drho_for_prof > drho_prof:
            rho0_last_prof = rho0
            write_profile_snapshot(state)

        # Track time evolution 
        drho_for_tevol = np.abs(rho0 - rho0_last_tevol) / rho0_last_tevol
        if drho_for_tevol > drho_tevol:
            rho0_last_tevol = rho0
            write_time_evolution(state)

        ##############
        ### 4. Log ###
        ##############

        if step_count % nlog == 0:
            write_log_entry(state, start_step)

    if state.t >= t_halt:
        if chatter:
            print("Simulation halted: max time exceeded")

def integrate_time_step(state, config, dt_prop, step_count, lum, a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work, Np1):
    """
    Advance state by one time step.
    Applies conduction, revirialization, updates time, and checks stability diagnostics.

    Arguments
    ---------
    state : State
        The current simulation state.
    config : Config
        Configuration object for simulation.
    dt_prop : float
        Proposed dt value returned by compute_time_step
    step_count : int
        Step count
    lum : ndarray (N+1,)
        Memory allocation for luminosity array
    a_alloc, b_alloc, c_alloc, y_alloc, x_alloc : ndarray (N-1,)
        Memory allocation for working arrays
    work : ndarray (N,)
        Memory allocation for working array
    Np1 : float
        Length of radial grid
    """

    # Store state attributes for fast access in loop and to pass into njit functions
    prec = config.prec
    sim  = config.sim
    init = config.init
    char = state.char

    a = float(sim.a); b = float(sim.b); c = float(sim.c)
    sigma_m = float(char.sigma_m_char)
    cored = (init.profile == 'abg') and (float(init.gamma) < 1.0)
    eps_du = float(prec.eps_du); eps_dr = float(prec.eps_dr)
    max_iter_dr = prec.max_iter_dr

    r       = np.asarray(state.r,   dtype=np.float64)
    m       = np.asarray(state.m,   dtype=np.float64)
    v2      = np.asarray(state.v2,  dtype=np.float64)
    rho     = np.asarray(state.rho, dtype=np.float64)

    # Compute total enclosed mass including baryons, perturbers, etc.
    # May need to move elsewhere depending on how m is updated
    # Current version just returns m as is
    m_tot = compute_mass(m)

    ### Step 1: Energy transport ###
    compute_luminosities(a, b, c, sigma_m, r, v2, rho, lum, cored)
    du_max, dt_prop = conduct_heat(v2, m, lum, work, dt_prop, eps_du)
    p = rho * v2 # Needed for revir

    ### Step 2: Reestablish hydrostatic equilibrium ###
    iter_dr = 0
    while True:
        status, dr_max = revirialize(r, rho, p, m_tot, 
                                     a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work, Np1) # Modifies r, rho, p in place

        # Shell crossing signaled by None
        if status == STATUS_SHELL_CROSSING:
            raise RuntimeError("Max iterations exceeded for shell crossing in conduction/revirialization step")
        
        # Check dr criterion
        """
        With new step to ensure equilibrium in initialization, no longer a need to accept larger dr in first time step.
        If needed, can reintroduce with 'and (step_count != 1):' in the if statement below.
        """
        if dr_max > eps_dr:
            if iter_dr >= max_iter_dr:
                raise RuntimeWarning("Max iterations exceeded for dr in revirialization step")
            iter_dr += 1
            continue # Go to top of loop, repeat revirialize with new values

        # Break out of loop
        break

    ### Step 3: Update state variables ###
    # m not updated in Lagrangian code

    v2_new = p / rho
    state.r = r
    state.rho = rho
    state.p = p
    state.v2 = v2_new
    state.dr_max = dr_max
    state.du_max = du_max

    state.rmid = 0.5 * (r[1:] + r[:-1])
    state.kn = 1.0 / (sigma_m * np.sqrt(p))
    sqrt_v2_new = np.sqrt(v2_new)
    state.trelax = 1.0 / (sqrt_v2_new * rho)

    state.maxvel    = float(np.max(sqrt_v2_new))
    state.minkn     = float(np.min(state.kn))
    state.mintrelax = float(np.min(state.trelax))

    # Diagnostics
    state.n_iter_dr += iter_dr
    state.dt_cum += float(dt_prop)
    if step_count != 1:
        state.dr_max_cum += float(dr_max)
    state.du_max_cum += float(du_max)
    state.dt_over_trelax_cum += float(dt_prop / state.mintrelax)

    state.dt = float(dt_prop)
    state.t += float(dt_prop)