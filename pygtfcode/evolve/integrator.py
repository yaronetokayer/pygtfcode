import numpy as np
from pygtfcode.io.write import write_profile_snapshot, write_log_entry, write_time_evolution
from pygtfcode.evolve.transport import compute_luminosities, conduct_heat, conduct_implicit_dulim, conduct_implicit_tcool_dulim #, conduct_implicit_nolim, conduct_implicit_Theta_dulim, conduct_implicit_Theta_nolim
from pygtfcode.evolve.hydrostatic import revirialize, STATUS_SHELL_CROSSING #, compute_mass
from pygtfcode.evolve.split import check_drfrac, split_grid, STATUS_SPLITS
from pygtfcode.util.calc import low_kn_boost

def run_until_stop(state, start_step, **kwargs):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    ##################
    ### Set locals ###
    ##################

    # User halting criteria
    steps       = kwargs.get('steps', None)
    time_limit  = kwargs.get('stoptime', None)
    rho_c_limit = kwargs.get('rho_c', None)
    step_i      = state.step_count if steps is not None else None
    time_i      = state.t if time_limit is not None else None

    # Locals for speed + type hardening
    config = state.config
    io = config.io; sim = config.sim; prec = config.prec
    
    t_evol = bool(io.t_evol); profiles = bool(io.profiles); chatter = bool(io.chatter)
    t_halt = float(sim.t_halt); rho_c_halt = float(sim.rho_c_halt); grid_splitting = bool(sim.grid_splitting)
    if t_evol:
        rho0_last_tevol = float(state.rho[0])
        drho_tevol = float(io.drho_tevol)
    if profiles:
        rho0_last_prof = float(state.rho[0])
        drho_prof = float(io.drho_prof)
    nlog = int(io.nlog); nupdate = int(io.nupdate)
    drfrac_max = float(prec.drfrac_max)

    # For adaptive time-stepping
    safety = 0.99
    kn_threshold = prec.kn_threshold
    du_boost = prec.du_boost
    kn_width = prec.kn_width

    # Preallocate working arrays for main loop
    # Found that preallocating for conduction tridiagonal solve does not save time
    a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work_n1, work_n2, work_nint = allocate_work_arrays(state.n)

    #################
    ### Main loop ###
    #################

    while state.t < t_halt:

        ###########################
        ### 1. Integrate system ###
        ###########################

        #--- Increment counter
        state.step_count += 1
        step_count = state.step_count
        
        #--- Estimate the proposed du-limited dt using proportional control
        eps_du_eff = prec.eps_du * low_kn_boost(state.minkn, kn_threshold, du_boost, kn_width)

        if step_count == 1:
            dt_prop = 1.0 # We have no maxdu yet
        else:
            err = eps_du_eff / state.du_max
            fac = safety * err
            dt_prop = fac * state.dt

        #--- Check for cell-splitting
        if grid_splitting:
            status = check_drfrac(state.r, work_nint, drfrac_max)
            if status == STATUS_SPLITS:
                split_grid(state, work_nint)
                state.resize_state_arrays()
                a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work_n1, work_n2, work_nint = allocate_work_arrays(state.n)

        #--- Integrate time step
        integrate_time_step(state, config, dt_prop, eps_du_eff, step_count, 
                            a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work_n1, work_n2)

        if step_count % nupdate == 0:
            print(f"Completed step {step_count}", end='\r', flush=True)


        ###########################
        ### 2. Halting criteria ###
        ###########################
        rho0 = state.rho[0]

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
        if profiles:
            drho_for_prof = np.abs(rho0 - rho0_last_prof) / rho0_last_prof
            if drho_for_prof > drho_prof:
                rho0_last_prof = rho0
                write_profile_snapshot(state)
                write_time_evolution(state) # Always have a time evolution column concurrent with a profile

        # Track time evolution
        if t_evol: 
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

def integrate_time_step(state, config,                                  # State arrays
                        dt_prop, eps_du_eff, step_count,                # Instantaneous variables
                        a_alloc, b_alloc, c_alloc, y_alloc, x_alloc,    # Memory allocations (N-1,)
                        work_n1, work_n2                                # Memory allocations (N,)
                        ):
    """
    Advance state by one time step.
    Applies conduction, revirialization, updates time, and checks stability diagnostics.

    Arguments
    ---------
    state : State
        The current simulation state.
    config : Config
        Configuration object for simulation.
    eps_du_eff : float
        Effective du criterion for adaptive time-stepping, which may be relaxed in low-kn regime.
    dt_prop : float
        Proposed dt value returned by compute_time_step
    step_count : int
        Step count
    a_alloc, b_alloc, c_alloc, y_alloc, x_alloc : ndarray (N-1,)
        Memory allocation for working arrays
    work_n1, work_n2 : ndarray (N,)
        Memory allocation for working arrays
    """

    # Store state attributes for fast access in loop and to pass into njit functions
    prec = config.prec; sim  = config.sim; char = state.char

    a = float(sim.a); b = float(sim.b); c = float(sim.c)
    sigma_m = float(char.sigma_m_char)
    alph = float(sim.alph); implicit_conduct = bool(sim.implicit_conduct)
    eps_dr = float(prec.eps_dr)
    max_iter_du = prec.max_iter_du; max_iter_dr = prec.max_iter_dr

    # Pointers to state arrays for easy access
    r       = np.asarray(state.r,       dtype=np.float64)
    m       = np.asarray(state.m,       dtype=np.float64)
    v2      = np.asarray(state.v2,      dtype=np.float64)
    rho     = np.asarray(state.rho,     dtype=np.float64)
    # Theta   = np.asarray(state.Theta,   dtype=np.float64)
    t_cool  = np.asarray(state.t_cool,  dtype=np.float64)

    # Compute total enclosed mass including baryons, perturbers, etc.
    # May need to move elsewhere depending on how m is updated
    # Current version just returns m as is
    # m_tot = compute_mass(m)

    ### Step 1: Energy transport ###
    if implicit_conduct:
        # implicit: work_n1 used to store dv2
        # du_max, dt_prop, iter_du = conduct_implicit_dulim(v2, rho, r, m, work_n1, dt_prop, a, b, c, sigma_m, alph, eps_du_eff, max_iter_du)
        du_max, dt_prop, iter_du = conduct_implicit_tcool_dulim(v2, rho, r, m, work_n1, t_cool, dt_prop, a, b, c, sigma_m, alph, eps_du_eff, max_iter_du)
    else:
        # explicit: work_n1 used to store dv2dt; work_n2 used to store luminosity
        init = config.init; cored = (init.profile == 'abg') and (float(init.gamma) < 1.0)
        compute_luminosities(a, b, c, sigma_m, alph, r, v2, rho, work_n2, cored)
        du_max, dt_prop, iter_du = conduct_heat(v2, m, work_n2, work_n1, dt_prop, eps_du_eff)
    
    if iter_du == -1:
        raise RuntimeError(f"Step {step_count}: Max iterations exceeded in implicit conduction step.")
        
    np.multiply(rho, v2, out=work_n2) # work_n2 is used for p, needed for revir and to set v2 later

    ### Step 2: Reestablish hydrostatic equilibrium ###
    iter_dr = 0
    while True:
        # work_n1 used to store old shell volumes; work_n2 is p
        status, dr_max = revirialize(r, rho, work_n2, m, 
                                     a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work_n1) # Modifies r, rho, p in place

        # Shell crossing signaled by None
        if status == STATUS_SHELL_CROSSING:
            raise RuntimeError(f"Step {step_count}: Shell crossing in revirialization step")
        
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

    # r, rho, and theta were modified in place already; no need to assign them
    # Still need to update v2 based on the new p and rho
    np.divide(work_n2, rho, out=state.v2)

    # Update rmid without allocations
    np.add(r[1:], r[:-1], out=state.rmid)
    state.rmid *= 0.5

    # Update kn without allocations
    np.sqrt(work_n2, out=state.kn)
    state.kn *= sigma_m
    np.reciprocal(state.kn, out=state.kn)
    state.minkn = float(np.min(state.kn))

    # Diagnostics
    state.n_iter_du += iter_du
    state.n_iter_dr += iter_dr
    state.dt_cum += float(dt_prop)
    if step_count != 1:
        state.dr_max_cum += float(dr_max)
    state.du_max_cum += float(du_max)

    state.du_max    = float(du_max)
    state.dt        = float(dt_prop)
    state.t         += float(dt_prop)

    ### Time scales ###
    # work_n1 to store sqrt(v2)
    np.sqrt(state.v2, out=work_n1)

    np.subtract(r[1:], r[:-1], out=state.t_sc)
    np.divide(state.t_sc, work_n1, out=state.t_sc)

    np.multiply(rho, work_n1, out=state.t_coll)
    np.multiply(state.t_coll, sigma_m, out=state.t_coll)
    np.reciprocal(state.t_coll, out=state.t_coll)

    np.sqrt(rho, out=state.t_dyn)
    np.reciprocal(state.t_dyn, out=state.t_dyn)

    ### Other testing diagnostics ###
    np.subtract(r[1:], r[:-1], out=state.drfrac)
    np.divide(state.drfrac, state.rmid, out=state.drfrac)
    
    # Luminosity
    init = config.init; cored = (init.profile == 'abg') and (float(init.gamma) < 1.0)
    compute_luminosities(a, b, c, sigma_m, alph, r, v2, rho, state.lum, cored)

def allocate_work_arrays(n):
    n_int = n - 1

    a_alloc   = np.empty(n_int, dtype=np.float64)
    b_alloc   = np.empty(n_int, dtype=np.float64)
    c_alloc   = np.empty(n_int, dtype=np.float64)
    y_alloc   = np.empty(n_int, dtype=np.float64)
    x_alloc   = np.empty(n_int, dtype=np.float64)

    work_n1   = np.empty(n, dtype=np.float64)
    work_n2   = np.empty(n, dtype=np.float64)
    work_nint = np.zeros(n, dtype=np.int64)

    return a_alloc, b_alloc, c_alloc, y_alloc, x_alloc, work_n1, work_n2, work_nint