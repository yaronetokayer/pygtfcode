import numpy as np

def run_until_stop(state):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    from pygtfcode.io.write import write_profile_snapshot, write_log_entry

    io = state.config.io
    chatter = state.config.io.chatter
    sim = state.config.sim
    t_halt = sim.t_halt
    rho0_last_prof = state.rho[0]
    rho0_last_tevol = state.rho[0]

    while state.t < t_halt:

        # Compute proposed dt
        dt_prop = compute_time_step(state)

        # Increment counter
        state.step_count += 1
        step_count = state.step_count

        # Integrate time step
        integrate_time_step(state, dt_prop, step_count)

        rho0 = state.rho[0]

        # Check halting criteria
        if rho0 > sim.rho_c_halt:
            if chatter:
                print("Simulation halted: central density exceeds halting value")
            break
        if np.isnan(rho0):
            if chatter:
                print("Simulation halted: central density is nan")
            pass

        # Check I/O criteria
        # Write profile to disk
        drho_for_prof = np.abs(rho0 - rho0_last_prof) / rho0_last_prof
        if drho_for_prof > io.drho_prof:
            rho0_last_prof = rho0
            write_profile_snapshot(state)
            state.snapshot_index += 1

        # Track time evolution
        drho_for_tevol = np.abs(rho0 - rho0_last_tevol) / rho0_last_tevol
        if drho_for_tevol > io.drho_tevol:
            rho0_last_tevol = rho0
            write_time_evolution(state)

        # Log
        if step_count % io.tlog == 0:
            write_log_entry(state)

    if state.t >= t_halt:
        if chatter:
            print("Simulation halted: max time exceeded")

def compute_time_step(state) -> float:
    """
    Compute time step to be used for integration step.

    Arguments
    ---------
    state : State
        The current simulation state.

    Returns
    -------
    float
        The recommended time step.
    """
    if state.step_count == 0:
        return 1.0e-7
    
    else:
        prec = state.config.prec
        # Relaxation-limited time step
        dt1 = prec.eps_dt * state.mintrelax
        # Energy stability-limited time step
        dt2 = state.dt * 0.95 * (prec.eps_du / state.du_max)

        return min(dt1, dt2)

def integrate_time_step(state, dt_prop, step_count):
    """
    Advance state by one time step.
    Applies conduction, revirialization, updates time, and checks stability diagnostics.

    Arguments
    ---------
    state : State
        The current simulation state.
    dt_prop : float
        Proposed dt value returned by compute_time_step
    step_count : int
        Step count
    """
    from pygtfcode.evolve.transport import compute_luminosities, conduct_heat
    from pygtfcode.evolve.hydrostatic import revirialize

    # Store state variables for fast access
    prec = state.config.prec
    sim = state.config.sim
    init = state.config.init
    a = sim.a
    b = sim.b
    c = sim.c
    sigma_m = state.char.sigma_m_char
    r_orig = state.r
    m = state.m
    v2_orig = state.v2
    p_orig = state.p
    u_orig = state.u
    rho_orig = state.rho
    cored = init.profile == 'abg' and init.gamma < 1.0

    # Compute current luminosity array
    lum = compute_luminosities(a, b, c, sigma_m, r_orig, v2_orig, p_orig, cored)

    iter_du = 0
    iter_v2 = 0
    iter_dr = 0
    eps_du = prec.eps_du
    eps_dr = prec.eps_dr
    max_iter_du = prec.max_iter_du
    max_iter_v2 = prec.max_iter_v2
    max_iter_dr = prec.max_iter_dr
    converged = False
    repeat_revir = False
    while not converged:
        ### Step 1: Energy transport ###
        u_new, p_cond, du_max_new = conduct_heat(m, u_orig, rho_orig, lum, dt_prop)

        # Check du criterion
        if du_max_new > eps_du:
            if iter_du >= max_iter_du:
                raise RuntimeError("Max iterations exceeded for du in conduction step")
            dt_prop *= 0.95 * (eps_du / du_max_new)
            iter_du += 1
            continue

        ### Step 2: Reestablish hydrostatic equilibrium ###
        while True:
            if repeat_revir:
                result = revirialize(r_new, rho_new, p_new, m)
            else:
                result = revirialize(r_orig, rho_orig, p_cond, m)

            # Check v2 criterion
            if result is None: # Negative v2 value
                if iter_v2 >= max_iter_v2:
                    raise RuntimeError("Max iterations exceeded for v2 in conduction/revirialization step")
                dt_prop *= 0.5
                iter_v2 += 1
                repeat_revir = False
                break # Exit inner loop, repeat conduct heat with original values and smaller timestep
            
            # Check dr criterion
            # Accept larger dr in first time step
            if (result[4] > eps_dr) and (step_count != 1):
                if iter_dr >= max_iter_dr:
                    raise RuntimeWarning("Max iterations exceeded for dr in revirialization step")
                iter_dr += 1
                r_new, rho_new, p_new, _, _ = result
                repeat_revir = True
                continue # Go to top of inner loop, repeat revirialize with new values

            else: # Both criteria are met, break out of inner and outer loop
                converged = True
                break

    ### Step 3: Update state variables ###
    # m not updated in Lagrangian code

    r_new, rho_new, p_new, v2_new, dr_max_new = result

    state.r = r_new
    state.rho = rho_new
    state.p = p_new
    state.v2 = v2_new
    state.u = u_new
    state.dr_max = dr_max_new
    state.du_max = du_max_new

    state.rmid = 0.5 * (r_new[1:] + r_new[:-1])
    state.u = 1.5 * v2_new
    state.kn = 1.0 / (state.char.sigma_m_char * np.sqrt(p_new))
    sqrt_v2_new = np.sqrt(v2_new)
    state.trelax = 1.0 / (sqrt_v2_new * rho_new)

    state.maxvel = np.max(sqrt_v2_new)
    state.minkn = np.min(state.kn)
    state.mintrelax = np.min(state.trelax)

    state.dt = dt_prop
    state.t += dt_prop
