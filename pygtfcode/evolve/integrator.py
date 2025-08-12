import numpy as np

def run_until_stop(state, start_step, **kwargs):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    from pygtfcode.io.write import write_profile_snapshot, write_log_entry, write_time_evolution

    # User halting criteria
    steps = kwargs.get('steps', None)
    time_limit = kwargs.get('stoptime', None)
    rho_c_limit = kwargs.get('rho_c', None)
    step_i = state.step_count if steps is not None else None
    time_i = state.t if time_limit is not None else None

    # Store attributes faster access in loop
    io = state.config.io
    chatter = state.config.io.chatter
    sim = state.config.sim
    t_halt = sim.t_halt
    rho0_last_prof = state.rho[0]
    rho0_last_tevol = state.rho[0]
    rho_c_halt = sim.rho_c_halt
    drho_prof = io.drho_prof
    drho_tevol = io.drho_tevol
    nlog = io.nlog

    while state.t < t_halt:

        # Increment counter
        state.step_count += 1
        step_count = state.step_count

        # Compute proposed dt
        dt_prop = compute_time_step(state)

        # Integrate time step
        integrate_time_step(state, dt_prop, step_count)

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

        # Check I/O criteria
        # Write profile to disk
        drho_for_prof = np.abs(rho0 - rho0_last_prof) / rho0_last_prof
        # if drho_for_prof > drho_prof:
        # if step_count % 5000 == 0 or step_count in [992857, 992858, 992859, 992860, 992861, 992862, 992863]:
        if step_count % 5000 == 0 or step_count in [468332, 468333, 468334, 468335, 468336, 468337]: # FOR DEBUGGING
            rho0_last_prof = rho0
            write_profile_snapshot(state)

        # Track time evolution 
        drho_for_tevol = np.abs(rho0 - rho0_last_tevol) / rho0_last_tevol
        if drho_for_tevol > drho_tevol:
            rho0_last_tevol = rho0
            write_time_evolution(state)

        # Log
        if step_count % nlog == 0:
            write_log_entry(state, start_step)

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
    if state.step_count == 1:
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

    # Store state attributes for fast access in loop and to pass into njit functions
    prec = state.config.prec
    sim = state.config.sim
    init = state.config.init
    a = sim.a
    b = sim.b
    c = sim.c
    sigma_m = state.char.sigma_m_char
    cored = init.profile == 'abg' and init.gamma < 1.0
    r_orig = state.r
    m = state.m
    v2_orig = state.v2
    p_orig = state.p
    u_orig = state.u
    rho_orig = state.rho

    # Compute current luminosity array
    lum = compute_luminosities(a, b, c, sigma_m, r_orig, v2_orig, p_orig, cored)
    # np.save('/Users/yaronetokayer/YaleDrive/Research/SIDM/pygtfcode/tests/arr.npy', lum) # FOR DEBUGGING

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
        p_cond, du_max_new = conduct_heat(m, u_orig, rho_orig, lum, dt_prop)

        # Check du criterion
        if du_max_new > eps_du:
            if iter_du >= max_iter_du:
                raise RuntimeError("Max iterations exceeded for du in conduction step")
            dt_prop *= 0.95 * (eps_du / du_max_new)
            iter_du += 1
            continue # Do not complete outer while loop, repeat conduct_heat with original values and smaller timestep

        ### Step 2: Reestablish hydrostatic equilibrium ###
        while True:
            if repeat_revir:
                result = revirialize(r_new, rho_new, p_new, m)
            else:
                result = revirialize(r_orig, rho_orig, p_cond, m)

            # Check v2 criterion
            # if result is None: # Negative v2 value
            if len(result) == 4: # FOR DEBUGGING
                if iter_v2 >= max_iter_v2:
                    np.save('/Users/yaronetokayer/YaleDrive/Research/SIDM/pygtfcode/tests/Model001/p.npy', result[1]) # FOR DEBUGGING
                    np.save('/Users/yaronetokayer/YaleDrive/Research/SIDM/pygtfcode/tests/Model001/r_new.npy', result[0]) # FOR DEBUGGING
                    np.save('/Users/yaronetokayer/YaleDrive/Research/SIDM/pygtfcode/tests/Model001/rho.npy', result[2]) # FOR DEBUGGING
                    np.save('/Users/yaronetokayer/YaleDrive/Research/SIDM/pygtfcode/tests/Model001/v2.npy', result[3]) # FOR DEBUGGING
                    raise RuntimeError("Max iterations exceeded for v2 in conduction/revirialization step")
                dt_prop *= 0.5
                iter_v2 += 1
                repeat_revir = False
                break # Exit inner loop, repeat conduct_heat with original values and smaller timestep
            
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

    if np.any(np.diff(r_new) < 0):
        print(f"r_new has a negative diff!! {step_count}")
        print(np.diff(r_new)[:4])

    state.r = r_new
    state.rho = rho_new
    state.p = p_new
    state.v2 = v2_new
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

    # Diagnostics
    state.n_iter_du += iter_du
    state.n_iter_v2 += iter_v2
    state.n_iter_dr += iter_dr
    state.dt_cum += dt_prop

    state.dt = dt_prop
    state.t += dt_prop