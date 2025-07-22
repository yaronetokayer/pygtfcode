def integrate_time_step(state):
    """
    Advance state by one time step. Applies conduction, PdV work,
    revirialization, updates time, and checks stability diagnostics.
    """
    from pygtfcode.evolve.transport import compute_luminosities, conduct_heat
    from pygtfcode.evolve.hydrostatic import revirialize

    lum = compute_luminosities(state)

    i = 0
    while True:
        # Apply conduction to compute new internal energy and pressure
        u_new, p_new, du_max_new = conduct_heat(state, lum)

        # Accept step if relative energy change is within tolerance
        if du_max_new <= state.config.prec.eps_du:
            state.u = u_new
            state.p = p_new
            state.du_max = du_max_new
            break

        # Otherwise, reduce timestep and retry
        i += 1
        if i >= state.config.prec.max_iter_du:
            raise RuntimeError("No convergence achieved in heat conduction step")
        
        # Adjust time and reduce dt based on relative error
        state.t -= state.dt
        state.dt *= 0.95 * (state.config.prec.eps_du / du_max_new)
        state.t += state.dt 

def run_until_stop(state):
    """
    Repeatedly step forward until t >= tstop or halting criterion met.
    """
    while state.t < state.config.sim.tstop:
        state.dt = compute_time_step(state)
        integrate_time_step(state)

def compute_time_step(state) -> float:
    """
    Compute time step to be used for integration step.

    Parameters
    ----------
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
        # Relaxation-limited time step
        dt1 = state.config.prec.eps_dt * state.mintrelax
        # Energy stability-limited time step
        dt2 = state.dt * 0.95 * (state.config.prec.eps_du / state.du_max)

        return min(dt1, dt2)