def step_forward(state):
    """
    Advance state by one time step. Applies conduction, PdV work,
    revirialization, updates time, and checks stability diagnostics.
    """
    # placeholder for now
    raise NotImplementedError("step_forward not yet implemented.")

def run_until_stop(state):
    """
    Repeatedly step forward until t >= tstop or halting criterion met.
    """
    while state.t < state.config.sim.tstop:
        state.step()
