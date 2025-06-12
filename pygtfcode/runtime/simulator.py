# Early on, we will call this:

# from pygtfcode.runtime.initialize import setup_grid, initialize_grid
# from pygtfcode.runtime.runtime_state import RuntimeState

# r = setup_grid(config)
# if the initial profile is truncated_nfw, we will also need to compute the potential grid
# if config.init.profile == "truncated_nfw":
#     rad, pot = integrate_potential(config)
# fields = initialize_grid(r, config)

# state = RuntimeState(r=r, **fields)