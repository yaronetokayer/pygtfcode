# pygtfcode

**pygtfcode** is a modern Python implementation of a 1D Lagrangian gravothermal fluid code. It simulates the dynamical evolution of self-interacting dark matter halos using the fluid approximation, based on a Fortran code originally developed by Prof. Frank van den Bosch (Yale University).

This implementation follows the formalism outlined in [Nishikawa et al. (2020)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.063009), with modular components for initialization, evolution, and output.

The notebooks in `examples/` predate the current API. Use the examples below and the package docstrings for current parameter names.

Contact yarone.tokayer-at-yale.edu with any questions/comments.

---

## Overview

The code is organized around **two user-facing classes**:

### 1. `Config`

Stores static input parameters, grouped into modules:

* `io`: Output paths and snapshot cadence
* `grid`: Grid resolution and domain
* `init`: Initial profile (NFW, truncated NFW, or $\alpha$-$\beta$-$\gamma$)
* `sim`: Physical parameters and simulation option (e.g., self-interaction cross section, calibrated parameters)
* `prec`: Precision tolerances and iteration limits
* `cosmo`: Cosmological parameters

### 2. `State`

Holds the dynamically evolving quantities:

* Radial grid and shell midpoints: `r`, `rmid`
* Physical variables: `m`, `rho`, `v2` (pressure is `rho*v2`; specific internal energy is `1.5*v2`)
* Diagnostics: `kn`, `mfp`, `kn_cond`, `mfp_cond`, `kn_c`, `kn_cond_c`, `ltemp`, `t_cool`, `t_dyn`, `du_max`
* Time tracking: `t`, `dt`, `step_count`, `snapshot_index`
* Characteristic scales (derived from `Config`)
* A `run()` method to evolve the system until collapse or a stopping condition is reached

Construct an initialized state with `State.from_config(config)`. Optionally pass
`ic_filepath` to start a new run from a saved profile. `State.from_dir()` is not
implemented and raises an error; it cannot resume a saved simulation.

`r` and `m` are edge arrays of length N+1. `rho`, `v2`, and the cell diagnostics
have length N and are associated with `rmid`.

In addition to these two classes, there are three plotting functions that are automatically imported, each of which also exists as a method of `State`:

### 1. `plot_time_evolution()`

Plot the time evolution of any one of a number of system-wide quantities.  Multiple simulations can be passed to compare their evolutions on the same plot.

### 2. `plot_snapshots()`

Plot up to three profiles of one simulation at specified points in time.

### 3. `make_movie()`

Animate the full evolution of up to three profiles.  This requires ffmpeg to be installed and callable with `ffmpeg` from the working directory.

Use `help()` for documentation on any of these classes, methods, and functions.

---

## Getting started

### Installation

```bash
git clone https://github.com/yaronetokayer/pygtfcode.git
cd pygtfcode
pip install -e .
```

Dependencies: Python 3.12+, `numpy`, `scipy`, `numba`, `matplotlib`, `tqdm`

### Example usage

```python
import pygtfcode as gtf

config = gtf.Config()
state = gtf.State.from_config(config)
state.run()
```

To read an existing simulation without constructing a state:

```python
from pygtfcode.io.read import load_snapshot_bundle
snapshot = load_snapshot_bundle('./sims/Model00002')  # Latest saved snapshot
snapshot = load_snapshot_bundle('./sims/Model00002', snapshot=59)
```

Alternatively, you can call `from pygtfcode import Config, State`.  In that case, plotting functions will not be automatically imported.

We can also run for a specified duration:
```python
state.run(steps=100) # Run for 100 simulation steps
state.run(stoptime=55.0) # Run for a duration of 55.0 simulation time units
state.run(rho_c=500.0) # Run until the central density exceeds 500.0
```

The first satisfied user or configuration stopping condition ends the run. The configured density limit is checked only after time > 50; `run(rho_c=...)` applies immediately after a step. Multiple `run()` commands can be executed in succession, and each will continue from the current state.  Call `state.reset()` to reset the state to its initial condition and reset the step counter.

To customize defaults:

```python
# Customize initial profile
config.init = "abg"                                 # Use ABG with default params
config.init = ("abg", {"alpha": 3.5, "beta": 4.5})  # Custom ABG

# Customize other configuration parameters
config.grid.drfrac_init = 0.05                    # Initial dr/r sets the cell count
config.io.model_no = 42
config.io.base_dir = "/tmp/sims"                    # Default is the current working directory
config.sim.sigma_m_0 = 1.0                         # Low-velocity amplitude [cm^2/g]
config.sim.w = 50.0                                # Velocity scale [km/s]; default inf is constant scattering
config.sim.smfp_order = 2                          # Normalized second order (default); 1 is also supported

# Switch to a truncated NFW
config.init = ("truncated_nfw", {"Zt": 0.05, "deltaP": 1e-4})

# Turn off chatter
config.io.chatter = False
```

If you don’t explicitly assign `config.io.model_no`, it is automatically set to the next available model number in `config.io.base_dir` (e.g., if `Model00000`, `Model00001`, and `Model00002` directories exist, it will assign `model_no = 3`) when the model number is first accessed. You can explicitly assign a `model_no` with `config = gtf.Config(io={"model_no": 5})` or with `config.io.model_no = 5` once config is instantiated.  Note that in that case, existing outputs may be overwritten. Set `config.io.overwrite = False` to reject initialization in an existing model directory.

### Plotting

There are three plotting functions that are imported with the `pygtfcode` package:

The `plot_time_evolution()` function plots the evolution of system-wide parameters over time.  It can plot any of the columns in the `time_evolution.txt` output.

```python
import pygtfcode as gtf

# Compare the central density evolution of two different simulations
# The default plot is rho0 (the innermost-cell density)
gtf.plot_time_evolution([state1, state2], show=True)

# Alternatively, the simulations can be called by their Config objects or by their model numbers
# v_max is in characteristic velocity units v_s
gtf.plot_time_evolution([config1, config2], quantity="v_max", show=True)

# base_dir needs to be specified if simulations are called by model number:
gtf.plot_time_evolution([5, 6], quantity="kn_cond_c", base_dir='./', show=True) # This is useful for simulations run in a different session

# The plot can be saved to a file
# Use 'show' to show the figure in standard output as well
gtf.plot_time_evolution(state1, filepath='./rho_c_vs_time.png', show=True)
```

The `plot_snapshots()` function plots up to three profiles in separate panels for one or multiple snapshots of the simulation.  Snapshots are specified by the index of the `profile_x.dat` file.  Like `plot_time_evolution()`, the State object, Config object, or model number can be used to specify the simulation you wish to plot.

```python
import pygtfcode as gtf

# Plot the initial density profile
gtf.plot_snapshots(state, show=True)

# Plot the mass profile at a specified snapshot
gtf.plot_snapshots(config, snapshots=50, profiles='m', show=True)

# Plot the density, v^2, and Knudsen number profiles, comparing several snapshots
gtf.plot_snapshots(4, snapshots=[0, 50, 100], profiles=['rho', 'v2', 'kn_cond'], base_dir='./', show=True)

# The plot can be saved to a file
# Use 'show=True' to show the figure in standard output as well
gtf.plot_snapshots(state, filepath='./initial_rho.png')
```

The `make_movie()` function generates animations of up to three profiles in separate panels for a simulation.  Like `plot_time_evolution()`, the State object, Config object, or model number can be used to specify the simulation you wish to plot.  Only the snapshots of the most recent run for the simulation model_no will be included, even if profiles with higher indices are in the directory from previous simulation runs.  You can check the current version of the `snapshot_conversion.txt` file for all snapshots that will be included in the animation

```python
import pygtfcode as gtf

# Plot the density profile
gtf.make_movie(state)

# All keyword arguments available in plot_snapshots function, other than 'snapshots', can be used here
gtf.make_movie(2, base_dir='./', profiles=['v2', 'p'], grid=True)
```

The plotting functions also exist as methods to the `State` object:

```python
import pygtfcode as gtf

config = gtf.Config()
state = gtf.State.from_config(config)
state.run()

state.plot_time_evolution(show=True)         # Accepts all keyword arguments, but cannot compare between simulations when used this way

state.plot_snapshots(show=True)              # Defaults to the latest state
state.plot_snapshots(snapshots=0, filepath="./initial_profs.png")   # Plot and save initial profiles
# Note that while the standalone function defaults to the initial profile, the `State` method defaults to the current state.

state.make_movie(profiles=['rho', 'kn', 'v2'])
```

---

## Output files

All outputs are written to the directory specified by `config.io.base_dir` and `model_no`.

### 1. `model_metadata.txt`

Stores all information about the simulation model for reference.  Unpacks all attributes of the `Config` object that instantiated the `State`.

### 2. `logfile.txt`

Logs relevant quantities every `nlog` steps (set in `config.io`).  If `chatter` is set to `True`, then these are also output to the console.

### 3. `profile_x.dat`

Radial profiles of all fluid variables, written each time the central density changes by a fractional amount `drho_prof` (set in `config.io`). The suffix `x` is the snapshot index.  `snapshot_conversion.txt` stores the conversion between the snapshot index `x` and simulation time.

Column names are written in the header and read dynamically. They include
`log_r`, `log_rmid`, `m`, `rho`, `v2`, amplitude-reference `kn`/`mfp`,
conductivity-effective `kn_cond`/`mfp_cond`, temperature-gradient diagnostics,
cell/face conductivities and their SMFP/LMFP ratios, and `x`, `K_L`, `K_S`,
`mfp_cond_ltemp`. `log_r` is the outer shell edge; face conductivity columns
use that coordinate. Radii and masses are in r_s and m_s units.

### 4. `time_evolution.txt`

Contains both core definitions (`_c`, half central density; `_m2`, density
slope -2), including `x`, `K_L`, and `K_S` evaluated at their mean dispersions.
`time`, `dt`, and `tsc_c` use t_s units; `time_Gyr` is physical time.
Finalization adds core-evolution slopes. Successive `run()` calls retain
earlier rows and regenerate these slopes.

Records the time evolution of relevant quantities, written each time the central density changes by a fractional amount `drho_tevol` (set in `config.io`).

---

## Package Layout

```
pygtfcode/
├── config.py               # Defines 'Config' class
├── state.py                # Defines 'State' class
│
├── parameters/             # Parameter subclasses for 'Config' attributes
│   ├── char_params.py
│   ├── constants.py
│   ├── grid_params.py
│   ├── init_params.py
│   ├── io_params.py
│   ├── prec_params.py
│   └── sim_params.py
│
├── profiles/               # Profile specific tools to set initial conditions of 'State'
│   ├── abg.py
│   ├── nfw.py
│   ├── truncated_nfw.py
│   └── profile_routines.py
│
├── evolve/                 # Integration and solver, used by 'State' methods
│   ├── integrator.py
│   ├── transport.py
│   └── hydrostatic.py
│
├── io/                     # I/O routines
│   └── write.py
│   └── read.py
│
├── plot/                   # Plotting routines
│   ├── time_evolution.py
│   └── snapshot.py
```

---

## Next steps

* v1.0 is complete, v2.0 in development
* v2.0 will be able to accommodate multiple species

---

## License

MIT License. See [LICENSE](./LICENSE) for details.


### Movie options and logfile widths

`make_movie()` now uses the deluxe renderer; `make_movie_deluxe()` remains
an alias. The default profiles are `['rho', 'v2']`, with an `rho0` inset
in the first panel. Set `parallel=False` for serial rendering.

```python
state.make_movie(profiles=['rho', 'v2'], insets=False, parallel=False)
state.make_movie(profiles=['rho', 'v2'], insets=['rho0', None])
```

`insets=False` disables every inset; `insets=None` preserves the default,
matching pygtf2. Lists allow individual panels to omit insets. A movie with
no insets or radius annotations needs only profile files
and `snapshot_conversion.txt`, not `time_evolution.txt`.

Logfile widths are set directly in the `columns` list inside
`pygtfcode/io/write.py` → `write_log_entry()`. Each entry contains
`(header, value, minimum_width)`, for example:

```python
('step', step, 10),
('time', state.t, 13),
('n', state.n, 6),
```

Edit the third value to adjust a column. Headers and values expand as needed
and are never truncated; floating-point precision remains six decimal
places in scientific notation. This controls both the file and console output.
