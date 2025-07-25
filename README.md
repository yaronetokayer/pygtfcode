# pygtfcode

**pygtfcode** is a modern Python implementation of a 1D Lagrangian gravothermal fluid code. It simulates the dynamical evolution of self-interacting dark matter halos using the fluid approximation, based on a Fortran code originally developed by Prof. Frank van den Bosch (Yale University).

This implementation follows the formalism outlined in Nishikawa et al. (2020), with modular components for initialization, evolution, and output.

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

### 2. `State`

Holds the dynamically evolving quantities:

* Radial grid and shell midpoints: `r`, `rmid`
* Physical variables: `m`, `rho`, `P`, `u`, `v2`, `kn`
* Time tracking: `t`, `dt`, `step_count`, `snapshot_index`, etc.
* Characteristic scales (derived from `Config`)
* A `run()` method to evolve the system until collapse or a stopping condition is reached

The `State` object is initialized with a `Config` object.  The `Config` object then becomes an attribute of `State`.  In this way, multiple `State`s can be instantiated with different `Config` objects.

---

## Getting started

### Installation

```bash
git clone https://github.com/yaronetokayer/pygtfcode.git
cd pygtfcode
pip install -e .
```

Dependencies: Python 3.8+, `numpy`, `scipy`, `numba`.

### Minimal example

```python
from pygtfcode import Config, State

config = Config()
state = State(config)
state.run()
```

We can also run for a specified duration:
```python
state.run(steps=100) # Run for 100 simulation steps
state.run(time=55.0) # Run for a duration of 55.0 simulation time units
state.run(rho_c=500.0) # Run until the central density exceeds 500.0
```

Halting criteria in `config` override these.

To customize defaults:

```python
# Customize initial profile
config.init = "abg"                                 # Use ABG with default params
config.init = ("abg", {"alpha": 3.5, "beta": 4.5})  # Custom ABG

# Customize other configuration parameters
config.grid.ngrid = 200
config.io.model_no = 42
config.io.base_dir = "/tmp/sims"
config.sim.sigma_m = 1.0

# Switch to a truncated NFW
config.init = ("truncated_nfw", {"Zt": 0.05, "deltaP": 1e-4})

# Turn off chatter
config.io.chatter = False
```
## Output files

All outputs are written to the directory specified by `config.io.base_dir` and `model_no`.

### 1. `model_metadata.txt`

Stores all information about the simulation model for reference.  Unpacks all attributes of the `Config` object that instantiated the `State`.

### 2. `logfile.txt`

Logs relevant quantities every `nlog` steps (set in `config.io`).  If `chatter` is set to `True`, then these are also output to the console.

### 3. `profile_x.dat`

Radial profiles of all fluid variables, written each time the central density changes by a fractional amount `drho_prof` (set in `config.io`). The suffix `x` is the snapshot index.  `snapshot_conversion.txt` stores the conversion between the snapshot index `x` and simulation time.

Each row contains:

```
i   log(r_i)   log(rmid_i)   m_i   rho_i   v2_i   trelax_i   kn_i
```

### 4. `time_evolution.txt`

Records the time evolution of relevant quantites, written each time the central density changes by a fractional amount `drho_tevol` (set in `config.io`).

---

## Package Layout

```
pygtfcode/
├── config.py               # Defines Config class
├── state.py                # Defines State class and evolution interface
│
├── parameters/             # Parameter subclasses
│   ├── char_params.py
│   ├── constants.py
│   ├── grid_params.py
│   ├── init_params.py
│   ├── io_params.py
│   ├── prec_params.py
│   └── sim_params.py
│
├── profiles/               # Profile and phase-space tools
│   ├── abg.py
│   ├── nfw.py
│   ├── truncated_nfw.py
│   └── profile_routines.py
│
├── evolve/                 # Integration and solver
│   ├── integrator.py
│   ├── transport.py
│   └── hydrostatic.py
│
├── io/                     # Output routines
│   └── write.py
```

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
