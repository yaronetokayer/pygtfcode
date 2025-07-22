# pygtfcode

**pygtfcode** is a Python implementation of a 1D gravothermal fluid code, adapted from a Fortran code by Frank van den Bosch originally designed to study the dynamical evolution of dark matter halos under the influence of heat transport, self-interactions, and baryonic potentials.

The goal is to create a clean, modern, modular codebase that is easy to configure, extend, and use within scientific workflows.

---

## Project Status

Core configuration and initialization functionality is complete:

* Modular configuration with defaults
* Support for multiple initial profiles (NFW, truncated NFW, ABG)
* Automatic computation of characteristic physical scales
* Grid setup and profile initialization
* Velocity dispersion and potential generation using phase-space integrals
* Runtime integration and simulation loop (in progress)
* Output, diagnostics, and visualization tools (in progress)

---

## Core Concepts

The simulation is structured around **two main user-facing classes**:

### 1. `Config`

Holds all static input parameters for a simulation run. This includes:

* `io`: Output paths and model metadata
* `grid`: Radial domain and resolution
* `init`: Initial profile (NFW, truncated NFW, or $\alpha$-$\beta$-$\gamma$)
* `sim`: Simulation options (e.g. interaction cross section)
* `prec`: Precision settings (e.g., step tolerances)

### 2. `State`

Holds the dynamically evolving quantities:

* Radial grid and shell midpoints: `r`, `rmid`
* Physical variables: `m`, `rho`, `P`, `u`, `v2`, `kn`
* Time tracking: `t`, `dt`, `step_count`, `snapshot_index`, etc.
* Characteristic scales (computed from `Config`)
* Profile interpolation functions (e.g., `rho_interp`, `pot_interp`)

The `State` object is initialized with a `Config` and handles setup internally via methods like `_set_param()`, `_setup_grid()`, and `_initialize_grid()`.

---

## Structure Overview

```
pygtfcode/
├── config.py                 # Main Config class
├── state.py                  # Main State class
│
├── parameters/               # Configuration parameter classes
│   ├── char_params.py        # Derived characteristic scales
│   ├── constants.py          # Physical and cosmological constants
│   ├── grid_params.py
│   ├── init_params.py        # NFW / ABG / truncated NFW
│   ├── io_params.py
│   ├── prec_params.py
│   └── sim_params.py
│
├── profiles/                 # Profile-specific helper functions
│   ├── abg.py
│   ├── nfw.py
│   ├── truncated_nfw.py
│   └── profile_routines.py   # Shared utilities (menc, sigr, etc.)
│
├── io/                       # File output routines
│   └── write.py
│
├── evolve/                   # Time integration and evolution routines
    └── integrator.py
```

---

## Example: Config usage

```python
from pygtfcode import Config

# Use all defaults (NFW profile, 300 grid points, etc.)
config = Config()

# Customize initial profile
config.init = "abg"                                 # Use ABG with default params
config.init = ("abg", {"alpha": 3.5, "beta": 4.5})  # Custom ABG

# Customize grid and output directory
config.grid.ngrid = 200
config.io.model_no = 42
config.io.base_dir = "/tmp/sims"

# Switch to a truncated NFW
config.init = ("truncated_nfw", {"Zt": 0.05, "deltaP": 1e-4})
```

---

## What's Next

* Implement output routines for logging and snapshot writing
* Complete `step()` and `run()` time integration interface
* Add visualization tools and Jupyter notebooks
* Build test coverage and CI integration

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/yaronetokayer/pygtfcode.git
cd pygtfcode
pip install -e .
```

Requires Python 3.8+, `numpy`, `scipy`, `numba`, and optionally `matplotlib` for plotting.

---

## License

MIT License. See [LICENSE](./LICENSE) for details.