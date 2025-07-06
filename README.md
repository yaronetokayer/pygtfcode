# pygtfcode

**pygtfcode** is a Python implementation of a 1D gravothermal fluid code, adapted from a Fortran code originally designed to study the dynamical evolution of dark matter halos under the influence of heat transport, self-interactions, and baryonic potentials.

The goal is to create a clean, modern, modular codebase that is easy to configure, extend, and use within scientific workflows.

---

## Project Status

Core configuration and initialization functionality is complete:

* Modular configuration with defaults
* Support for multiple initial profiles (NFW, truncated NFW, ABG)
* Derived characteristic parameters from input mass and concentration
* Grid setup and profile initialization routines (in progress)
* Runtime integration and simulation loop (next)
* Output, diagnostics, and visualization tools

---

## Core Concepts

The simulation is structured around **two main user-facing classes**:

### 1. `Config`

Holds all static input parameters for a simulation run. This includes:

* `io`: Output paths and model numbering
* `grid`: Radial domain and resolution
* `init`: Initial profile (NFW, truncated NFW, or ABG)
* `sim`: Simulation options (e.g. interaction cross section)
* `prec`: Precision settings (iteration limits, step sizes)

### 2. `State`

Holds the dynamically evolving quantities:

* Grid arrays: `r`, `m`, `rho`, `P`, `u`, `v2`
* Diagnostic quantities
* Time tracking
* Characteristic scales (computed from config)

The `State` object is initialized with a `Config` and handles the setup of the simulation state (via `set_param`, `setup_grid`, and `initialize_grid`).

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
│   ├── menc.py               # Enclosed mass for any profile
│   ├── nfw.py
│   └── truncated_nfw.py
│
├── io/                       # File output routines
│   └── write.py
│
├── runtime_backup.py         # Archived legacy runtime logic
└── __init__.py
```

---

## Example: Config usage

```python
from pygtfcode import Config

# Use all defaults (NFW profile, 300 grid points, etc.)
config = Config()

# Customize initial profile
config.init = "abg"                             # Use ABG with default params
config.init = ("abg", {"alpha": 3.5, "beta": 4.5})  # Custom ABG

# Customize grid and output directory
config.grid.Ngrid = 500
config.io.model_no = 42
config.io.base_dir = "/tmp/sims"

# Switch to a truncated NFW
config.init = ("truncated_nfw", {"Zt": 0.05, "deltaP": 1e-4})
```

---

## What's Next

* Implement `State._initialize_grid()` and related setup routines
* Integrate time-step evolution with a `Simulator` class
* Add output and diagnostic plotting tools
* Write example notebooks

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/yourname/pygtfcode.git
cd pygtfcode
pip install -e .
```

Requires Python 3.8+, `numpy`, `scipy`, and optionally `matplotlib` for plotting.

---

## Contributors

* Yarone Tokayer

---

## License

MIT License. See [LICENSE](./LICENSE) for details.