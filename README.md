# pygtfcode

**pygtfcode** is a Python implementation of the gravothermal fluid code used to study the thermodynamic evolution of self-gravitating dark matter halos. It is a modular, NumPy-based reworking of a legacy Fortran code originally developed by Frank van den Bosch.

---

## Project Status

This project is currently in **active development**. The framework for configuring, initializing, and setting up static profiles is complete. The next milestone is testing the initial profile routines and integrating the simulator loop.

---

## Structure Overview

```
pygtfcode/
├── config.py               # Central class for organizing simulation parameters
├── parameters/             # All user-defined static parameter classes
│   ├── io_params.py        # I/O configuration (base directory, model #)
│   ├── grid_params.py      # Grid resolution and radial range
│   ├── init_params.py      # Initial profile (NFW, truncated NFW, ABG)
│   ├── prec_params.py      # Precision tuning parameters
│   ├── sim_params.py       # Simulation control flags (to be implemented)
│   ├── constants.py        # Physical constants (modifiable if needed)
│   └── __init__.py
├── profiles/               # Profile-specific logic (rho, Menc, sigma2, etc.)
│   ├── nfw.py              # Analytic NFW profile formulas
│   ├── truncated_nfw.py    # DF-based truncated profile setup
│   ├── abg.py              # Placeholder for ABG profile logic
│   ├── menc.py             # Profile-aware dispatch for M(<r)
│   ├── v2.py               # (planned) Velocity dispersion dispatch
│   └── __init__.py
├── runtime/                # Runtime simulation logic
│   ├── initialize.py       # setup_grid() and initialize_grid()
│   ├── runtime_state.py    # Class for dynamic state variables (rho, u, P, etc.)
│   ├── simulator.py        # (planned) Main time integration loop
│   └── __init__.py
├── io/                     # File I/O routines (write_output, etc.)
│   ├── write.py
│   └── __init__.py
└── utils/                  # Placeholder for helper routines
    └── __init__.py
```

---

## Core Concepts

### `Config`

A single container class that manages:

* Static parameter classes (IO, grid, profile, simulation, precision)
* Global constants
* Flexible override of any defaults

### `InitParams`

Supports:

* `NFWParams`
* `TruncatedNFWParams`
* `ABGParams`
  Each subclass defines what parameters it needs (e.g., `Zt`, `deltaP` for truncated NFW).

### `setup_grid()` and `initialize_grid()`

* Constructs the radial grid in log-space.
* Initializes `M(r)`, `rho(r)`, `P(r)`, `u(r)`, and `v²(r)` based on the selected profile.

### `integrate_potential()`

For truncated NFW only, solves Poisson’s equation from the center outward using the DF to determine $\rho(\Phi)$.

---

## What's Next

* Test profile routines for correctness (`setup_grid`, `menc`, etc.)
* Begin writing a demonstration Jupyter notebook
* Implement `RuntimeState` and build the main simulator loop

---

## Installation

This package is designed for local development using `pyenv` and a virtual environment.

```bash
pip install -e .
```

You’ll also want:

```bash
pip install numpy scipy
```

---

## Contributors

* **Yarone Tokayer** — project lead and primary developer

---

## License

MIT License
