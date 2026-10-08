import numpy as np
import os
from pygtfcode.io.read import extract_time_evolution_data
from pygtfcode.util.calc_slopes import calc_balberg_zeta, calc_dlnmc_dlnvc, calc_dlnrhoc_dlnvc, calc_s_dsdr, calc_dlogrho_dlogp
from pygtfcode.util.calc_core import calc_core_r_rho_m_v2, calc_rmn_rho_m_v2
from pygtfcode.util.calc_runtime import low_kn_boost, calc_kappa_cell, calc_kappa_edge
from pygtfcode.util.calc_kp import factors
from pygtfcode.parameters.constants import Constants as const

def _safe_div(num, den):
    return 0.0 if den == 0 else num / den

def make_dir(state):
    """
    Create the model directory if it doesn't exist.

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    model_dir = state.config.io.model_dir
    base_dir = state.config.io.base_dir

    full_path = os.path.join(base_dir, model_dir)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        if state.config.io.chatter:
            print(f"Created directory: {full_path}")
    else:
        if not state.config.io.overwrite:
            raise FileExistsError(f"Model directory exists and overwrite=False: {full_path}")
        if state.config.io.chatter:
            print(f"Directory already exists: {full_path}")

def write_metadata(state):
    """
    Write model metadata to disk for reference.

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    io = state.config.io
    filename = os.path.join(io.base_dir, io.model_dir, f"model_metadata.txt")

    def dump_attrs(obj, indent=0):
        lines = []
        for key in sorted(vars(obj)):
            val = getattr(obj, key)
            if hasattr(val, '__dict__'):  # it's a nested config object
                lines.append(" " * indent + f"{key}:")
                lines.extend(dump_attrs(val, indent + 4))
            else:
                lines.append(" " * indent + f"{key}: {val}")
        return lines

    with open(filename, "w") as f:
        f.write(f"Model {io.model_no:05d} Metadata\n")
        f.write("=" * 40 + "\n\n")
        lines = dump_attrs(state.config)
        f.write("\n".join(lines) + "\n")

    if state.config.io.chatter:
        print(f"Model information written to model_metadata.txt")

def write_char_params(state):
    """
    Write characteristic parameters to a file.

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    char        = state.char
    io          = state.config.io
    filename    = os.path.join(io.base_dir, io.model_dir, f"char_params.txt")

    # Get items from char's attributes
    items = list(char.__dict__.items())

    # Extract names and values, converting None to NaN and ensuring floats
    names = [key for key, value in items]
    values = [
        np.nan if value is None else float(value)
        for key, value in items
    ]

    # Set column width for formatting
    col_width = 18
    # Create header string with right-aligned names
    header = "".join(f"{name:>{col_width}}" for name in names)
    # Format string for scientific notation
    fmt = f"%{col_width}.8e"

    # Save the values to file with header
    np.savetxt(
        filename,
        [values],
        header=header,
        fmt=fmt,
        delimiter='',
        comments=""
    )

    # Print message if chatter is enabled
    if state.config.io.chatter:
        print(f"Characteristic parameters written to char_params.txt")

def write_log_entry(state, start_step):
    """ 
    Append a line to the simulation log file.
    Overwrites any lines with step_count >= current step_count.

    Arguments
    ---------
    state : State
        The current simulation state.
    start_step : int
        The starting value of the current simulation run
    """
    io = state.config.io; prec = state.config.prec
    filepath = os.path.join(io.base_dir, io.model_dir, f"logfile.txt"); chatter = io.chatter
    kn_threshold = prec.kn_threshold; du_boost = prec.du_boost; kn_width = prec.kn_width
    step = state.step_count

    maxvel      = np.max(np.sqrt(state.v2))

    eps_du_eff = prec.eps_du * low_kn_boost(state.kn_cond_c, kn_threshold, du_boost, kn_width)

    # Average each accepted step's limiter fraction, not the ratio of averages.
    # dr is the final HE correction, not total shell displacement. Iteration
    # counters count retries/additional solves; split/merge count operations.
    count = state.log_steps
    columns = [
        ('step', step), ('time', state.t),
        ('<dt>', state.dt_cum / count if count else None),
        ('n', state.n), ('rho0', state.rho[0]), ('v_max', maxvel),
        ('kn_c', state.kn_c), ('kn_cond_c', state.kn_cond_c),
        ('eps_du_eff', eps_du_eff),
        ('<du lim>', state.du_limit_cum / count if count else None),
        ('<dr lim>', state.dr_max_cum / prec.eps_dr / count if count else None),
        ('<n_retry_du>', state.n_iter_du / count if count else None),
        ('<n_iter_dr>', state.n_iter_dr / count if count else None),
        ('n_split', state.n_split), ('n_merge', state.n_merge),
    ]
    header = '  '.join(f'{name:>13}' for name, _ in columns) + '\n'
    new_line = '  '.join(
        f'{"N/A":>13}' if value is None else
        f'{value:13d}' if isinstance(value, (int, np.integer)) else
        f'{value:13.6e}' for _, value in columns
    ) + '\n'
    _update_file(filepath, header, new_line, step)

    state.n_iter_du = state.n_iter_dr = 0
    state.n_split = state.n_merge = state.log_steps = 0
    state.dt_cum = state.du_max_cum = state.dr_max_cum = 0.0
    state.du_limit_cum = 0.0

    if chatter:
        if step == 0:
            print("Log file initialized:")
        if step == start_step:
            print(header[:-1])
        print(new_line[:-1])

def write_profile_snapshot(state, initialize=False, ic_filename=None):
    """ 
    Write full radial profiles to disk.

    Arguments
    ---------
    state : State
        The current simulation state.
    initialize : bool
        If True, this is part of initializing the grid and should not increment the snapshot index.
    ic_filename : str, optional
        If provided, this is part of writing an initial condition file.
    """
    if ic_filename is None:
        io = state.config.io
        filename = os.path.join(io.base_dir, io.model_dir, f"profile_{state.snapshot_index}.dat")
    else:
        filename = ic_filename

    # If not initializing, remove any higher-index snapshot files
    if not initialize:
        snapshot_dir = os.path.join(io.base_dir, io.model_dir)

        for fname in os.listdir(snapshot_dir):
            if not fname.startswith("profile_") or not fname.endswith(".dat"):
                continue

            try:
                idx = int(fname[len("profile_"):-len(".dat")])
            except ValueError:
                continue  # ignore unexpected files

            if idx > state.snapshot_index:
                os.remove(os.path.join(snapshot_dir, fname))

    # kn/mfp remain amplitude-reference columns. The appended *_cond columns
    # describe conductivity; mfp_cond is an effective length in units of r_s,
    # not a literal collision mean free path.
    # On the fly computations
    s, dsdr = calc_s_dsdr(state.v2, state.rho, state.rmid)
    dlnrhodlnp = calc_dlogrho_dlogp(state.v2, state.rho)
    drltemp = np.empty_like(state.rho)
    drltemp[0] = np.nan
    drltemp[1:] = (state.r[2:] - state.r[1:-1]) / state.ltemp[1:]
    mfpltemp = state.mfp / state.ltemp
    mfp_cond_ltemp = state.mfp_cond / state.ltemp
    x = np.sqrt(state.v2) / state.char.w_char
    moments = np.array([factors(T, state.char.w_char, state.config.sim.smfp_order)
                        for T in state.v2])
    # Transport factors at cell temperatures; not collision cross sections.
    k_l_factor, k_s_factor = moments[:, 0], moments[:, 1]
    sim = state.config.sim
    a = float(sim.a); b = float(sim.b); c = float(sim.c); sigma_m_0 = float(state.char.sigma_m_0_char); alph = float(sim.alph);
    k_lc, k_sc, k_totc = calc_kappa_cell(state.v2, state.rho, state.rmid, a, b, c, sigma_m_0, alph, float(state.char.w_char), sim.smfp_order,)
    krat_c = k_sc / k_lc
    k_le, k_se, k_tote = calc_kappa_edge(state.v2, state.rho, state.r, a, b, c, sigma_m_0, alph, float(state.char.w_char), sim.smfp_order,)
    krat_e = k_se / k_le

    with open(filename, "w") as f:
        header = (
            f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
            f"{'rho':>12}  {'v2':>12}  {'kn':>12}  {'ltemp':>12}  {'mfp':>12}  {'drfrac':>12}  "
            f"{'drltemp':>12}  {'mfpltemp':>12}  "
            f"{'k_sc':>12}  {'k_lc':>12}  {'k_totc':>12}  "
            f"{'k_se':>12}  {'k_le':>12}  {'k_tote':>12}  "
            f"{'krat_c':>12}  {'krat_e':>12}  "
            f"{'dttcool':>12}  {'tdyntcool':>12}  {'s':>12}  {'dsdr':>12}  {'dlnrhodlnp':>12}  "
            f"{'kn_cond':>12}  {'mfp_cond':>12}  "
            f"{'x':>12}  {'K_L':>12}  {'K_S':>12}  {'mfp_cond_ltemp':>16}\n"
        )
        dt = state.dt ### for the timescales

        f.write(header)
        for i in range(len(state.r) - 1):
            f.write(
                f"{i:6d}  "
                f"{np.log10(state.r[i+1]):12.6e}  "
                f"{np.log10(state.rmid[i]):12.6e}  "
                f"{state.m[i+1]:12.6e}  "
                f"{state.rho[i]:12.6e}  "
                f"{state.v2[i]:12.6e}  "
                f"{state.kn[i]:12.6e}  "
                f"{state.ltemp[i]:12.6e}  "
                f"{state.mfp[i]:12.6e}  "
                f"{state.drfrac[i]:12.6e}  "
                f"{drltemp[i]:12.6e}  "
                f"{mfpltemp[i]:12.6e}  "
                f"{k_sc[i]:12.6e}  "
                f"{k_lc[i]:12.6e}  "
                f"{k_totc[i]:12.6e}  "
                f"{k_se[i]:12.6e}  "
                f"{k_le[i]:12.6e}  "
                f"{k_tote[i]:12.6e}  "
                f"{krat_c[i]:12.6e}  "
                f"{krat_e[i]:12.6e}  "
                f"{_safe_div(dt, state.t_cool[i]):12.6e}  "
                f"{_safe_div(state.t_dyn[i], state.t_cool[i]):12.6e}  "
                f"{s[i]:12.6e}  "
                f"{dsdr[i]:12.6e}  "
                f"{dlnrhodlnp[i]:12.6e}  "
                f"{state.kn_cond[i]:12.6e}  "
                f"{state.mfp_cond[i]:12.6e}  "
                f"{x[i]:12.6e}  {k_l_factor[i]:12.6e}  {k_s_factor[i]:12.6e}  "
                f"{mfp_cond_ltemp[i]:16.6e}\n"
            )
    
    if ic_filename is None:
        append_snapshot_conversion(state)

        if io.chatter:
            if (ic_filename is None) and (state.step_count == 0):
                print("Initial profiles written to disk.")

        if not initialize: # Do not increment if this is part of intializing the grid
            state.snapshot_index += 1

    else:
        print(f"Initial condition file written to {filename}.")

def append_snapshot_conversion(state):
    """
    Append conversion between snapshot_index and time

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    filepath = os.path.join(
        state.config.io.base_dir, 
        state.config.io.model_dir, 
        f"snapshot_conversion.txt"
        )
    index = state.snapshot_index
    
    header = (f"{'index':>6}  {'time':>12}  {'step':>10}\n")

    new_line = (
        f"{index:6d}  "
        f"{state.t:12.6e}  "
        f"{state.step_count:10d}\n"
    )

    _update_file(filepath, header, new_line, index)

def write_time_evolution(state, last=False):
    """
    Append time evolution data to time_evolution.txt.

    tsc_c is r_c / sqrt(v2_c), converted to the same t_s units as time.
    K_L_c/K_S_c and K_L_m2/K_S_m2 are evaluated at the mass-averaged
    dispersions v2_c and v2_m2, not averages of local transport factors.

    Arguments
    ---------
    state : State
        The current simulation state.
    last : bool
        If True, append derived core-evolution slopes and zeta_balb.
    """
    filepath = os.path.join(
        state.config.io.base_dir,
        state.config.io.model_dir,
        "time_evolution.txt"
    )
    step    = state.step_count
    t       = state.t
    t_Gyr   = t * state.char.t_s * const.sec_to_Gyr
    
    r = state.r; rmid = state.rmid; rho = state.rho; v2 = state.v2; m = state.m

    r_c, rho_c, m_c, v2_c, tsc_c            = calc_core_r_rho_m_v2(r, rmid, rho, v2, m)
    r_m2, rho_m2, m_m2, v2_m2               = calc_rmn_rho_m_v2(r, rmid, rho, v2, m, 2.0)
    drfrac_max                              = np.max(np.diff(r[1:]) / np.sqrt(r[1:-1] * r[2:]))

    maxvel      = np.max(np.sqrt(state.v2))
    # Convert dispersion-crossing time from r_s/v_s to amplitude time t_s.
    tsc_c *= state.config.sim.a * state.char.sigma_m_0_char
    what = state.char.w_char
    order = state.config.sim.smfp_order
    x_c = np.sqrt(v2_c) / what
    x_m2 = np.sqrt(v2_m2) / what
    kl_c, ks_c, _, _ = factors(v2_c, what, order)
    kl_m2, ks_m2, _, _ = factors(v2_m2, what, order)

    columns = [
        ("step", step),
        ("dt", state.dt),
        ("n", state.n),
        ("time", t),
        ("time_Gyr", t_Gyr),
        ("rho0", state.rho[0]),
        ("v_max", maxvel),
        ("kn_c", state.kn_c),  # Amplitude-reference mean, as before.
        ("kn_cond_c", state.kn_cond_c),  # Conductivity-effective core mean.
        ("r_c", r_c),
        ("rho_c", rho_c),
        ("m_c", m_c),
        ("v2_c", v2_c),
        ("x_c", x_c), ("K_L_c", kl_c), ("K_S_c", ks_c),
        ("r_m2", r_m2),
        ("rho_m2", rho_m2),
        ("m_m2", m_m2),
        ("v2_m2", v2_m2),
        ("x_m2", x_m2), ("K_L_m2", kl_m2), ("K_S_m2", ks_m2),
        ("drfrac_max", drfrac_max),
        ("tsc_c", tsc_c)
    ]

    # Build header
    header = "  ".join(f"{name:>12}" for name, _ in columns) + "\n"

    # Build row
    formatted_values = []
    for name, value in columns:
        if isinstance(value, int):
            formatted_values.append(f"{value:12d}")
        else:
            formatted_values.append(f"{value:12.6e}")

    new_line = "  ".join(formatted_values) + "\n"

    _update_file(filepath, header, new_line, step)

    if state.config.io.chatter and step == 0:
        print("Time evolution file initialized.")

    if last:
        tevol_data = extract_time_evolution_data(filepath)
        dlnmc_dlnvc = calc_dlnmc_dlnvc(tevol_data['m_c'], tevol_data['v2_c'], 31)
        _append_column_to_time_evolution_file(filepath, "dlnmc_dlnvc", dlnmc_dlnvc)
        dlnrhoc_dlnvc = calc_dlnrhoc_dlnvc(tevol_data['rho_c'], tevol_data['v2_c'], 31)
        _append_column_to_time_evolution_file(filepath, "dlnrhocdlnvc", dlnrhoc_dlnvc)
        zeta_c = calc_balberg_zeta(tevol_data['m_c'], tevol_data['v2_c'], 31)
        _append_column_to_time_evolution_file(filepath, "zeta_balb", zeta_c)

        if state.config.io.chatter:
            print("Time evolution file finalized.")

def _update_file(filepath, header, new_line, index):
    """
    Helper function to update a file.
    If the file doesn't exist, it initializes it.
    If the file does exist, it appends the new_line, erasing all lines with
    a first column >= index.

    Arguments
    ---------
    filepath : str
        Path to the file.
    header : str
        Header row.
    new_line : str
        Row to be appended.
    index : int
        Index to compare to determine where to place new_line
    """

    lines = []

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()

        if lines and lines[0].strip() == header.strip():
            lines = [lines[0]] + [line for line in lines[1:] if int(line.split()[0]) < index]
        elif lines and lines[0].split()[:len(header.split())] == header.split():
            # A completed history has appended slope columns. Drop only those
            # derived columns when continuing, preserving earlier base rows.
            ncols = len(header.split())
            lines = [header] + [
                '  '.join(line.split()[:ncols]) + '\n'
                for line in lines[1:] if int(line.split()[0]) < index
            ]
        elif index == 0 or not lines:
            lines = [header]
        else:
            raise ValueError(f"Cannot append incompatible output schema to {filepath}")
    else:
        lines = [header]

    lines.append(new_line)

    with open(filepath, "w") as f:
        f.writelines(lines)

def _append_column_to_time_evolution_file(filepath, colname, values):
    """
    Append a new column to an existing time evolution file.

    Arguments
    ---------
    filepath : str
        Path to the time evolution file.
    colname : str
        Name of the new column.
    values : ndarray, shape (N,)
        Values to append to each data row.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header = lines[0]
    data_lines = lines[1:]

    if len(data_lines) != values.shape[0]:
        raise ValueError("Number of values does not match number of data rows")

    # Add new column name to header
    new_header = header.rstrip("\n") + f"  {colname:>12}\n"

    # Add one new value to each data row
    new_lines = [new_header]

    for line, value in zip(data_lines, values):
        new_line = line.rstrip("\n") + f"  {value:12.6e}\n"
        new_lines.append(new_line)

    with open(filepath, "w") as f:
        f.writelines(new_lines)
