import numpy as np
import os
from pygtfcode.io.read import extract_time_evolution_data
from pygtfcode.util.calc import calc_smfp_r_rho_m_v2, calc_core_r_rho_m_v2, calc_rm2_rho_m_v2, calc_mintheta_r_rho_m_v2, calc_balberg_zeta, low_kn_boost, calc_dlnmc_dlnvc, calc_dlnrhoc_dlnvc
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
    # minTheta    = np.min(state.Theta)

    eps_du_eff = prec.eps_du * low_kn_boost(state.minkn, kn_threshold, du_boost, kn_width)

    # header = f"{'step':>10}  {'time':>12}  {'<dt>':>12}  {'rho0':>12}  {'v_max':>12}  {'Kn_min':>12}  {'eps_du_eff':>10}  {'Theta_min':>9}  {'<du lim>':>8}  {'<dr lim>':>8}  {'<n_iter_du>':>11}  {'<n_iter_dr>':>11}\n"
    header = f"{'step':>10}  {'time':>12}  {'<dt>':>12}  {'n':>5}  {'RDF':>5}  {'rho0':>12}  {'v_max':>12}  {'Kn_min':>12}  {'eps_du_eff':>10}  {'<du lim>':>8}  {'<dr lim>':>8}  {'<n_iter_du>':>11}  {'<n_iter_dr>':>11}\n"

    if step == start_step: # Restart
        new_line = f"{step:10d}  {state.t:12.6e}           N/A  {state.n:5d}  {state.revir_delay_fac:5d}  {state.rho[0]:12.6e}  {maxvel:12.6e}  {state.minkn:12.6e}  {eps_du_eff:10.4e}       N/A       N/A          N/A          N/A\n"

    else:
        nlog = io.nlog
        if step - start_step < nlog:                # First log since restart
            nlog = step - start_step
        elif step % nlog == 0:                      # Intermediate (regular) log
            pass
        elif ( step - start_step ) % nlog != 0:     # Final state
            nlog = ( step - start_step ) % nlog

        # new_line = f"{step:10d}  {state.t:12.6e}  {state.dt_cum / nlog:12.6e}  {state.rho[0]:12.6e}  {maxvel:12.6e}  {state.minkn:12.6e}  {eps_du_eff:10.4e}  {minTheta:9.3e}  {state.du_max_cum / eps_du_eff / nlog:8.2e}  {state.dr_max_cum / prec.eps_dr / nlog:8.2e}  {state.n_iter_du / nlog:11.5e}  {state.n_iter_dr / nlog:11.5e}\n"
        new_line = f"{step:10d}  {state.t:12.6e}  {state.dt_cum / nlog:12.6e}  {state.n:5d}  {state.revir_delay_fac:5d}  {state.rho[0]:12.6e}  {maxvel:12.6e}  {state.minkn:12.6e}  {eps_du_eff:10.4e}  {state.du_max_cum / eps_du_eff / nlog:8.2e}  {state.dr_max_cum / prec.eps_dr / state.n_revir_calls:8.2e}  {state.n_iter_du / nlog:11.5e}  {state.n_iter_dr / state.n_revir_calls:11.5e}\n"

    _update_file(filepath, header, new_line, step)

    state.n_iter_du = 0
    state.n_iter_dr = 0
    state.n_revir_calls = 0
    state.dt_cum = 0.0
    state.du_max_cum = 0.0
    state.dr_max_cum = 0.0

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
    ic_file : string, optional
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

    with open(filename, "w") as f:
        # header = (
        #     f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
        #     f"{'rho':>12}  {'v2':>12}  {'kn':>12}  {'Theta':>12}\n"
        # )
        # header = (
        #     f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
        #     f"{'rho':>12}  {'v2':>12}  {'kn':>12}\n"
        # )
        header = (
            f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
            f"{'rho':>12}  {'v2':>12}  {'kn':>12}  {'drfrac':>12}  {'lum':>12}  {'dttcool':>12}  {'tsctcool':>12}  {'tdyntcool':>12}  {'dttsc':>12}\n"
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
                # f"{state.Theta[i]:12.6e}\n"
                f"{state.drfrac[i]:12.6e}  "
                f"{state.lum[i+1]:12.6e}  "
                f"{_safe_div(dt, state.t_cool[i]):12.6e}  "
                f"{_safe_div(state.t_sc[i], state.t_cool[i]):12.6e}  "
                f"{_safe_div(state.t_dyn[i], state.t_cool[i]):12.6e}  "
                f"{_safe_div(dt, state.t_sc[i]):12.6e}\n"
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
    Append time evolution data to time_evolution.dat

    Arguments
    ---------
    state : State
        The current simulation state.
    last : bool
        If True, then also compute zeta profile
    """
    filepath = os.path.join(
        state.config.io.base_dir,
        state.config.io.model_dir,
        "time_evolution.txt"
    )
    step    = state.step_count
    t       = state.t
    t_Gyr   = t * state.char.t0 * const.sec_to_Gyr
    
    r = state.r; rmid = state.rmid; rho = state.rho; v2 = state.v2; m = state.m
    # Theta = state.Theta

    r_c, rho_c, m_c, v2_c                   = calc_core_r_rho_m_v2(r, rmid, rho, v2, m)
    r_m2, rho_m2, m_m2, v2_m2               = calc_rm2_rho_m_v2(r, rmid, rho, v2, m)
    r_smfp, rho_smfp, m_smfp, v2_smfp       = calc_smfp_r_rho_m_v2(r, rmid, state.kn, rho,  v2, m)
    # r_minTh, rho_minTh, m_minTh, v2_minTh   = calc_mintheta_r_rho_m_v2(r, rmid, rho, v2, m, Theta)

    maxvel      = np.max(np.sqrt(state.v2))
    # minTheta    = np.min(Theta)

    columns = [
        ("step", step),
        ("dt", state.dt),
        ("time", t),
        ("time_Gyr", t_Gyr),
        ("rho0", state.rho[0]),
        ("v_max", maxvel),
        ("Kn_min", state.minkn),
        # ("minTheta", minTheta),
        ("r_c", r_c),
        ("rho_c", rho_c),
        ("m_c", m_c),
        ("v2_c", v2_c),
        ("r_m2", r_m2),
        ("rho_m2", rho_m2),
        ("m_m2", m_m2),
        ("v2_m2", v2_m2),
        ("r_smfp", r_smfp),
        ("rho_smfp", rho_smfp),
        ("m_smfp", m_smfp),
        ("v2_smfp", v2_smfp),
        # ("r_minTh", r_minTh),
        # ("rho_minTh", rho_minTh),
        # ("m_minTh", m_minTh),
        # ("v2_minTh", v2_minTh),
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
        dlnrhoc_dlnvc = calc_dlnmc_dlnvc(tevol_data['rho_c'], tevol_data['v2_c'], 31)
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
        else:
            lines = [header]
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