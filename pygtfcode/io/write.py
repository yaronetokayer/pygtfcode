import numpy as np
import os
from pygtfcode.parameters.constants import Constants as const
from pygtfcode.util.calc import calc_smfp_r_rho_m_v2, calc_core_r_rho_m_v2, calc_rm2_rho_m_v2

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
        f.write(f"Model {io.model_no:03d} Metadata\n")
        f.write("=" * 40 + "\n\n")
        lines = dump_attrs(state.config)
        f.write("\n".join(lines) + "\n")

    if state.config.io.chatter:
        print(f"Model information written to model_metadata.txt")

def write_char_params(state, char):
    """
    Write characteristic parameters to a file.

    Arguments
    ---------
    state : State
        The current simulation state.
    char : object
        The characteristic parameters object.
    """
    io = state.config.io
    filename = os.path.join(io.base_dir, io.model_dir, f"char_params.txt")

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
        print(f"\tCharacteristic parameters written to char_params.txt")

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
    io = state.config.io
    prec = state.config.prec
    filepath = os.path.join(io.base_dir, io.model_dir, f"logfile.txt")
    chatter = io.chatter
    step = state.step_count

    header = f"{'step':>10}  {'time':>12}  {'<dt>':>12}  {'rho0':>12}  {'v_max':>12}  {'Kn_min':>12}  {'<dt lim>':>8}  {'<du lim>':>8}  {'<dr lim>':>8}  {'<n_iter_du>':>11}  {'<n_iter_dr>':>11}\n"

    if step == start_step: # Restart
        new_line = f"{step:10d}  {state.t:12.6e}           N/A  {state.rho[0]:12.6e}  {state.maxvel:12.6e}  {state.minkn:12.6e}       N/A       N/A       N/A          N/A          N/A\n"

    else:
        nlog = io.nlog
        if step - start_step < nlog:                # First log since restart
            nlog = step - start_step
        elif step % nlog == 0:                      # Intermediate (regular) log
            pass
        elif ( step - start_step ) % nlog != 0:     # Final state
            nlog = ( step - start_step ) % nlog

        new_line = f"{step:10d}  {state.t:12.6e}  {state.dt_cum / nlog:12.6e}  {state.rho[0]:12.6e}  {state.maxvel:12.6e}  {state.minkn:12.6e}  {state.dt_over_trelax_cum / prec.eps_dt / nlog:8.2e}  {state.du_max_cum / prec.eps_du / nlog:8.2e}  {state.dr_max_cum / prec.eps_dr / nlog:8.2e}  {state.n_iter_du / nlog:11.5e}  {state.n_iter_dr / nlog:11.5e}\n"

    _update_file(filepath, header, new_line, step)

    state.n_iter_du = 0
    state.n_iter_dr = 0
    state.dt_cum = 0.0
    state.du_max_cum = 0.0
    state.dr_max_cum = 0.0
    state.dt_over_trelax_cum = 0.0

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
        header = (
            f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
            f"{'rho':>12}  {'v2':>12}  {'trelax':>12}  {'kn':>12}\n"
        )
        f.write(header)
        for i in range(len(state.r) - 1):
            f.write(
                f"{i:6d}  "
                f"{np.log10(state.r[i+1]):12.6e}  "
                f"{np.log10(state.rmid[i]):12.6e}  "
                f"{state.m[i+1]:12.6e}  "
                f"{state.rho[i]:12.6e}  "
                f"{state.v2[i]:12.6e}  "
                f"{state.trelax[i]:12.6e}  "
                f"{state.kn[i]:12.6e}\n"
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
    
    header = (f"{'index':>6}  {'time':>12}  {'time_Gyr':>12}  {'step':>10}\n")

    new_line = (
        f"{index:6d}  "
        f"{state.t:12.6e}  "
        f"{state.t * state.char.t0 * const.sec_to_Gyr:12.6e}  "
        f"{state.step_count:10d}\n"
    )

    _update_file(filepath, header, new_line, index)

def write_time_evolution(state):
    """
    Append time evolution data to time_evolution.dat

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    filepath = os.path.join(
        state.config.io.base_dir,
        state.config.io.model_dir,
        "time_evolution.txt"
    )
    step = state.step_count
    t = state.t
    
    r = state.r; rmid = state.rmid; rho = state.rho; v2 = state.v2; m = state.m

    r_c, rho_c, m_c, v2_c = calc_core_r_rho_m_v2(r, rmid, rho, v2, m)
    r_m2, rho_m2, m_m2, v2_m2 = calc_core_r_rho_m_v2(r, rmid, rho, v2, m)
    r_smfp, rho_smfp, m_smfp, v2_smfp = calc_smfp_r_rho_m_v2(r, rho,  v2, m, state.char.sigma_m_char)

    columns = [
        ("step", step),
        ("time", t),
        ("rho0", state.rho[0]),
        ("v_max", state.maxvel),
        ("Kn_min", state.minkn),
        ("mintrel", state.mintrelax),
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