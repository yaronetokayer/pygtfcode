import numpy as np
import os

def make_dir(state):
    """
    Create the model directory if it doesn't exist.

    Parameters
    ----------
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

def initialize_log(state):
    """
    Initialize the log file with a header row.

    Parameters
    ----------
    filepath : str
        Path to the log file.
    """

    filepath = state.config.io.logpath

    header = (
        f"{'step':>6}  {'time':>12}  {'dt':>12}  {'rho_c':>12}  "
        f"{'v_max':>12}  {'Kn_min':>12}\n"
    )

    with open(filepath, "w") as f:
        f.write(header)

    if state.config.io.chatter:
        print("Log file initialized:")
        print(header.strip())

def write_log_entry(state):
    """ 
    Append a line to the simulation log file.
    Overwrites any lineswith step_count greater than or equal to the current step_count.

    Parameters
    ----------
    state : State
        The current simulation state.
    """

    filepath = state.config.io.logpath

    new_line = (
        f"{state.step_count:6d}  "
        f"{state.t:12.6e}  "
        f"{state.dt:12.6e}  "
        f"{state.rho[0]:12.6e}  "
        f"{state.maxvel:12.6e}  "
        f"{state.minkn:12.6e}\n"
    )

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
            header = lines[0]
            data_lines = [line for line in lines[1:] if int(line.split()[0]) < state.step_count]

        with open(filepath, "w") as f:
            f.write(header)
            f.writelines(data_lines)
            f.write(new_line)
    else:
        initialize_log(state)
        with open(filepath, "a") as f:
            f.write(new_line)

    if state.config.io.chatter:
        print(new_line.strip())

def write_profile_snapshot(state):
    """ 
    Write full radial profiles to disk.

    Parameters
    ----------
    state : State
        The current simulation state.
    """
    filename = os.path.join(state.config.io.base_dir, state.config.io.model_dir, f"profile_{state.snapshot_index}.dat")

    with open(filename, "w") as f:
        header = (
            f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
            f"{'rho':>12}  {'v2':>12}  {'trelax':>12}  {'kn':>12}\n"
        )
        f.write(header)
        for i in range(len(state.r) - 1):
            f.write(
                f"{i:6d}  "
                f"{np.log(state.r[i+1]):12.6e}  "
                f"{np.log(state.rmid[i]):12.6e}  "
                f"{state.m[i+1]:12.6e}  "
                f"{state.rho[i]:12.6e}  "
                f"{state.v2[i]:12.6e}  "
                f"{state.trelax[i]:12.6e}  "
                f"{state.kn[i]:12.6e}\n"
            )

    