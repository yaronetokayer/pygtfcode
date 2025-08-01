import os
import numpy as np
import matplotlib.pyplot as plt

def get_time_conversion(filepath, index, phys=False):
    """
    Get conversion from index to time from snapshot_conversion.txt.

    Parameters
    ----------
    filepath : str
        Path to the profile_x.dat file.
    index : int
        Snapshot index at which to get time
    phys : bool
        If True, get value in Gyr, otherwise in simulation units

    Returns
    -------
    float
        Time value.
    """
    # Find corresponding timestep.log in the same ModelXXX directory
    model_dir = os.path.dirname(filepath)
    conv_path = os.path.join(model_dir, "snapshot_conversion.txt")

    data = np.loadtxt(conv_path, usecols=(0, 1, 2), skiprows=1)
    step = np.insert(data[:, 0].astype(int), 0, 0)
    t_t0 = np.insert(data[:, 1], 0, 0.0)
    t_Gyr = np.insert(data[:, 2], 0, 0.0)

    # Lookup time
    idx = np.where(step == index)[0][0]
    if not phys:
        t = t_t0[idx]
    else:
        t = t_Gyr[idx]

    return t

def extract_snapshot_data(filename):
    """
    Extract data from a snapshot timestep file.

    Parameters
    ----------
    filename : str
        Path to the timestep_*.dat file.

    Returns
    -------
    dict
        Dictionary of numpy arrays with keys:
        'log_r', 'log_rmid', 'm', 'rho', 'v2', 'trel', 'kn', 'time'
    """
    data = np.loadtxt(filename, usecols=range(1, 8), skiprows=1)

    # Extract timestep number from filename and get time
    basename = os.path.basename(filename)
    step = int(basename.replace("profile_", "").replace(".dat", ""))
    t = get_time_conversion(filename, step)

    return {
        'log_r': data[:, 0],
        'log_rmid': data[:, 1],
        'm': data[:, 2],
        'rho': data[:, 3],
        'v2': data[:, 4],
        'trelax': data[:, 5],
        'kn': data[:, 6],
        'time': t
    }

def plot_profile(ax, profile, data_list, legend=True, grid=False):
    """
    Plot specified profile on the passed axis object

    Arguments
    ---------
    ax : Axis
        Axis object on which to plot
    profile : str
        Profile to plot.  Options are 'rho', 'm', 'v2', 'trelax', 'kn'
    data_list : dict
        Dictionary returned by extract_snapshot_data()
    legend : bool, optional
        If True, include a legend in the plot
    grid : bool, optional
        If True, shows grid on axes
    """
    cmap = plt.get_cmap('tab10')
    ylim_lower = None
    for ind, data in enumerate(data_list):
        xkey = 'log_r' if profile == 'm' else 'log_rmid'
        x = data[xkey]
        y = data[profile]
        if ind == 0:
            ylim_lower = np.min(y[y > 0]) * 0.5
            xlim_lower = 10**np.min(x) * 0.8
            ylim_upper = np.max(y) * 100
            xlim_upper = 10**np.max(x) * 1.2
        else:
            ylim_lower = np.min([ylim_lower, np.min(y[y > 0]) * 0.5])
            xlim_lower = np.min([xlim_lower, 10**np.min(x) * 0.8])
            ylim_upper = np.max([ylim_upper, np.max(y) * 100])
            xlim_upper = np.max([xlim_upper, 10**np.max(x) * 1.2])

        ax.plot(10**x, y, lw=2, color=cmap(ind % 10), label=f"t={data['time']:.2e}")

        if profile == 'kn' and ind == 0:
            ax.axhline(1.0, color='black', ls=':')
            ax.text(0.95, 1.1, 'LMFP', ha='right', va='bottom', fontsize=12, transform=ax.get_yaxis_transform())
            ax.text(0.95, 0.9, 'SMFP', ha='right', va='top', fontsize=12, transform=ax.get_yaxis_transform())

        if ind == len(data_list) - 1:
            if profile == 'kn':
                ylim_lower = np.min([ylim_lower, 0.1])
            ax.set_xlim([xlim_lower, xlim_upper])
            ax.set_ylim([ylim_lower, ylim_upper])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Radius [$r_\mathrm{s,0}$]', fontsize=14)
    ax.set_ylabel(profile, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, which="both", ls="--")

def plot_snapshots(model, snapshots=[0], profiles='rho', base_dir=None, filepath=None, show=False, grid=False):
    """
    Plot up to three profiles at specified points in time for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    snapshots : int or list of int
        Snapshot indices to plot
    profiles : str or list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'trelax', 'kn'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    filepath : str, optional
        If provided, save the plot to this file.
    show : bool, optional
        If True, show the plot even if saving.  Default is False.
    grid : bool, optional
        If True, shows grid on axes
    """

    if type(snapshots) != list:
        snapshots = [snapshots]

    n = 1 if type(profiles) != list else len(profiles) # number of panels

    def _resolve_dir(model, ind):
        if hasattr(model, 'config'): # Passed state object
            return os.path.join(model.config.io.base_dir, model.config.io.model_dir, f"profile_{ind}.dat")
        elif hasattr(model, 'io'): # Passed config object
            return os.path.join(model.io.base_dir, model.io.model_dir, f"profile_{ind}.dat")
        elif isinstance(model, int): # Passed model number
            if base_dir is None:
                raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
            model_dir = f"Model{model:03d}"
            return os.path.join(base_dir, model_dir, f"profile_{ind}.dat")
        else:
            raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")

    data_list = [extract_snapshot_data(_resolve_dir(model,ind)) for ind in snapshots]

    fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

    if n == 1:
        plot_profile(axs, profiles, data_list, legend=True, grid=grid)
    else:
        for ind, ax in enumerate(axs):
            legend = False if ind < len(axs) - 1 else True
            plot_profile(ax, profiles[ind], data_list, legend=legend, grid=grid)

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches=None)
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()