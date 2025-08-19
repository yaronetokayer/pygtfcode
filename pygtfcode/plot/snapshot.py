import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
import shutil

def extract_snapshot_indices(model_dir):
    """
    Extract snapshot indices and times from snapshot_conversion.txt.

    Parameters
    ----------
    model_dir : str
        Path to the model directory.

    Returns
    -------
    dict
        Dictionary with keys 'step', 't_t0', and 't_Gyr' containing numpy arrays.
    """
    path = os.path.join(model_dir, "snapshot_conversion.txt")

    data = np.loadtxt(path, usecols=(0, 1, 2), skiprows=1)
    if data.ndim == 1:
        # Only one row of data
        snapshot_index = np.array([int(data[0])])
        t_t0 = np.array([data[1]])
        t_Gyr = np.array([data[2]])
    else:
        snapshot_index = data[:, 0].astype(int)
        t_t0 = data[:, 1]
        t_Gyr = data[:, 2]
    return {
        'snapshot_index': snapshot_index,
        't_t0': t_t0,
        't_Gyr': t_Gyr
    }

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

    data = extract_snapshot_indices(model_dir)

    # Lookup time
    idx = np.where(data['snapshot_index'] == index)[0][0]
    if not phys:
        t = data['t_t0'][idx]
    else:
        t = data['t_Gyr'][idx]

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
        'log_r', 'log_rmid', 'm', 'rho', 'v2', 'p', 'trel', 'kn', 'time'
    """
    data = np.loadtxt(filename, usecols=range(1, 9), skiprows=1)

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
        'p': data[:, 5],
        'trelax': data[:, 6],
        'kn': data[:, 7],
        'time': t
    }

def plot_profile(ax, profile, data_list, legend=True, grid=False, for_movie=False):
    """
    Plot specified profile on the passed axis object

    Arguments
    ---------
    ax : Axis
        Axis object on which to plot
    profile : str
        Profile to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'kn'
    data_list : dict
        Dictionary returned by extract_snapshot_data()
    legend : bool, optional
        If True, include a legend in the plot
    grid : bool, optional
        If True, shows grid on axes
    for_movie : bool, should not be set by user
        If True, then plot_snapshots() is being called by make_movie()
        This controls the colormap of the plots
    """
    # Set colormap
    if for_movie:
        from matplotlib.colors import ListedColormap
        if len(data_list) == 1:
            cmap = ListedColormap(['black'])
        else:
            cmap = ListedColormap(['gray', 'black'])
    else:
        cmap = plt.get_cmap('tab20')

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

def plot_snapshots(model, snapshots=[0], profiles='rho', base_dir=None, filepath=None, show=False, grid=False, for_movie=False):
    """
    Plot up to three profiles at specified points in time for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    snapshots : int or list of int
        Snapshot indices to plot
    profiles : str or list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'kn'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    filepath : str, optional
        If provided, save the plot to this file.
    show : bool, optional
        If True, show the plot even if saving.  Default is False.
    grid : bool, optional
        If True, shows grid on axes
    for_movie : bool, should not be set by user
        If True, then being called by make_movie()
        This controls the colormap of the plots
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
        plot_profile(axs, profiles, data_list, legend=True, grid=grid, for_movie=for_movie)
    else:
        for ind, ax in enumerate(axs):
            legend = False if ind < len(axs) - 1 else True
            plot_profile(ax, profiles[ind], data_list, legend=legend, grid=grid, for_movie=for_movie)

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches=None)
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()

def make_movie(model, filepath=None, base_dir=None, profiles='rho', grid=False, fps=20):
    """
    Animate up to three profiles for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXX/movie_{profiles}.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    profiles : str or list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'kn'
    grid : bool, optional
        If True, shows grid on axes
    fps : int, optional
        Frames per second for the output movie. Default is 20

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """

    n = 1 if type(profiles) != list else len(profiles) # number of panels

    # Get the model directory
    if hasattr(model, 'config'):        # Passed state object
        model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
    elif hasattr(model, 'io'):          # Passed config object
        model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
    elif isinstance(model, int):        # Passed model number
        if base_dir is None:
            raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
        model_dir = f"Model{model:03d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load snapshot indices
    snapshot_indices_data = extract_snapshot_indices(model_dir)
    indices = snapshot_indices_data['snapshot_index']

    # Create a temporary directory for storing images
    temp_dir = os.path.join(model_dir, "temp_images")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)         # Delete the directory and all its contents
    os.makedirs(temp_dir)

    image_paths = []                    # List to store paths of generated images

    print(f"Generating {len(indices)} frames...")
    for ind in tqdm(indices, desc="Frames", unit="frame"):
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
        if not os.path.isfile(snapshot_path):
            continue                    # Skip if the snapshot file does not exist

        # Define the output image path for the current frame
        image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

        # Plot the profile, including the initial profile for comparison
        if ind == 0:
            plot_snapshots(model, profiles=profiles, base_dir=base_dir, filepath=image_path, grid=grid, for_movie=True)
        else:
            plot_snapshots(model, snapshots=[0,ind], profiles=profiles, base_dir=base_dir, filepath=image_path, grid=grid, for_movie=True)

        image_paths.append(image_path)  # Add the image path to the list

    print("Compiling into a movie using ffmpeg...")
    # Define the output movie path
    if isinstance(profiles, (list, tuple)):
        profiles_str = "_".join(map(str, profiles))
    else:
        profiles_str = str(profiles)

    output_movie_path = (
        filepath if filepath is not None 
        else os.path.join(model_dir, f"movie_{profiles_str}.mp4")
    )

    # Construct the ffmpeg command to create the movie
    movie_command = [
        "ffmpeg",
        "-y",                                           # Overwrite output file if it exists
        "-framerate", str(fps),                         # Set frames per second
        "-i", os.path.join(temp_dir, "frame_%04d.png"), # Input image sequence
        "-c:v", "libx264",                              # Use H.264 codec
        "-pix_fmt", "yuv420p",                          # Set pixel format for compatibility
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",     # Ensure even dimensions
        output_movie_path
    ]

    # Run the ffmpeg command
    subprocess.run(movie_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

    print("Deleting frames...")
    # Clean up temporary images
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print the location of the saved movie
    print(f"Movie saved to {output_movie_path}")