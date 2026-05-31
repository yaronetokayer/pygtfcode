import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
import shutil
from pygtfcode.io.read import extract_snapshot_data, extract_snapshot_indices, extract_time_evolution_data
from pygtfcode.util.interpolate import interp_powerlaw_edges_to_cells

def get_profile_axis_limits(profile, data_list, xaxis='r'):
    if xaxis == 'r':
        xkey = 'log_r' if profile == 'm' else 'log_rmid'
    elif xaxis == 'm':
        xkey = 'm'

    xlim_lower = np.inf
    xlim_upper = -np.inf
    ylim_lower = np.inf
    ylim_upper = -np.inf

    for data in data_list:
        if xaxis == 'r':
            x = 10**data[xkey]
        elif xaxis == 'm':
            if profile == 'm':
                x = data['m']
            else:
                m_edges = data['m']
                r_edges = 10**data['log_r']
                rmid = 10**data['log_rmid']
                x = interp_powerlaw_edges_to_cells(r_edges, m_edges, rmid)

        y = data[profile]

        positive_y = y[y > 0] # Guard against zero or negative profiles
        if len(positive_y) > 0:
            ylim_lower = min(ylim_lower, np.min(positive_y) * 0.5)

        ylim_upper = max(ylim_upper, np.max(y) * 10)

        xlim_lower = min(xlim_lower, np.min(x) * 0.8)
        xlim_upper = max(xlim_upper, np.max(x) * 1.2)

    if profile in ['kn', 'Theta']:
        ylim_lower = min(ylim_lower, 0.1)

    return (xlim_lower, xlim_upper), (ylim_lower, ylim_upper)

def plot_profile(ax, profile, data_list, xaxis='r', axislims=None, legend=True, legend_loc=None, grid=False, for_movie=False):
    """
    Plot specified profile on the passed axis object

    Arguments
    ---------
    ax : Axis
        Axis object on which to plot
    profile : str
        Profile to plot.  Options are 'rho', 'm', 'v2', 'kn'
    data_list : dict
        Dictionary returned by extract_snapshot_data()
    xaxis : str, optional
        X-axis to plot.  Default is 'r'.  Other option is 'm'.
    axislims : list of tuples or None
        [(xmin, xmax), (ymin, ymax)]
    legend : bool, optional
        If True, include a legend in the plot
    legend_loc : str, optional
        If not None, use this for the legend location
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

    if xaxis == 'r':
        xkey = 'log_r' if profile == 'm' else 'log_rmid'
    elif xaxis == 'm':
        xkey = 'm'

    # Get axis limits
    if axislims is None:
        xlim, ylim = get_profile_axis_limits(profile, data_list, xaxis=xaxis)
    else:
        xlim, ylim = axislims

    # Plot data
    for ind, data in enumerate(data_list):
        rmid = 10**data['log_rmid']
        if xaxis == 'r':
            x = 10**data[xkey] if profile == 'm' else rmid
        elif xaxis == 'm':
            m_edges = data[xkey]
            r_edges = 10**data['log_r']
            x = interp_powerlaw_edges_to_cells(r_edges, m_edges, rmid)

        y = data[profile]

        ax.plot( x, y, lw=2, color=cmap(ind % 10), label=f"t={data['time']:.2e}")

        if profile in ['kn', 'Theta'] and ind == 0:
            ax.axhline(1.0, color='black', ls=':')
            if profile == 'kn':
                ax.text( 0.95, 1.1, 'LMFP', ha='right', va='bottom', fontsize=12, transform=ax.get_yaxis_transform())
                ax.text( 0.95, 0.9, 'SMFP', ha='right', va='top', fontsize=12, transform=ax.get_yaxis_transform())

    # Cosmetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if xaxis == 'r':
        ax.set_xlabel(r'Radius [$r_\mathrm{s,0}$]', fontsize=14)
    elif xaxis == 'm':
        ax.set_xlabel(r'$M_\mathrm{enc}$ [$M_\mathrm{s}$]', fontsize=14)
    ax.set_ylabel(profile, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    if legend:
        if legend_loc is None:
            ax.legend()
        else:
            ax.legend(loc=legend_loc)
    if grid:
        ax.grid(True, which="both", ls="--")

def plot_snapshots(model, snapshots=[0], profiles='rho', xaxis=None, base_dir=None, filepath=None, show=False, grid=False, for_movie=False):
    """
    Plot up to three profiles at specified points in time for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    snapshots : int or list of int
        Snapshot indices to plot
    profiles : str or list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'kn'
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
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

    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis]

    def _resolve_dir(model, ind):
        if hasattr(model, 'config'): # Passed state object
            return os.path.join(model.config.io.base_dir, model.config.io.model_dir, f"profile_{ind}.dat")
        elif hasattr(model, 'io'): # Passed config object
            return os.path.join(model.io.base_dir, model.io.model_dir, f"profile_{ind}.dat")
        elif isinstance(model, int): # Passed model number
            if base_dir is None:
                raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
            model_dir = f"Model{model:05d}"
            return os.path.join(base_dir, model_dir, f"profile_{ind}.dat")
        else:
            raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")

    # Change any '-1' entries to the last snapshot index
    for ind, val in enumerate(snapshots):
        if val == -1:
            snapshot_indices_data = extract_snapshot_indices(os.path.dirname(_resolve_dir(model, 0)))
            snapshots[ind] = snapshot_indices_data['index'][-1]

    n = 1 if type(profiles) != list else len(profiles) # number of panels

    data_list = [extract_snapshot_data(_resolve_dir(model,ind)) for ind in snapshots]

    fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

    if n == 1:
        profile = profiles[0] if type(profiles) == list else profiles
        plot_profile(axs, profile, data_list, xaxis=xaxis[0], legend=True, grid=grid, for_movie=for_movie)
    else:
        for ind, ax in enumerate(axs):
            legend = False if ind < len(axs) - 1 else True
            plot_profile(ax, profiles[ind], data_list, xaxis=xaxis[ind], legend=legend, grid=grid, for_movie=for_movie)

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
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
        Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_{profiles}.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
    profiles : str or list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'kn'
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
        model_dir = f"Model{model:05d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load snapshot indices
    snapshot_indices_data = extract_snapshot_indices(model_dir)
    indices = snapshot_indices_data['index']

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

def make_movie_deluxe(model, profiles=None, insets=None, xaxis=None, add_radii=None, filepath=None, base_dir=None, grid=False, fps=20):
    """
    Animate profiles wit constant scale and with inset for time evolution.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    profiles : list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'kn', 'Theta'.
    insets : list of str or None, optional
        Inset plots to include.  Options are any quantity in time_evolution.txt
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
    add_radii : list, optional
        List of radii to add to profiles from time_evolution.txt
        Options: 'r_c', 'r_m2', 'r_smfp', 'r_minTh'
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_deluxe.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
    grid : bool, optional
        If True, shows grid on axes
    fps : int, optional
        Frames per second for the output movie. Default is 20

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """
    # Collect profiles and insets
    if profiles is None:
        profiles = ['rho', 'v2']
    elif isinstance(profiles, str):
        profiles = [profiles]
    if insets is None:
        insets = ['rho0'] + [None] * (len(profiles) - 1)
    elif isinstance(insets, str) or insets is None:
        insets = [insets]
    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis]

    # Validate profiles
    valid_profiles = ['rho', 'm', 'v2', 'kn', 'Theta']
    if any(profile not in valid_profiles for profile in profiles):
        raise ValueError(f"Invalid profile specified. Valid options are: {valid_profiles}")
    
    # Validate radii
    valid_radii = ['r_c', 'r_m2', 'r_smfp', 'r_minTh']
    if add_radii is not None:
        if isinstance(add_radii, str):
            add_radii = [add_radii]
        if any(radius not in valid_radii for radius in add_radii):
            raise ValueError(f"Invalid radius specified. Valid options are: {valid_radii}")
        
    # Validate xaxis
    valid_xaxis = ['r', 'm']
    if any(x not in valid_xaxis for x in xaxis):
        raise ValueError(f"Invalid x-axis specified. Valid options are: {valid_xaxis}")

    # Number of panels
    n = len(profiles) 

    # Get the model directory
    if hasattr(model, 'config'):        # Passed state object
        model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
    elif hasattr(model, 'io'):          # Passed config object
        model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
    elif isinstance(model, int):        # Passed model number
        if base_dir is None:
            raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
        model_dir = f"Model{model:05d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load rhoc time evolution data
    print(f"Getting time evolution data...")
    time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
    time_data = extract_time_evolution_data(time_evolution_path)
    tevo_t = time_data['time']

    # Validate insets
    valid_insets = list(time_data.keys())
    if any(inset not in valid_insets for inset in insets if inset is not None):
        raise ValueError(f"Invalid inset specified. Valid options are: {valid_insets}")
    if len(insets) != len(profiles):
        raise ValueError("'insets' must have the same length as 'profiles'.")

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']
    index_t                 = snapshot_indices_data['time']

    # Get axis limits
    print(f"Getting axis limits...")

    snapshot_data_list = []

    for ind in indices:
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")

        if not os.path.isfile(snapshot_path):
            continue

        snapshot_data_list.append(extract_snapshot_data(snapshot_path))

    axislims = {}

    for i, profile in enumerate(profiles):
        xlim, ylim = get_profile_axis_limits(profile, snapshot_data_list, xaxis=xaxis[i])
        axislims[profile] = (xlim, ylim)

    # Create a temporary directory for storing images
    temp_dir = os.path.join(model_dir, "temp_images")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)             # Delete the directory and all its contents
    os.makedirs(temp_dir)

    image_paths = []                        # List to store paths of generated images

    print(f"Generating {len(indices)} frames...")
    for ind in tqdm(indices, desc="Frames", unit="frame"):
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
        if not os.path.isfile(snapshot_path):
            continue                        # Skip if the snapshot file does not exist

        # Define the output image path for the current frame
        image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

        # Extract data for current frame and initial frame
        initial_snapshot_path   = os.path.join(model_dir, f"profile_0.dat")
        data_list               = [
            extract_snapshot_data(initial_snapshot_path), 
            extract_snapshot_data(snapshot_path)
            ]
        
        # Plot profile and initial profile
        fig, axs = plt.subplots(1, n, figsize=(6*n, 5))
        axs = np.atleast_1d(axs)

        for i, ax in enumerate(axs):
            profile = profiles[i]
            inset   = insets[i]
            xax     = xaxis[i]

            legend = True if i == 0 else False
            plot_profile(ax, profile, data_list, xaxis=xax, axislims=axislims[profile], legend=legend, legend_loc='lower left', grid=grid, for_movie=True)

            if add_radii is not None:
                for radius in add_radii:
                    r = np.interp(index_t[ind], tevo_t, time_data[radius])
                    if xax == 'r':
                        # If r is outside the x-axis limits, skip plotting
                        if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
                            continue
                        ax.axvline(r, color='red', ls='--', zorder=-10)
                        ax.text(r, ax.get_ylim()[0]*2.0, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)
                    elif xax == 'm':
                        m = np.interp(r, 10**data_list[1]['log_r'], data_list[1]['m'])
                        # If r is outside the x-axis limits, skip plotting
                        if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
                            continue
                        ax.axvline(m, color='red', ls='--', zorder=-10)
                        ax.text(m, ax.get_ylim()[0]*2.0, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)
                    # If r is outside the x-axis limits, skip plotting
                    # if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
                    #     continue
                    # ax.axvline(r, color='red', ls='--', zorder=-10)
                    # ax.text(r, ax.get_ylim()[0]*2.0, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)

            if inset is not None:
                tevo_y = time_data[inset]
                axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
                axin.axvline(index_t[ind], color='grey')
                axin.plot(tevo_t, tevo_y, color='black')
                axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_y),
                            color='red', s=50)
                axin.set_ylabel(inset, fontsize=12)
                axin.set_xlabel('$t$', fontsize=12)
                axin.set_yscale('log')
                axin.tick_params(
                    axis='both',
                    which='both',
                    labelbottom=False,
                    labelleft=False,
                    labeltop=False,
                    labelright=False,
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    direction='in'
                )

        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        image_paths.append(image_path)  # Add the image path to the list

    print("Compiling into a movie using ffmpeg...")

    if filepath is not None:
        output_movie_path = filepath
    else:
        output_movie_path = os.path.join(model_dir, f"movie_deluxe.mp4")

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

def make_movie_balberg(model, filepath=None, base_dir=None, grid=False, fps=20):
    """
    Animate profiles for comparison with Balberg study

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_{profiles}.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
    grid : bool, optional
        If True, shows grid on axes
    fps : int, optional
        Frames per second for the output movie. Default is 20

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """
    # Collect profiles and insets
    profiles = ['rho', 'v2']
    insets = ['rho0', 'minTheta']
    
    # Validate radii
    add_radii = ['r_c', 'r_m2', 'r_smfp', 'r_minTh']

    # Get the model directory
    if hasattr(model, 'config'):        # Passed state object
        model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
    elif hasattr(model, 'io'):          # Passed config object
        model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
    elif isinstance(model, int):        # Passed model number
        if base_dir is None:
            raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
        model_dir = f"Model{model:05d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load rhoc time evolution data
    print(f"Getting time evolution data...")
    time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
    time_data = extract_time_evolution_data(time_evolution_path)
    tevo_t = time_data['time']

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']
    index_t                 = snapshot_indices_data['time']

    # Get axis limits
    print(f"Getting axis limits...")

    snapshot_data_list = []

    for ind in indices:
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")

        if not os.path.isfile(snapshot_path):
            continue

        snapshot_data_list.append(extract_snapshot_data(snapshot_path))

    axislims = {}

    for profile in profiles:
        xlim, ylim = get_profile_axis_limits(profile, snapshot_data_list)
        axislims[profile] = (xlim, ylim)

    # Create a temporary directory for storing images
    temp_dir = os.path.join(model_dir, "temp_images")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)             # Delete the directory and all its contents
    os.makedirs(temp_dir)

    image_paths = []                        # List to store paths of generated images

    print(f"Generating {len(indices)} frames...")
    for ind in tqdm(indices, desc="Frames", unit="frame"):
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
        if not os.path.isfile(snapshot_path):
            continue                        # Skip if the snapshot file does not exist

        # Define the output image path for the current frame
        image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

        # Extract data for current frame and initial frame
        initial_snapshot_path   = os.path.join(model_dir, f"profile_0.dat")
        data_list               = [
            extract_snapshot_data(initial_snapshot_path), 
            extract_snapshot_data(snapshot_path)
            ]
        
        # Plot profile and initial profile
        fig, axs = plt.subplots(2, 2, figsize=(6*2, 5*2))
        axs = np.atleast_1d(axs)

        # Top row
        for i, ax in enumerate(axs[0]):
            profile = profiles[i]
            inset = insets[i]

            legend = True if i == 0 else False
            plot_profile(ax, profile, data_list, axislims=axislims[profile], legend=legend, legend_loc='lower left', grid=grid, for_movie=True)

            if add_radii is not None:
                for radius in add_radii:
                    r = np.interp(index_t[ind], tevo_t, time_data[radius])
                    # If r is outside the x-axis limits, skip plotting
                    if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
                        continue
                    ax.axvline(r, color='red', ls='--', zorder=-10)
                    ax.text(r, ax.get_ylim()[0]*2.0, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)

            if inset is not None:
                tevo_y = time_data[inset]
                axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
                axin.axvline(index_t[ind], color='grey')
                axin.plot(tevo_t, tevo_y, color='black')
                axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_y),
                            color='red', s=50)
                axin.set_ylabel(inset, fontsize=12)
                axin.set_xlabel('$t$', fontsize=12)
                axin.set_yscale('log')
                axin.tick_params(
                    axis='both',
                    which='both',
                    labelbottom=False,
                    labelleft=False,
                    labeltop=False,
                    labelright=False,
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    direction='in'
                )

        # Bottom row
        for i, ax in enumerate(axs[1]):
            if i == 0:
                yquant = time_data['m_c']
                xquant = time_data['rho_c']
                ax.loglog(xquant, yquant, color='black')
                ax.set_ylabel('$M_\\mathrm{core}$/$M_\\mathrm{s}$', fontsize=16)
                ax.set_xlabel('$\\rho_\\mathrm{core}$/$\\rho_\\mathrm{s}$', fontsize=16)
            elif i == 1:
                yquant = time_data['zeta_c']
                xquant = time_data['v2_c']
                ax.plot(xquant, yquant, color='black')
                ax.set_xscale('log')
                ax.set_ylabel('$\\zeta$', fontsize=16)
                ax.set_xlabel('$v^2_\\mathrm{core}$/$v^2_\\mathrm{s}$', fontsize=16)
            x = np.interp(index_t[ind], tevo_t, xquant)
            y = np.interp(index_t[ind], tevo_t, yquant)
            ax.scatter(x, y, color='red', s=50)
            ax.tick_params(axis='both', labelsize=12)

        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        image_paths.append(image_path)  # Add the image path to the list

    print("Compiling into a movie using ffmpeg...")

    if filepath is not None:
        output_movie_path = filepath
    else:
        output_movie_path = os.path.join(model_dir, f"movie_balberg.mp4")

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