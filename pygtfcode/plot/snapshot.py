import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
import subprocess
from tqdm import tqdm
import shutil
from pygtfcode.io.read import extract_snapshot_data, extract_snapshot_indices, extract_time_evolution_data
from concurrent.futures import ProcessPoolExecutor, as_completed

VALID_PROFILES = [
    'rho', 'm', 'v2', 'kn', 'mfp', 'kn_cond', 'mfp_cond',
    'x', 'K_L', 'K_S', 'mfp_cond_ltemp',
    'dttcool', 'tdyntcool', 'drfrac', 'drltemp', 'ltemp', 
    's', 'dsdr', 'dlnrhodlnp', 'tsctcool', 'mfpltemp', 'dlnrhodlnr', 'dlnvdlnr', 
    'k_sc', 'k_lc', 'k_totc', 'k_se', 'k_le', 'k_tote', 'krat_c', 'krat_e'
    ]
EDGE_QUANTITIES = ['m', 'k_se', 'k_le', 'k_tote', 'krat_e']
LINEAR_Y_PROFILES = ['s', 'dlnrhodlnr', 'dlnvdlnr', 'x']
SYMLOG_Y_PROFILES = ['dsdr', 'dlnrhodlnp']
# Conductivity equality occurs at krat=1, not generally at Kn=1.
LINE_AT_1_PROFILES = ['dttcool', 'tdyntcool', 'mfpltemp', 'mfp_cond_ltemp', 'drltemp', 'krat_c', 'krat_e']
VALID_RADII = ['r_c', 'r_m2', 'r_minTh', 'r_m25']

def get_profile_axis_limits(profile, data_list, xaxis='r'):
    if xaxis not in ('r', 'm'):
        raise ValueError("xaxis must be 'r' or 'm'")
    if xaxis == 'r':
        xkey = 'log_r' if profile in EDGE_QUANTITIES else 'log_rmid'
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
            if profile in EDGE_QUANTITIES:
                x = data['m']
            else:
                m_edges = data['m']
                x = np.empty_like(m_edges)
                x[0] = 0.5 * m_edges[0]
                x[1:] = 0.5 * (m_edges[:-1] + m_edges[1:])

        if profile not in data:
            raise ValueError(f"Profile {profile!r} is absent from this output file.")
        y = data[profile]

        if profile in LINEAR_Y_PROFILES or profile in SYMLOG_Y_PROFILES:
            finite_y = y[np.isfinite(y)]

            if finite_y.size > 0:
                y_min = np.nanmin(finite_y)
                y_max = np.nanmax(finite_y)

                y_range = y_max - y_min

                if y_range == 0:
                    pad = 0.1 * abs(y_max) if y_max != 0 else 1.0
                else:
                    pad = 0.1 * y_range

                ylim_lower = min(ylim_lower, y_min - pad)
                ylim_upper = max(ylim_upper, y_max + pad)

        else:
            # Existing log-style y-limit behavior
            positive_y = y[(y > 0) & np.isfinite(y)]

            if positive_y.size > 0:
                ylim_lower = min(ylim_lower, np.nanmin(positive_y) * 0.5)

            if np.any(np.isfinite(y)):
                ylim_upper = max(ylim_upper, np.max(y[np.isfinite(y)]) * 10)

        if np.any(np.isfinite(x)):
            xlim_lower = min(xlim_lower, np.nanmin(x) * 0.8)
            xlim_upper = max(xlim_upper, np.nanmax(x) * 1.2)

    if not np.isfinite(xlim_lower) or not np.isfinite(xlim_upper):
        raise ValueError('No finite coordinates to plot')
    if not np.isfinite(ylim_lower) or not np.isfinite(ylim_upper):
        # Initial rate diagnostics or legacy columns may be wholly undefined.
        ylim_lower, ylim_upper = ((-1.0, 1.0) if profile in LINEAR_Y_PROFILES
                                 or profile in SYMLOG_Y_PROFILES else (0.1, 10.0))

    return (xlim_lower, xlim_upper), (ylim_lower, ylim_upper)

def plot_profile(ax, profile, data_list, xaxis='r', axislims=None, legend=True, legend_loc=None, grid=False, for_movie=False):
    """
    Plot specified profile on the passed axis object

    Arguments
    ---------
    ax : Axis
        Axis object on which to plot
    profile : str
        Header-named profile to plot; see VALID_PROFILES.
    data_list : list of dict
        Snapshot dictionaries returned by extract_snapshot_data().
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

    if xaxis not in ('r', 'm'):
        raise ValueError("xaxis must be 'r' or 'm'")
    if xaxis == 'r':
        xkey = 'log_r' if profile in EDGE_QUANTITIES else 'log_rmid'
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
            x = 10**data[xkey] if profile in EDGE_QUANTITIES else rmid
        elif xaxis == 'm':
            m_edges = data[xkey]
            x = np.empty_like(m_edges)
            x[0] = 0.5 * m_edges[0]
            x[1:] = 0.5 * (m_edges[:-1] + m_edges[1:])
            if profile in EDGE_QUANTITIES:
                x = m_edges

        if profile not in data:
            raise ValueError(f"Profile {profile!r} is absent from this output file.")
        y = data[profile]

        ax.plot( x, y, lw=2, color=cmap(ind % 10), label=f"t={data['time']:.2e}")

        if profile in LINE_AT_1_PROFILES and ind == 0:
            ax.axhline(1.0, color='black', ls=':')
            if profile in ('krat_c', 'krat_e') and ylim[0] < 1.0 < ylim[1]:
                ax.text(0.95, 1.1, 'LMFP', ha='right', va='bottom', fontsize=12, transform=ax.get_yaxis_transform(), clip_on=True)
                ax.text(0.95, 0.9, 'SMFP', ha='right', va='top', fontsize=12, transform=ax.get_yaxis_transform(), clip_on=True)
        if profile in ['dsdr'] and ind == 0:
            ax.axhline(0.0, color='black', ls=':')
        if profile in ['dlnrhodlnp'] and ind == 0:
            ax.axhline(0.6, color='black', ls=':')

    # Cosmetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xscale('log')
    if profile in LINEAR_Y_PROFILES:
        ax.set_yscale('linear')
    elif profile in SYMLOG_Y_PROFILES:
        ax.set_yscale('symlog')
        ax.axhline(2.0, color='grey', ls='--', lw=1.0)
        ax.axhline(-2.0, color='grey', ls='--', lw=1.0)
    else:
        ax.set_yscale('log')
    if xaxis == 'r':
        ax.set_xlabel(r'Radius [$r_\mathrm{s}$]', fontsize=14)
    elif xaxis == 'm':
        ax.set_xlabel(r'$M_\mathrm{enc}$ [$M_\mathrm{s}$]', fontsize=14)
    if profile == 's':
        ax.set_ylabel(r'$s=\log(v^3/\rho)$', fontsize=14)
    else:
        labels = {'kn': 'Kn (amplitude reference)', 'mfp': 'Reference MFP [r_s]',
                  'kn_cond': 'Conductivity-effective Kn', 'mfp_cond': 'Effective transport length [r_s]',
                  'x': 'v / w', 'K_L': 'LMFP factor K_L', 'K_S': 'SMFP factor K_S',
                  'mfp_cond_ltemp': 'Effective transport length / temperature scale',
                  'krat_c': 'Cell conductivity ratio (SMFP / LMFP)',
                  'krat_e': 'Face conductivity ratio (SMFP / LMFP)'}
        ax.set_ylabel(labels.get(profile, profile), fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    if legend:
        if legend_loc is None:
            ax.legend()
        else:
            ax.legend(loc=legend_loc)
    if grid:
        ax.grid(True, which="both", ls="--")

def plot_snapshots(model, snapshots=None, profiles='rho', xaxis=None, base_dir=None, filepath=None, show=False, grid=False, for_movie=False):
    """
    Plot up to three profiles at specified points in time for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    snapshots : int or list of int
        Snapshot indices to plot
    profiles : str or list of str, optional
        Profiles from VALID_PROFILES that are present in the supplied files.
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

    if snapshots is None:
        snapshots = [0]
    elif isinstance(snapshots, (list, tuple, np.ndarray)):
        snapshots = list(snapshots)
    else:
        snapshots = [snapshots]
    profiles = [profiles] if isinstance(profiles, str) else list(profiles)
    if not profiles or not snapshots:
        raise ValueError("At least one profile and snapshot must be specified")

    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis] * len(profiles)

    if len(xaxis) != len(profiles):
        raise ValueError('xaxis must have one entry per profile')

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
    elif show:
        plt.show()
    return fig, axs

def _deluxe_frame(args):
    """
    Worker function for rendering one movie frame.

    Must be top-level so ProcessPoolExecutor can pickle it.
    """
    (
        ind, model_dir, temp_dir, n, profiles, insets, xaxis, add_radii,
        axislims, grid, index_t, tevo_t, time_data, vertical,
    ) = args

    import os
    import numpy as np

    # Safer for multiprocessing / headless rendering.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
    if not os.path.isfile(snapshot_path):
        return None

    image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

    initial_snapshot_path = os.path.join(model_dir, "profile_0.dat")
    data_list = [
        extract_snapshot_data(initial_snapshot_path),
        extract_snapshot_data(snapshot_path),
    ]

    if vertical:
        fig, axs = plt.subplots(
            n, 1,
            figsize=(6, 4 * n),
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)
    else:
        fig, axs = plt.subplots(
            1, n,
            figsize=(6 * n, 5),
        )

    axs = np.atleast_1d(axs)

    for i, ax in enumerate(axs):
        profile = profiles[i]
        inset = insets[i]
        xax = xaxis[i]

        legend = True if i == 0 else False

        plot_profile(
            ax,
            profile,
            data_list,
            xaxis=xax,
            axislims=axislims[profile],
            legend=legend,
            legend_loc="lower left",
            grid=grid,
            for_movie=True,
        )

        if add_radii is not None:
            for radius in add_radii:
                r = np.interp(index_t[ind], tevo_t, time_data[radius])

                text_y = 0.05

                if xax == "r":
                    if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
                        continue

                    ax.axvline(r, color="red", ls="--", zorder=-10)
                    ax.text(
                        r,
                        text_y,
                        radius,
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        color="red",
                        fontsize=10,
                        ha="right",
                        va="bottom",
                        zorder=-10,
                    )

                elif xax == "m":
                    m = np.interp(r, 10 ** data_list[1]["log_r"], data_list[1]["m"],)

                    if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
                        continue

                    ax.axvline(m, color="red", ls="--", zorder=-10)
                    ax.text(
                        m,
                        text_y,
                        radius,
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        color="red",
                        fontsize=10,
                        ha="right",
                        va="bottom",
                        zorder=-10,
                    )

        if inset is not None:
            tevo_y = time_data[inset]

            axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
            axin.axvline(index_t[ind], color="grey")
            axin.plot(tevo_t, tevo_y, color="black")
            axin.scatter(
                index_t[ind],
                np.interp(index_t[ind], tevo_t, tevo_y),
                color="red",
                s=50,
            )

            axin.set_ylabel(inset, fontsize=12)
            axin.set_xlabel("$t$", fontsize=12)
            axin.set_yscale("log" if np.any(np.isfinite(tevo_y) & (tevo_y > 0)) else "linear")
            axin.tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labelleft=False,
                labeltop=False,
                labelright=False,
                top=True,
                bottom=True,
                left=True,
                right=True,
                direction="in",
            )

    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return image_path

def make_movie_deluxe_serial(model, profiles=None, insets=None, xaxis=None, add_radii=None, filepath=None, base_dir=None, grid=False, fps=20):
    """
    Animate profiles wit constant scale and with inset for time evolution.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    profiles : list of str, optional
        Profiles from VALID_PROFILES that are present in the supplied files.
    insets : False, str, list of str or None, optional
        False disables all insets; None uses rho0 in the first panel.
        Inset plots to include.  Options are any quantity in time_evolution.txt
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
    add_radii : list, optional
        List of radii to add to profiles from time_evolution.txt
        Options: 'r_c', 'r_m2', 'r_minTh'
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
    if insets is False:
        insets = [None] * len(profiles)
    elif insets is True:
        raise ValueError('Use insets=None for defaults or False to disable all insets.')
    elif insets is None:
        insets = ['rho0'] + [None] * (len(profiles) - 1)
    elif isinstance(insets, str) or insets is None:
        insets = [insets]
    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis] * len(profiles)

    # Validate profiles
    if any(profile not in VALID_PROFILES for profile in profiles):
        raise ValueError(f"Invalid profile specified. Valid options are: {VALID_PROFILES}")
    
    # Validate radii
    if add_radii is not None:
        if isinstance(add_radii, str):
            add_radii = [add_radii]
        if any(radius not in VALID_RADII for radius in add_radii):
            raise ValueError(f"Invalid radius specified. Valid options are: {VALID_RADII}")
        
    # Validate xaxis
    valid_xaxis = ['r', 'm']
    if any(x not in valid_xaxis for x in xaxis):
        raise ValueError(f"Invalid x-axis specified. Valid options are: {valid_xaxis}")

    if not profiles or len(xaxis) != len(profiles):
        raise ValueError('Specify profiles and one xaxis entry per profile')

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
    needs_history = any(inset is not None for inset in insets) or bool(add_radii)
    time_data = extract_time_evolution_data(time_evolution_path) if needs_history else {}
    tevo_t = time_data.get('time', np.array([]))
    if add_radii is not None:
        missing = [radius for radius in add_radii if radius not in time_data]
        if missing:
            raise ValueError(f'Radii absent from this time history: {missing}')

    # Validate insets
    valid_insets = [key for key in time_data if key != 'model_id']
    if any(inset not in valid_insets for inset in insets if inset is not None):
        raise ValueError(f"Invalid inset specified. Valid options are: {valid_insets}")
    if len(insets) != len(profiles):
        raise ValueError("'insets' must have the same length as 'profiles'.")

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']
    index_t                 = dict(zip(indices, snapshot_indices_data['time']))

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
                        ax.text(r, 0.05, radius, transform=ax.get_xaxis_transform(), rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)
                    elif xax == 'm':
                        m = np.interp(r, 10**data_list[1]['log_r'], data_list[1]['m'])
                        # If r is outside the x-axis limits, skip plotting
                        if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
                            continue
                        ax.axvline(m, color='red', ls='--', zorder=-10)
                        ax.text(m, 0.05, radius, transform=ax.get_xaxis_transform(), rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)


            if inset is not None:
                tevo_y = time_data[inset]
                axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
                axin.axvline(index_t[ind], color='grey')
                axin.plot(tevo_t, tevo_y, color='black')
                axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_y),
                            color='red', s=50)
                axin.set_ylabel(inset, fontsize=12)
                axin.set_xlabel('$t$', fontsize=12)
                axin.set_yscale('log' if np.any(np.isfinite(tevo_y) & (tevo_y > 0)) else 'linear')
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

def make_movie_deluxe_parallel(model, profiles=None, insets=None, xaxis=None, add_radii=None, vertical=False, filepath=None, base_dir=None, grid=False, fps=20):
    """
    Animate profiles wit constant scale and with inset for time evolution.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    profiles : list of str, optional
        Profiles from VALID_PROFILES that are present in the supplied files.
    insets : False, str, list of str or None, optional
        False disables all insets; None uses rho0 in the first panel.
        Inset plots to include.  Options are any quantity in time_evolution.txt
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
    add_radii : list, optional
        List of radii to add to profiles from time_evolution.txt
        Options: 'r_c', 'r_m2', 'r_minTh'
    vertical : bool, optional
        Whether to stack panels vertically.
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
    if insets is False:
        insets = [None] * len(profiles)
    elif insets is True:
        raise ValueError('Use insets=None for defaults or False to disable all insets.')
    elif insets is None:
        insets = ['rho0'] + [None] * (len(profiles) - 1)
    elif isinstance(insets, str) or insets is None:
        insets = [insets]
    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis] * len(profiles)

    # Validate profiles
    if any(profile not in VALID_PROFILES for profile in profiles):
        raise ValueError(f"Invalid profile specified. Valid options are: {VALID_PROFILES}")
    
    # Validate radii
    if add_radii is not None:
        if isinstance(add_radii, str):
            add_radii = [add_radii]
        if any(radius not in VALID_RADII for radius in add_radii):
            raise ValueError(f"Invalid radius specified. Valid options are: {VALID_RADII}")
        
    # Validate xaxis
    valid_xaxis = ['r', 'm']
    if any(x not in valid_xaxis for x in xaxis):
        raise ValueError(f"Invalid x-axis specified. Valid options are: {valid_xaxis}")

    if not profiles or len(xaxis) != len(profiles):
        raise ValueError('Specify profiles and one xaxis entry per profile')

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
    
    # Load time evolution data
    print(f"Getting time evolution data...")
    time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
    needs_history = any(inset is not None for inset in insets) or bool(add_radii)
    time_data = extract_time_evolution_data(time_evolution_path) if needs_history else {}
    tevo_t = time_data.get('time', np.array([]))
    if add_radii is not None:
        missing = [radius for radius in add_radii if radius not in time_data]
        if missing:
            raise ValueError(f'Radii absent from this time history: {missing}')

    # Validate insets
    valid_insets = [key for key in time_data if key != 'model_id']
    if any(inset not in valid_insets for inset in insets if inset is not None):
        raise ValueError(f"Invalid inset specified. Valid options are: {valid_insets}")
    if len(insets) != len(profiles):
        raise ValueError("'insets' must have the same length as 'profiles'.")

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']
    index_t                 = dict(zip(indices, snapshot_indices_data['time']))

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

    # Determine number of parallel processes
    max_workers = max(1, min(os.cpu_count() - 2, 7))

    print(f"Generating {len(indices)} frames using {max_workers} parallel processes...")

    frame_args = [
        (
            ind,
            model_dir,
            temp_dir,
            n,
            profiles,
            insets,
            xaxis,
            add_radii,
            axislims,
            grid,
            index_t,
            tevo_t,
            time_data,
            vertical,
        )
        for ind in indices
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_deluxe_frame, args) for args in frame_args]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Frames", unit="frame"):
            image_path = future.result()
            if image_path is not None:
                image_paths.append(image_path)

    # Keep list deterministic (although we never end up using it)
    image_paths.sort()

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

def make_movie(model, parallel=True, **kwargs):
    """Animate profiles using the deluxe renderer with fixed axis limits.

    parallel=True uses separate rendering processes; False renders serially.
    profiles defaults to ['rho', 'v2']. insets=None adds rho0 to the first
    panel; insets=False disables all insets. A per-panel list of column names
    and None entries is also accepted. time_evolution.txt is required only
    for insets or marked radii.

    Other options: xaxis, add_radii, filepath, base_dir, grid, and fps.
    The parallel renderer additionally accepts vertical.
    Requires ffmpeg. See make_movie_deluxe_serial/parallel for details.
    """
    renderer = make_movie_deluxe_parallel if parallel else make_movie_deluxe_serial
    return renderer(model, **kwargs)


def make_movie_deluxe(model, parallel=True, **kwargs):
    """Compatibility alias for make_movie, which now uses the deluxe renderer."""
    return make_movie(model, parallel=parallel, **kwargs)

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
    insets = ['rho0', 'kn_cond_c']
    
    # Validate radii
    add_radii = ['r_c', 'r_m2']  # Radii present in current time histories.

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
    if add_radii is not None:
        missing = [radius for radius in add_radii if radius not in time_data]
        if missing:
            raise ValueError(f'Radii absent from this time history: {missing}')

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']
    index_t                 = dict(zip(indices, snapshot_indices_data['time']))

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
                    ax.text(r, 0.05, radius, transform=ax.get_xaxis_transform(), rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10)

            if inset is not None:
                tevo_y = time_data[inset]
                axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
                axin.axvline(index_t[ind], color='grey')
                axin.plot(tevo_t, tevo_y, color='black')
                axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_y),
                            color='red', s=50)
                axin.set_ylabel(inset, fontsize=12)
                axin.set_xlabel('$t$', fontsize=12)
                axin.set_yscale('log' if np.any(np.isfinite(tevo_y) & (tevo_y > 0)) else 'linear')
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
                yquant = time_data['zeta_balb'] if 'zeta_balb' in time_data else time_data['zeta_c']
                xquant = time_data['v2_c']
                ax.plot(xquant, yquant, color='black')
                ax.set_xscale('log')
                ax.set_ylabel('$\\zeta$', fontsize=16)
                ax.set_xlabel('$v^2_\\mathrm{core}$/$v^2_\\mathrm{s}$', fontsize=16)
            finite_x = xquant[np.isfinite(xquant) & (xquant > 0)]
            if finite_x.size and finite_x.max() / finite_x.min() < 2:
                # Log tick labels otherwise crowd very short histories.
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(ScalarFormatter())
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
