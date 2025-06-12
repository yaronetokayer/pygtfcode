import numpy as np

def setup_grid(config):
    """
    Constructs the radial grid in log-space between rmin and rmax.

    Parameters
    ----------
    config : Config
        The simulation configuration object.

    Returns
    -------
    r : ndarray of shape (Ngrid + 1,)
        Radial Lagrangian grid points, with r[0] = 0 and the rest spaced
        logarithmically between rmin and rmax.
    """
    rmin = config.grid.rmin
    rmax = config.grid.rmax
    Ngrid = config.grid.Ngrid

    xlgrmin = np.log10(rmin)
    xlgrmax = np.log10(rmax)

    r = np.empty(Ngrid + 1)
    r[0] = 0.0

    xlgr = np.linspace(xlgrmin, xlgrmax, Ngrid)
    r[1:] = 10**xlgr

    return r

def initialize_grid(r, config):
    """
    Computes initial physical quantities on the radial grid.

    Returns
    -------
    dict of ndarray
        Keys: 'M', 'rho', 'P', 'u', 'v2'
    """
    ...
