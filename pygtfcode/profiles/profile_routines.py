import numpy as np
from pygtfcode.profiles.nfw import menc_nfw, sigr_nfw
from pygtfcode.profiles.abg import menc_abg, sigr_abg
from pygtfcode.profiles.truncated_nfw import menc_trunc, sigr_trunc

def _as_f64(x):
    """
    Helper function to ensure double precision for all input values
    """
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim else float(a)

def menc(r, state, **kwargs):
    """
    Compute enclosed mass at radius r, in units of m_s.

    Parameters
    ----------
    r : float or array-like
        Radius in units of scale radius (r / r_s).
    state : State
        The simulation state object.

    Returns
    -------
    float or ndarray
        Enclosed mass at r, normalized by m_s.
    """
    r = _as_f64(r)
    profile = state.config.init.profile
    if profile == "nfw":
        return menc_nfw(r)
    elif profile == "truncated_nfw":
        return menc_trunc(r, state, **kwargs)
    elif profile == "abg":
        return menc_abg(r, state.config)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")

def sigr(r, state):
    """
    Compute radial velocity dispersion squared v^2(r).

    Parameters
    ----------
    r : float or array-like
        Radius in units of scale radius (r / r_s).
    state : State
        The simulation state object.

    Returns
    -------
    float or ndarray
        One-dimensional velocity dispersion squared in units of v_s^2.
    """
    r = _as_f64(r)
    profile = state.config.init.profile
    if profile == "nfw":
        return sigr_nfw(r, state.config)
    elif profile == "truncated_nfw":
        return sigr_trunc(r, state)
    elif profile == "abg":
        return sigr_abg(r, state.config)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")
