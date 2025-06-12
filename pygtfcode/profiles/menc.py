from pygtfcode.profiles.nfw import fNFW
from pygtfcode.profiles.truncated_nfw import toint4
from pygtfcode.profiles.abg import toint4b
from scipy.integrate import quad


def menc(r, config):
    """
    Compute the enclosed mass M(r) in units of Mvir.

    Parameters
    ----------
    r : float or ndarray
        Radius (in units of r_s).
    config : Config
        Global simulation configuration object.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass M(<r) in units of Mvir.
    """
    profile = config.init.profile

    if profile == "nfw":
        return fNFW(r)

    elif profile == "truncated_nfw":
        result, _ = quad(lambda x: toint4(x, config), 0.0, r, epsabs=1e-5, epsrel=1e-5)
        return result

    elif profile == "abg":
        result, _ = quad(lambda x: toint4b(x, config), 0.0, r, epsabs=1e-5, epsrel=1e-5)
        return result

    else:
        raise ValueError(f"Unknown profile type: {profile}")
