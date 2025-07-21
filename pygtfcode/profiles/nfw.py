import numpy as np
from scipy.integrate import quad

def fNFW(r):
    """
    Analytic NFW mass profile: M(<r) / Mvir

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass as a fraction of Mvir.
    """
    
    return np.log(1 + r) - r / (1 + r)

def _nfw_velocity_integrand(x):
    fac = np.log(1.0 + x) - x / (1.0 + x)
    return fac / (x**3 * (1.0 + x)**2)

def menc_nfw(r):
    return fNFW(r)

def sigr_nfw(r, config):
    """
    Velocity dispersion squared at radius r (in units of v0^2).

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.

    Returns
    -------
    v2 : float or ndarray
        Velocity dispersion squared.
    """
    r = np.atleast_1d(r)
    result = np.empty_like(r)

    for i, ri in enumerate(r):
        integral, _ = quad(_nfw_velocity_integrand, ri, np.inf, epsabs=config.prec.epsabs, epsrel=config.prec.epsrel)
        result[i] = ri * (1 + ri)**2 * integral

    return result if result.shape[0] > 1 else result[0]
