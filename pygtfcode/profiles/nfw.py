import numpy as np
from scipy.integrate import quad

def fNFW(r):
    """
    Analytic NFW mass profile: M(<r) / m_s

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass in units of m_s; fNFW(cvir) is the virial normalization.
    """
    
    r = np.asarray(r, dtype=np.float64)

    return np.log(1.0 + r) - r / (1.0 + r)

def _nfw_velocity_integrand(x):
    fac = np.log(1.0 + x) - x / (1.0 + x)
    return fac / (x**3 * (1.0 + x)**2)

def menc_nfw(r):
    return fNFW(r)

def sigr_nfw(r, config):
    """
    Velocity dispersion squared at radius r (in units of v_s^2).

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.

    Returns
    -------
    v2 : float or ndarray
        One-dimensional velocity dispersion squared in units of v_s^2.
    """
    epsabs = float(config.prec.epsabs)
    epsrel = float(config.prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in np.ndenumerate(r):
        ri_f = float(ri)
        integral, _ = quad(_nfw_velocity_integrand, ri_f, np.inf, epsabs=epsabs, epsrel=epsrel)
        out[i] = ri * (1.0 + ri)**2 * integral

    return float(out) if out.ndim == 0 else out
