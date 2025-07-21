from scipy.integrate import quad
import numpy as np

def chi(config):
    """
    Computes the chi parameter used in the ABG profile normalization.

    Parameters
    ----------
    config : Config
        The simulation configuration object. Must use an ABG profile.

    Returns
    -------
    float
        The value of chi.
    """
    alpha = config.init.alpha
    beta = config.init.beta
    gamma = config.init.gamma
    expo = (beta - gamma) / alpha

    def chi_integrand(x):
        return x**(2 - gamma) / (1 + x**alpha)**expo

    result, _ = quad(chi_integrand, 0.0, 1e4, epsabs=config.prec.epsabs, epsrel=config.prec.epsrel)
    return result

def _abg_jeans_mass_integrand(x, alpha, beta, gamma):
    """
    Mass integrand in the spherical Jeans equation for ABG profile.
    """
    return x**(2 - gamma) / (1 + x**alpha)**((beta - gamma) / alpha)

def _abg_velocity_integrand(x, alpha, beta, gamma, epsabs, epsrel):
    """
    Integrand for the velocity dispersion from the Jeans equation.
    """
    chi_integrand = lambda y: y**(2 - gamma) / (1 + y**alpha)**((beta - gamma) / alpha)
    chi_x, _ = quad(chi_integrand, 0.0, x, epsabs=epsabs, epsrel=epsrel)
    rho_x = x**(-gamma) / (1 + x**alpha)**((beta - gamma) / alpha)
    return rho_x * chi_x / x**2

def menc_abg(r, config):
    """
    Enclosed mass profile for the alpha-beta-gamma profile.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    config : Config
        The simulation configuration object (must use ABG profile).

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass in units of Mvir.
    """
    alpha = config.init.alpha
    beta = config.init.beta
    gamma = config.init.gamma

    r = np.atleast_1d(r)
    result = np.empty_like(r)

    for i, ri in enumerate(r):
        integral, _ = quad(_abg_jeans_mass_integrand, 0, ri, args=(alpha, beta, gamma), epsabs=config.prec.epsabs, epsrel=config.prec.epsrel)
        result[i] = integral

    return result if result.shape[0] > 1 else result[0] 

def sigr_abg(r, config):
    """
    Velocity dispersion squared for alpha-beta-gamma profile at radius r.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    config : Config
        The simulation configuration object (must use ABG profile).

    Returns
    -------
    v2 : float or ndarray
        Velocity dispersion squared.
    """
    alpha = config.init.alpha
    beta = config.init.beta
    gamma = config.init.gamma

    epsabs=config.prec.epsabs
    epsrel=config.prec.epsrel

    r = np.atleast_1d(r)
    result = np.empty_like(r)

    for i, ri in enumerate(r):
        integrand = lambda x: _abg_velocity_integrand(x, alpha, beta, gamma, epsabs, epsrel)
        integral, _ = quad(integrand, ri, np.inf, epsabs=epsabs, epsrel=epsrel)
        rho_ri = ri**(-gamma) / (1 + ri**alpha)**((beta - gamma) / alpha)
        result[i] = integral / rho_ri

    return result if result.shape[0] > 1 else result[0]