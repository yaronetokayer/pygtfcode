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

    result, _ = quad(chi_integrand, 0.0, 1e4, epsabs=1e-6, epsrel=1e-6)
    return result
