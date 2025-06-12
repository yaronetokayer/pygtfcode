import numpy as np

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

