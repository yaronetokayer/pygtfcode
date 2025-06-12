import numpy as np
from scipy.integrate import solve_ivp, quad
from pygtfcode.profiles.truncated_nfw import potential, rho_from_df_truncated
from pygtfcode.profiles.menc import menc
from pygtfcode.parameters.constants import Constants

# Potential
def potential(r, config):
    """
    Analytic potential for NFW profile with truncation constant Zt.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of scale radius r_s.
    config : Config

    Returns
    -------
    float or ndarray
        Potential value Phi(r).
    """
    r = np.asarray(r)
    Zt = config.init.Zt  # Only defined for TruncatedNFWParams

    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.log1p(r) / r
        phi[r == 0] = np.inf  # avoids divide-by-zero
    return phi - Zt

# Eddington inversion
def df_trunc(E, config):
    # Placeholder for the actual distribution function
    # Could use config.init.Zt, deltaP, etc.
    raise NotImplementedError("df_trunc needs implementation.")

def rho_from_df_truncated(Phi, config):
    """
    Compute rho(Phi) from the truncated DF using Eddington inversion.

    Parameters
    ----------
    Phi : float
        Gravitational potential value.
    config : Config

    Returns
    -------
    float
        Density at this potential.
    """
    if Phi <= 0.0:
        return 0.0

    integrand = lambda Z: df_trunc(Z, config) * np.sqrt(2 * (Phi - Z)) # Eddington inversion
    result, _ = quad(integrand, 0.0, Phi, epsabs=1e-5, epsrel=1e-5)
    return 4 * np.pi * result

# Mass integrand
def rho_r2_truncated_nfw(r, config, rad, pot):
    """
    Computes rho(r) * r^2 for use in Menc integral.

    Parameters
    ----------
    r : float
    config : Config
    rad : ndarray
        Radial grid used for potential.
    pot : ndarray
        Corresponding potential values.

    Returns
    -------
    float
        rho(r) * r^2
    """
    if r < rad[0]:
        rho = 1.0 / (r * (1 + r)**2)
    else:
        # Linear interpolation for Phi(r)
        j = np.searchsorted(rad, r) - 1
        j = np.clip(j, 0, len(rad) - 2)
        Phi = pot[j] + (r - rad[j]) / (rad[j+1] - rad[j]) * (pot[j+1] - pot[j])
        rho = rho_from_df_truncated(Phi, config)
    return rho * r**2

# Potential integration
def integrate_potential(config):
    """
    Integrates the Poisson equation outward from a small radius to compute
    the potential profile for a truncated NFW halo.

    Returns
    -------
    rad : ndarray
        Radial points at which potential is computed.
    pot : ndarray
        Corresponding potential values.
    rcut : float
        Final radius where potential drops to zero.
    Mtot : float
        Total enclosed mass at rcut.
    """
    epsilon = 1.0e-2
    deltaP = config.prec.eps_dt  # Or use config.init.deltaP if stored there
    cvir = config.init.cvir

    r0 = 0.01  # initial radius
    r1 = (1 - epsilon) * r0
    r2 = (1 + epsilon) * r0

    # initial potential and dPhi/dr via finite difference
    Phi0 = potential(r0)
    dPhidr0 = (potential(r2) - potential(r1)) / (r2 - r1)

    # initial conditions: y = [Phi, dPhi/dr]
    y0 = [Phi0, dPhidr0]

    # Prepare arrays
    rad = [r0]
    pot = [Phi0]
    rcur = r0
    dr = deltaP / dPhidr0
    rnext = rcur + dr
    rcut = None

    def rhs(r, y):
        Phi, dPhidr = y
        rho = rho_from_df_truncated(Phi, config)
        return [dPhidr, -4 * np.pi * Constants.gee * rho - 2 / r * dPhidr]

    while y0[0] > 0:
        sol = solve_ivp(rhs, (rcur, rnext), y0, method='RK45', max_step=(rnext - rcur),
                        rtol=1e-5, atol=1e-5)
        Phi_end = sol.y[0, -1]
        dPhidr_end = sol.y[1, -1]

        # store
        rad.append(rnext)
        pot.append(Phi_end)

        # update
        rcur = rnext
        y0 = [Phi_end, dPhidr_end]
        dr = deltaP / dPhidr_end
        if dr < 0:
            raise RuntimeError("Negative integration step: dr < 0")
        dr = min(dr, 0.01)
        rnext = rcur + dr
        rcut = rcur

    # Final adjustments
    pot[-1] = 0.0  # potential should decay to zero at large r
    rad = np.array(rad)
    pot = np.array(pot)

    # total mass
    Mtot = menc(rcut, config)

    # log10 update for xlgrmax (this affects grid generation)
    log_rcut = np.log10(0.99 * rcut)
    config.grid.rmax = min(config.grid.rmax, 10**log_rcut)

    # log output
    print(f"                rcut : {rcut:.6g}")
    print(f"           log[rmax] : {log_rcut:.6g}")
    print(f"      Mtot = M(rmax) : {Mtot:.6g}")
    print(f"             M(rvir) : {menc(cvir, config):.6g}")
    print()

    return rad, pot, rcut, Mtot
