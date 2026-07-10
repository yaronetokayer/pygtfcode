import numpy as np
import math
from numba import njit, float64

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def interp_linear_to_interfaces(r_edges_1d, q_cells_1d) -> np.ndarray:
    """
    Linearly interpolate a cell-centered quantity q to interface locations
    using the non-uniform-spacing-aware formula:

        fac_i   = (r_i   - r_{i-1}) / (r_{i+1} - r_{i-1})     for i = 1..N-1
        q_{i|i+1} = q_i + fac_i * (q_{i+1} - q_i)

    Here r_* are edge (interface) radii with length N+1, q_cells has length N,
    and the returned array has length N-1 (interfaces i=1..N-1).

    Parameters
    ----------
    r_edges_1d : (N+1,) float64
        Edge (interface) radii, monotonic increasing.
    q_cells_1d : (N,) float64
        Cell-centered quantity defined between edges.

    Returns
    -------
    out : (N-1,) float64
        Interpolated values at interfaces i=1..N-1.
    """
    # interfaces we fill are i = 1..N-1  -> indices 1: N in edge space
    num = r_edges_1d[1:-1] - r_edges_1d[:-2]          # r_i   - r_{i-1}
    den = r_edges_1d[2:]   - r_edges_1d[:-2]          # r_{i+1} - r_{i-1}
    fac = num / den                                    # shape (N-1,)

    qL = q_cells_1d[:-1]                               # left cell value (i)
    qR = q_cells_1d[1:]                                # right cell value (i+1)
    return qL + fac * (qR - qL)                        # shape (N-1,)

@njit(float64(float64[:], float64[:], float64), cache=True,)
def interp_pl_to_r(x, y, x_eval):
    """
    Interpolate y to x_eval assuming a local power law.

    The local relation is

        y(x) = y0 * (x / x0)**p,

    where p is determined from the two neighboring data points.

    This function can be used for either edge quantities or
    shell-centered quantities:

        interp_pl_to_r(r, m, r_eval)
        interp_pl_to_r(rmid, rho, r_eval)

    Interpolation is performed within the grid. Outside the grid, the
    nearest pair of points is used for power-law extrapolation.

    A special case is included for enclosed quantities satisfying

        x[0] = 0
        y[0] = 0.

    Since log-log interpolation through the origin is undefined, the
    first two strictly positive points are used to extrapolate inward.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Coordinates at which y is defined. Must be strictly increasing.
    y : ndarray, shape (N,)
        Values to interpolate. The two points defining the local power
        law must be strictly positive, except that x[0] = y[0] = 0 is
        supported.
    x_eval : float
        Coordinate at which to evaluate the interpolated quantity.

    Returns
    -------
    y_eval : float
        Interpolated value.
    """
    N = y.shape[0]

    if N == 1:
        return y[0]

    # Preserve exact endpoint values, including a possible (0, 0).
    if x_eval == x[0]:
        return y[0]

    if x_eval == x[N - 1]:
        return y[N - 1]

    # Locate the pair surrounding x_eval. The first or last pair is
    # used when extrapolating.
    if x_eval < x[0]:
        i = 0

    elif x_eval > x[N - 1]:
        i = N - 2

    else:
        lo = 0
        hi = N - 1

        while hi - lo > 1:
            mid = (lo + hi) // 2

            if x_eval < x[mid]:
                hi = mid
            else:
                lo = mid

        i = lo

    x0 = x[i]
    x1 = x[i + 1]
    y0 = y[i]
    y1 = y[i + 1]

    if x_eval == x0:
        return y0

    if x_eval == x1:
        return y1

    # Standard local power-law interpolation.
    if (
        x_eval > 0.0
        and x0 > 0.0
        and x1 > 0.0
        and y0 > 0.0
        and y1 > 0.0
    ):
        power = math.log(y1 / y0) / math.log(x1 / x0)
        return y0 * math.exp(power * math.log(x_eval / x0))

    # A power law cannot be inferred directly between (0, 0) and the
    # first positive point. Infer the inner slope from the first two
    # positive points and extrapolate inward.
    if (
        i == 0
        and x[0] == 0.0
        and y[0] == 0.0
        and x_eval > 0.0
    ):
        j0 = 1

        while j0 < N and (x[j0] <= 0.0 or y[j0] <= 0.0):
            j0 += 1

        j1 = j0 + 1

        while j1 < N and (x[j1] <= 0.0 or y[j1] <= 0.0):
            j1 += 1

        if j1 >= N:
            raise ValueError(
                "At least two positive points are required to "
                "infer a power law near the origin."
            )

        power = (
            math.log(y[j1] / y[j0])
            / math.log(x[j1] / x[j0])
        )

        return y[j0] * math.exp(
            power * math.log(x_eval / x[j0])
        )

    raise ValueError(
        "Power-law interpolation requires positive coordinates and "
        "positive values."
    )