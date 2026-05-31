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

@njit(float64[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def interp_powerlaw_edges_to_cells(r_edges_1d, q_edges_1d, r_cells_1d) -> np.ndarray:
    """
    Power-law interpolate edge values q_edges_1d to cell-center locations.
    Assumes that within each cell, q(r) = q_L * (r / r_L)^a

    All input arrays are the same length; implied left edge is (r,q) = (0,0).

    Requires:
        r_edges_1d > 0
        q_edges_1d > 0
        r_cells_1d[i] inside [r_edges_1d[i], r_edges_1d[i+1]]
    """
    N = r_cells_1d.shape[0]
    out = np.empty(N, dtype=np.float64)

    # First cell: use linear interpolation from (0,0) to (r_edges_1d[0], q_edges_1d[0])
    out[0] = q_edges_1d[0] * r_cells_1d[0] / r_edges_1d[0]

    # Remaining cells: use power-law interpolation between edges
    log_rL = math.log(r_edges_1d[0])
    log_qL = math.log(q_edges_1d[0])

    for i in range(N):
        log_rC = math.log(r_cells_1d[i])

        log_rR = math.log(r_edges_1d[i + 1])
        log_qR = math.log(q_edges_1d[i + 1])

        fac = (log_rC - log_rL) / (log_rR - log_rL)

        out[i] = math.exp(log_qL + fac * (log_qR - log_qL))

        log_rL = log_rR
        log_qL = log_qR

    return out