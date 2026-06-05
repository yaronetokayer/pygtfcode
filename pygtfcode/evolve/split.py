import numpy as np
import math
from numba import njit, float64, types, void, int64

STATUS_NO_SPLITS = 0
STATUS_SPLITS = 1
_TINY64 = np.finfo(np.float64).tiny

@njit((float64[:], int64[:], float64), cache=True, fastmath=True)
def check_drfrac(r, nsplit, drfrac_max):
    """
    Check the dr/r condition in each radial cell.

    For each cell i, compute

        drfrac = (r[i+1] - r[i]) / sqrt(r[i] * r[i+1])

    If drfrac exceeds drfrac_max, compute how many additional
    splits are needed so that each child cell satisfies the same
    condition, assuming equal logarithmic spacing inside the cell.

    The array nsplit is updated in-place:

        nsplit[i] = 0  -> no split
        nsplit[i] = 1  -> split into 2 pieces
        nsplit[i] = 2  -> split into 3 pieces
        etc.

    The innermost cell, i = 0, is never split.

    Returns
    -------
    status : int
        STATUS_NO_SPLITS if no cells need splitting.
        STATUS_SPLITS if at least one cell needs splitting.
    """
    n = nsplit.size
    status = STATUS_NO_SPLITS

    nsplit[0] = 0

    # For a cell with q = rout/rin,
    #
    #   drfrac = (q - 1) / sqrt(q)
    #          = 2 sinh(0.5 log q)
    #
    # Therefore the maximum allowed log-width is
    #
    #   dlogr_max = 2 asinh(0.5 * drfrac_max)
    #

    dlogr_max = 2.0 * math.asinh(0.5 * drfrac_max)

    for i in range(1, n):
        nsplit[i] = 0

        rin = r[i]
        rout = r[i + 1]

        if rin <= _TINY64 or rout <= rin:
            continue

        dlogr = np.log(rout / rin)

        if dlogr > dlogr_max:
            # Number of equal-log pieces needed.
            npieces = int(math.ceil(dlogr / dlogr_max))

            # Number of additional splits/inserted edges.
            nsplit[i] = npieces - 1

            status = STATUS_SPLITS

    return status

@njit((float64[:], float64[:], float64[:], int64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def _split_grid_kernel(r, m, v2, nsplit, r_new, m_new, v2_new, rho_new):
    """
    Numba kernel to fill new split arrays.
    New cell edges are even in log space in the old cell.
    Mass is conserved.
    v2 is the same as the parent cell to conserve energy.
    rho is derived from mcell.
    """
    n_old = nsplit.size

    j = 0

    #--- Iterate through the new cells

    r_new[0] = r[0]
    m_new[0] = m[0]

    rR = r[0]; mR = m[0]
    for i in range(n_old):
        rL = rR
        rR = r[i + 1]

        mL = mR
        mR = m[i + 1]

        npieces = nsplit[i] + 1
        inv_npieces = 1.0 / npieces

        # Fill the right edges of the new r and m subcells,
        # and then cell-centered quantities
        for k in range(1, npieces + 1):
            frac = k * inv_npieces
            j_edge = j + k

            # Equal-log spacing.
            if rL > 0.0:
                r_new[j_edge] = rL * (rR / rL) ** frac
            else:
                if npieces == 1:
                    r_new[j_edge] = rR
                else:
                    raise RuntimeError("Encountered innermost cell split. This shouldn't happen.")

            # Linear interpolation of cumulative mass across the cell.
            m_new[j_edge] = mL + frac * (mR - mL)

            j_cell = j_edge - 1

            r_edge = r_new[j_edge]
            r_prev = r_new[j_edge - 1]

            dm_cell     = m_new[j_edge] - m_new[j_edge - 1]
            dr3_cell    = r_edge*r_edge*r_edge - r_prev*r_prev*r_prev

            rho_new[j_cell] = 3.0 * dm_cell / dr3_cell
            v2_new[j_cell]  = v2[i]

        j += npieces

def split_grid(state, nsplit):
    """
    Master function to split the r grid and recompute m, v2, rho
    Reassign the state arrays for the rest of the integration loop
    """
    n_old = state.n
    n_extra = int(np.sum(nsplit[:n_old]))
    n_new = n_old + n_extra

    if n_extra == 0:
        return
    
    # Allocate new arrays
    r_new   = np.empty(n_new + 1, dtype=np.float64)
    m_new   = np.empty(n_new + 1, dtype=np.float64)
    v2_new  = np.empty(n_new, dtype=np.float64)
    rho_new = np.empty(n_new, dtype=np.float64)

    # Remap function
    _split_grid_kernel(
        state.r, state.m, state.v2, nsplit,
        r_new, m_new, v2_new, rho_new,
    )

    # Checks
    if not np.all(np.diff(r_new) > 0.0):
        raise RuntimeError("split_grid produced non-monotonic r_new.")

    if not np.all(np.diff(m_new) >= 0.0):
        raise RuntimeError("split_grid produced non-monotonic m_new.")

    # replace state arrays
    state.r     = r_new
    state.m     = m_new
    state.v2    = v2_new
    state.rho   = rho_new

    state.n     = n_new
