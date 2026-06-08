import numpy as np
import math
from numba import njit, float64, types, void, int64

STATUS_NO_SPLITS = 0; STATUS_SPLITS = 1
STATUS_NO_MERGES = 0; STATUS_MERGES = 1
_TINY64 = np.finfo(np.float64).tiny

@njit((float64[:], int64[:], float64), cache=True, fastmath=True)
def check_drfrac_split(r, nsplit, drfrac_max):
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

@njit((float64[:], float64[:], float64[:], int64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def _split_grid_kernel_pl(r, m, v2, nsplit, r_new, m_new, v2_new, rho_new):
    """
    Numba kernel to fill new split arrays.

    New cell edges are even in log space in the old cell.
    Mass is conserved.

    v2 is reconstructed as a local power law in radius,
    then renormalized so that dm*v2 is conserved inside
    each parent cell.

    rho is derived from mcell.
    """
    n_old = nsplit.size

    j = 0

    r_new[0] = r[0]; m_new[0] = m[0]

    rR = r[0]; mR = m[0]

    for i in range(n_old):
        rL = rR; rR = r[i + 1]

        mL = mR; mR = m[i + 1]

        npieces = nsplit[i] + 1
        inv_npieces = 1.0 / npieces

        dm_parent = mR - mL
        e_parent = dm_parent * v2[i]

        # --------------------------------------------------
        # Estimate local power-law slope:
        #
        #     v2(r) ~ r^alpha
        #
        # using neighboring cell-centered values.
        # --------------------------------------------------

        alpha = 0.0

        if v2[i] > 0.0:
            rci = 0.5 * (rL + rR)

            if i > 0 and i < n_old - 1:
                rc_minus = 0.5 * (r[i - 1] + r[i])
                rc_plus  = 0.5 * (r[i + 1] + r[i + 2])

                if (
                    rc_minus > 0.0
                    and rc_plus > rc_minus
                    and v2[i - 1] > 0.0
                    and v2[i + 1] > 0.0
                ):
                    alpha = math.log(v2[i + 1] / v2[i - 1]) / math.log(rc_plus / rc_minus)

            elif i > 0:
                rc_minus = 0.5 * (r[i - 1] + r[i])

                if rc_minus > 0.0 and rci > rc_minus and v2[i - 1] > 0.0:
                    alpha = math.log(v2[i] / v2[i - 1]) / math.log(rci / rc_minus)

            elif i < n_old - 1:
                rc_plus = 0.5 * (r[i + 1] + r[i + 2])

                if rc_plus > rci and v2[i + 1] > 0.0:
                    alpha = math.log(v2[i + 1] / v2[i]) / math.log(rc_plus / rci)

        # --------------------------------------------------
        # First child-cell pass:
        # build r_new, m_new, rho_new, and provisional v2_new.
        # Also accumulate provisional child energy.
        # --------------------------------------------------

        e_trial = 0.0

        for k in range(1, npieces + 1):
            frac = k * inv_npieces
            j_edge = j + k

            if rL > 0.0:
                r_new[j_edge] = rL * (rR / rL) ** frac
            else:
                if npieces == 1:
                    r_new[j_edge] = rR
                else:
                    raise RuntimeError("Encountered innermost cell split. This shouldn't happen.")

            m_new[j_edge] = mL + frac * (mR - mL)

            j_cell = j_edge - 1

            r_edge = r_new[j_edge]
            r_prev = r_new[j_edge - 1]

            dm_cell = m_new[j_edge] - m_new[j_edge - 1]
            dr3_cell = r_edge*r_edge*r_edge - r_prev*r_prev*r_prev

            rho_new[j_cell] = 3.0 * dm_cell / dr3_cell

            # Child cell center. Use arithmetic midpoint for consistency.
            rc_child = 0.5 * (r_prev + r_edge)

            if rci > 0.0 and rc_child > 0.0 and v2[i] > 0.0:
                v2_trial = v2[i] * (rc_child / rci) ** alpha
            else:
                v2_trial = v2[i]

            v2_new[j_cell] = v2_trial
            e_trial += dm_cell * v2_trial

        # --------------------------------------------------
        # Second child-cell pass:
        # renormalize v2_new so parent dm*v2 is conserved.
        # --------------------------------------------------

        if e_trial > 0.0:
            fac = e_parent / e_trial

            for k in range(npieces):
                v2_new[j + k] *= fac
        else:
            for k in range(npieces):
                v2_new[j + k] = v2[i]

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
    _split_grid_kernel_pl(
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

@njit((float64[:], int64[:], float64, float64), cache=True, fastmath=True)
def check_drfrac_merge(r, merge_mask, drfrac_min, drfrac_max):
    """
    Check whether adjacent cells are small enough to merge.

    merge_mask[i] = 1 means merge cell i with cell i+1.
    merge_mask[i] = 0 means no merge starts at cell i.

    The innermost cell, i = 0, is never merged.
    """
    n = merge_mask.size
    status = STATUS_NO_MERGES

    dlogr_min = 2.0 * math.asinh(0.5 * drfrac_min)
    dlogr_max = 2.0 * math.asinh(0.5 * drfrac_max)

    for i in range(n):
        merge_mask[i] = 0

    i = 1
    while i < n - 1:
        r0 = r[i]
        r1 = r[i + 1]
        r2 = r[i + 2]

        if r0 <= _TINY64 or r1 <= r0 or r2 <= r1:
            i += 1
            continue

        dlogr_i     = math.log(r1 / r0)
        dlogr_ip    = math.log(r2 / r1)
        dlogr_merge = math.log(r2 / r0)

        if (
            dlogr_i < dlogr_min
            and dlogr_ip < dlogr_min
            and dlogr_merge < dlogr_max
        ):
            merge_mask[i] = 1
            status = STATUS_MERGES
            i += 2
        else:
            i += 1

    return status

@njit((float64[:], float64[:], float64[:], int64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def _merge_grid_kernel(r, m, v2, merge_mask,
                       r_new, m_new, v2_new, rho_new):
    """
    Numba kernel to merge adjacent cells.

    merge_mask[i] = 1 means merge old cell i with old cell i+1.

    Conserves:
        - cumulative mass m
        - shell thermal energy dm * v2

    rho is recomputed from the merged shell mass and volume.
    """
    n_old = merge_mask.size

    # i = index in old grid
    # j = index in new grid
    i = 0
    j = 0

    # Copy leftmost edge.
    r_new[0] = r[0]; m_new[0] = m[0]

    while i < n_old:

        ########################################################
        # Case 1: merge cell i with cell i+1
        ########################################################

        if merge_mask[i] == 1:

            # Old cumulative masses:
            #
            # m[i] ----- m[i+1] ----- m[i+2]
            #
            # giving two shell masses dm1 and dm2.

            mL = m[i]
            mM = m[i + 1]
            mR = m[i + 2]

            dm1 = mM - mL
            dm2 = mR - mM
            dm  = dm1 + dm2

            # The merged shell ends at the old outer edge
            # of cell i+1.

            j_edge = j + 1

            r_new[j_edge] = r[i + 2]
            m_new[j_edge] = mR

            # Mass-weighted average preserves dm*v2.

            if dm > 0.0:
                v2_new[j] = (
                    dm1 * v2[i]
                    + dm2 * v2[i + 1]
                ) / dm
            else:
                v2_new[j] = v2[i]

            # Recompute density from shell mass and shell volume.

            r_edge = r_new[j_edge]
            r_prev = r_new[j_edge - 1]

            dr3_cell = (
                r_edge*r_edge*r_edge
                - r_prev*r_prev*r_prev
            )

            rho_new[j] = 3.0 * dm / dr3_cell

            # Skip both old cells.
            i += 2
            j += 1

        ########################################################
        # Case 2: copy cell i unchanged
        ########################################################

        else:

            mL = m[i]
            mR = m[i + 1]

            dm = mR - mL

            j_edge = j + 1

            # Copy outer edge and cumulative mass.

            r_new[j_edge] = r[i + 1]
            m_new[j_edge] = mR

            # Cell-centered quantities are unchanged.

            v2_new[j] = v2[i]

            r_edge = r_new[j_edge]
            r_prev = r_new[j_edge - 1]

            dr3_cell = (
                r_edge*r_edge*r_edge
                - r_prev*r_prev*r_prev
            )

            rho_new[j] = 3.0 * dm / dr3_cell

            i += 1
            j += 1

def merge_grid(state, merge_mask):
    """
    Master function to merge adjacent cells and recompute m, v2, rho.

    merge_mask[i] = 1 means merge cell i with cell i+1.
    """
    n_old = state.n
    n_merge = int(np.sum(merge_mask[:n_old]))
    n_new = n_old - n_merge

    if n_merge == 0:
        return
    
    # Allocate new arrays.
    r_new   = np.empty(n_new + 1, dtype=np.float64)
    m_new   = np.empty(n_new + 1, dtype=np.float64)
    v2_new  = np.empty(n_new, dtype=np.float64)
    rho_new = np.empty(n_new, dtype=np.float64)

    # Remap function
    _merge_grid_kernel(
        state.r, state.m, state.v2, merge_mask,
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