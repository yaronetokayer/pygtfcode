import math

class GridParams:
    """
    Parameters defining the radial grid for the simulation.

    Attributes
    ----------
    rmin : float
        First positive edge radius in units of r_s; the central edge is zero.
    rmax : float
        Maximum radius of the grid, in units of the scale radius (r / r_s).
    drfrac_init : float
        Initial shell width dr/sqrt(r_inner*r_outer); sets the cell count.
    grid_splitting : bool
        Enable adaptive splitting and merging outside the innermost cell.
    drfrac_max : float
        Width threshold for splitting and maximum allowed merged width.
    drfrac_min : float
        Both adjacent widths must be below this threshold to merge.
        Must be smaller than drfrac_max.
    """

    def __init__(
        self, 
        rmin: float = 1e-2,
        rmax: float = 2e2,
        drfrac_init: float = 5.0e-2,
        grid_splitting : bool = True,
        drfrac_max : float = 1.0e-1,
        drfrac_min : float = 1.0e-2,
    ):
        self._rmin = None
        self._rmax = None
        self._drfrac_init = None
        self._grid_splitting = None
        self._drfrac_max = None
        self._drfrac_min = None

        self.rmin = rmin
        self.rmax = rmax
        self.drfrac_init = drfrac_init
        self.grid_splitting = grid_splitting
        self.drfrac_max = drfrac_max
        self.drfrac_min = drfrac_min

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0 or (self._rmax is not None and value >= self._rmax):
            raise ValueError("Require 0 < rmin < rmax")
        self._rmin = float(value)

    @property
    def rmax(self):
        return self._rmax

    @rmax.setter
    def rmax(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0 or (self._rmin is not None and value <= self._rmin):
            raise ValueError("Require 0 < rmin < rmax")
        self._rmax = float(value)

    @property
    def drfrac_init(self):
        return self._drfrac_init

    @drfrac_init.setter
    def drfrac_init(self, value):
        self._validate_positive(value, "drfrac_init")
        self._drfrac_init = float(value)

    @property
    def grid_splitting(self):
        return self._grid_splitting
    
    @grid_splitting.setter
    def grid_splitting(self, value):
        if not isinstance(value, bool):
            raise ValueError("grid_splitting must be a boolean")
        self._grid_splitting = value

    @property
    def drfrac_max(self):
        return self._drfrac_max

    @drfrac_max.setter
    def drfrac_max(self, value):
        self._validate_positive(value, "drfrac_max")
        if self._drfrac_min is not None and value <= self._drfrac_min:
            raise ValueError("Require drfrac_min < drfrac_max")
        self._drfrac_max = float(value)

    @property
    def drfrac_min(self):
        return self._drfrac_min

    @drfrac_min.setter
    def drfrac_min(self, value):
        self._validate_positive(value, "drfrac_min")
        if self._drfrac_max is not None and value >= self._drfrac_max:
            raise ValueError("Require drfrac_min < drfrac_max")
        self._drfrac_min = float(value)

    def _validate_positive(self, value, name):
        if isinstance(value, bool) or not math.isfinite(value) or not (value > 0):
            raise ValueError(f"{name} must be a positive float.")

    def __repr__(self):
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        ]
        attr_strs = [f"{attr}={getattr(self, attr)!r}" for attr in attrs]
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
