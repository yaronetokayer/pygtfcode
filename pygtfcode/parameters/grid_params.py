class GridParams:
    """
    Parameters defining the radial grid for the simulation.

    Attributes
    ----------
    rmin : float
        Minimum radius of the grid, in units of the scale radius (r / r_s).
    rmax : float
        Maximum radius of the grid, in units of the scale radius (r / r_s).
    ngrid : int
        Number of radial grid points (must be > 1).
    """

    def __init__(
        self, 
        rmin: float = 1e-2,
        rmax: float = 2e2,
        ngrid: int = 200,
        grid_splitting : bool = True,
        drfrac_max : float = 1e-1,
        drfrac_min : float = 1e-2,
    ):
        self._rmin = None
        self._rmax = None
        self._ngrid = None
        self._grid_splitting = None
        self._drfrac_max = None
        self._drfrac_min = None

        self.rmin = rmin
        self.rmax = rmax
        self.ngrid = ngrid
        self.grid_splitting = grid_splitting
        self.drfrac_max = drfrac_max
        self.drfrac_min = drfrac_min

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        if value <= 0 or (self._rmax is not None and value >= self._rmax):
            raise ValueError("Require 0 < rmin < rmax")
        self._rmin = float(value)

    @property
    def rmax(self):
        return self._rmax

    @rmax.setter
    def rmax(self, value):
        if value <= 0 or (self._rmin is not None and value <= self._rmin):
            raise ValueError("Require 0 < rmin < rmax")
        self._rmax = float(value)

    @property
    def ngrid(self):
        return self._ngrid

    @ngrid.setter
    def ngrid(self, value):
        if not isinstance(value, int) or value <= 1:
            raise ValueError("ngrid must be an integer greater than 1")
        self._ngrid = value

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
        self._drfrac_max = float(value)

    @property
    def drfrac_min(self):
        return self._drfrac_min

    @drfrac_min.setter
    def drfrac_min(self, value):
        self._validate_positive(value, "drfrac_min")
        self._drfrac_min = float(value)

    def _validate_positive(self, value, name):
        if not (value > 0):
            raise ValueError(f"{name} must be a positive float.")

    def __repr__(self):
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        ]
        attr_strs = [f"{attr}={getattr(self, attr)!r}" for attr in attrs]
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"

