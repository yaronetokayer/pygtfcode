class GridParams:
    """
    Parameters defining the radial grid for the simulation.

    Attributes
    ----------
    rmin : float
        Minimum radius of the grid, in units of the scale radius (r / r_s).
    rmax : float
        Maximum radius of the grid, in units of the scale radius (r / r_s).
    Ngrid : int
        Number of radial grid points (must be > 1).
    """

    def __init__(self, rmin: float = 1e-3, rmax: float = 1e2, Ngrid: int = 300):
        self._rmin = None
        self._rmax = None
        self._Ngrid = None

        self.rmin = rmin
        self.rmax = rmax
        self.Ngrid = Ngrid

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
    def Ngrid(self):
        return self._Ngrid

    @Ngrid.setter
    def Ngrid(self, value):
        if not isinstance(value, int) or value <= 1:
            raise ValueError("Ngrid must be an integer greater than 1")
        self._Ngrid = value

    def __repr__(self):
        return f"GridParams(rmin={self.rmin}, rmax={self.rmax}, Ngrid={self.Ngrid})"

