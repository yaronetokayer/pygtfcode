class PrecisionParams:
    """
    Parameters controlling numerical precision and convergence behavior.

    Attributes
    ----------
    eps_du : float
        Maximum allowed relative change in internal energy (u) per time step.
    eps_dr : float
        Maximum allowed relative change in radius per time step.
    eps_dt : float
        Epsilon factor for adjusting time step size.
    max_iter_du : int
        Maximum iterations allowed for conduction step convergence.
    max_iter_revir : int
        Maximum iterations allowed for re-virialization step convergence.
    """

    def __init__(
        self,
        eps_du: float = 1e-2,
        eps_dr: float = 1e-2,
        eps_dt: float = 1e-2,
        max_iter_du: int = 10,
        max_iter_revir: int = 50,
    ):
        self._eps_du = None
        self._eps_dr = None
        self._eps_dt = None
        self._max_iter_du = None
        self._max_iter_revir = None

        self.eps_du = eps_du
        self.eps_dr = eps_dr
        self.eps_dt = eps_dt
        self.max_iter_du = max_iter_du
        self.max_iter_revir = max_iter_revir

    @property
    def eps_du(self):
        return self._eps_du

    @eps_du.setter
    def eps_du(self, value):
        self._validate_positive(value, "eps_du")
        self._eps_du = float(value)

    @property
    def eps_dr(self):
        return self._eps_dr

    @eps_dr.setter
    def eps_dr(self, value):
        self._validate_positive(value, "eps_dr")
        self._eps_dr = float(value)

    @property
    def eps_dt(self):
        return self._eps_dt

    @eps_dt.setter
    def eps_dt(self, value):
        self._validate_positive(value, "eps_dt")
        self._eps_dt = float(value)

    @property
    def max_iter_du(self):
        return self._max_iter_du

    @max_iter_du.setter
    def max_iter_du(self, value):
        self._validate_nonnegative_int(value, "max_iter_du")
        self._max_iter_du = int(value)

    @property
    def max_iter_revir(self):
        return self._max_iter_revir

    @max_iter_revir.setter
    def max_iter_revir(self, value):
        self._validate_nonnegative_int(value, "max_iter_revir")
        self._max_iter_revir = int(value)

    def _validate_positive(self, value, name):
        if not (value > 0):
            raise ValueError(f"{name} must be a positive float.")

    def _validate_nonnegative_int(self, value, name):
        if not (isinstance(value, int) and value >= 0):
            raise ValueError(f"{name} must be a non-negative integer.")

    def __repr__(self):
        return (
            f"PrecisionParams(eps_du={self.eps_du}, eps_dr={self.eps_dr}, "
            f"eps_dt={self.eps_dt}, max_iter_du={self.max_iter_du}, "
            f"max_iter_revir={self.max_iter_revir})"
        )
