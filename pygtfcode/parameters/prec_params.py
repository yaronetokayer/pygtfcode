class PrecisionParams:
    """
    Parameters controlling numerical precision and convergence behavior.

    Attributes
    ----------
    eps_du : float
        Maximum allowed relative change in internal energy (u) per time step.
    kn_threshold : float
        Threshold in Knudsen number below which the du criterion is relaxed.
    kn_width : float
        Width of the transition in Knudsen number for boosting eps_du.
    du_boost : float
        Factor by which to boost eps_du in the low-Knudsen regime.
    eps_dr : float
        Maximum allowed relative change in radius per time step.
    max_iter_du : int
        Maximum iterations allowed for re-virialization step convergence.
    max_iter_dr : int
        Maximum iterations allowed for re-virialization step convergence.
    epsabs : float
        Absolute tolerance for numerical integration routines.
    epsrel : float
        Relative tolerance for numerical integration routines.
    """

    def __init__(
        self,
        eps_du : float = 1.0e-4,
        kn_threshold : float = 0.01,
        kn_width : float = 1.0,
        du_boost : float = 100.0,
        eps_dr : float = 1.0e-12,
        max_iter_du : int = 10,
        max_iter_dr : int = 100,
        epsabs : float = 1e-6,
        epsrel : float = 1e-6
    ):
        self._eps_du = None
        self._kn_threshold = None
        self._kn_width = None
        self._du_boost = None
        self._eps_dr = None
        self._max_iter_du = None
        self._max_iter_dr = None
        self._epsabs = None
        self._epsrel = None

        self.eps_du = eps_du
        self.kn_threshold = kn_threshold
        self.kn_width = kn_width
        self.du_boost = du_boost
        self.eps_dr = eps_dr
        self.max_iter_du = max_iter_du
        self.max_iter_dr = max_iter_dr
        self.epsabs = epsabs
        self.epsrel = epsrel

    @property
    def eps_du(self):
        return self._eps_du

    @eps_du.setter
    def eps_du(self, value):
        self._validate_positive(value, "eps_du")
        self._eps_du = float(value)

    @property
    def kn_threshold(self):
        return self._kn_threshold

    @kn_threshold.setter
    def kn_threshold(self, value):
        self._validate_positive(value, "kn_threshold")
        self._kn_threshold = float(value)

    @property
    def kn_width(self):
        return self._kn_width

    @kn_width.setter
    def kn_width(self, value):
        self._validate_positive(value, "kn_width")
        self._kn_width = float(value)

    @property
    def du_boost(self):
        return self._du_boost

    @du_boost.setter
    def du_boost(self, value):
        self._validate_positive(value, "du_boost")
        self._du_boost = float(value)

    @property
    def eps_dr(self):
        return self._eps_dr

    @eps_dr.setter
    def eps_dr(self, value):
        self._validate_positive(value, "eps_dr")
        self._eps_dr = float(value)

    @property
    def max_iter_du(self):
        return self._max_iter_du

    @max_iter_du.setter
    def max_iter_du(self, value):
        self._validate_nonnegative_int(value, "max_iter_du")
        self._max_iter_du = int(value)

    @property
    def max_iter_dr(self):
        return self._max_iter_dr

    @max_iter_dr.setter
    def max_iter_dr(self, value):
        self._validate_nonnegative_int(value, "max_iter_dr")
        self._max_iter_dr = int(value)

    @property
    def epsabs(self):
        return self._epsabs

    @epsabs.setter
    def epsabs(self, value):
        self._validate_positive(value, "epsabs")
        self._epsabs = float(value)

    @property
    def epsrel(self):
        return self._epsrel

    @epsrel.setter
    def epsrel(self, value):
        self._validate_positive(value, "epsrel")
        self._epsrel = float(value)

    def _validate_positive(self, value, name):
        if not (value > 0):
            raise ValueError(f"{name} must be a positive float.")

    def _validate_nonnegative_int(self, value, name):
        if not (isinstance(value, int) and value >= 0):
            raise ValueError(f"{name} must be a non-negative integer.")

    def __repr__(self):
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        ]
        attr_strs = [f"{attr}={getattr(self, attr)!r}" for attr in attrs]
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
