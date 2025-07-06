class SimParams:
    """
    Placeholder for simulation control parameters.

    This will eventually include:
    - Maximum central density
    - Time step settings
    - Output frequency
    - Flags for physics modules (e.g. evaporation)
    """
    def __init__(
            self, 
            sigma_m=10.0,
            a=2.256758,
            b=1.38,
            c=0.75
    ):
        self._sigma_m = None
        self._a = None
        self._b = None
        self._c = None

        self.sigma_m = sigma_m # self-interaction cross-section in cm^2/g
        self.a = a
        self.b = b
        self.c = c

    @property
    def sigma_m(self):
        return self._sigma_m

    @sigma_m.setter
    def sigma_m(self, value):
        if value <= 0:
            raise ValueError("sigma_m must be positive")
        self._sigma_m = float(value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value <= 0:
            raise ValueError("a must be positive")
        self._a = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value <= 0:
            raise ValueError("b must be positive")
        self._b = float(value)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        if value <= 0:
            raise ValueError("c must be positive")
        self._c = float(value)

    def __repr__(self):
        return (f"SimParams(sigma_m={self.sigma_m}, a={self.a}, "
                f"b={self.b}, c={self.c})")
