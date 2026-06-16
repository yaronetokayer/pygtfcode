class CosmoParams:
    """
    Cosmological parameters
    """

    def __init__(
        self,
        xhubble     : float = 0.7,
        Omega_m     : float = 0.3,
        Delta_vir   : float = 200.0,
        z           : float = 0.0,
    ):
        self._xhubble   = None
        self._Omega_m   = None
        self._Delta_vir = None
        self._z         = None

        self.xhubble    = xhubble
        self.Omega_m    = Omega_m
        self.Delta_vir  = Delta_vir
        self.z          = z

    @property
    def xhubble(self):
        return self._xhubble

    @xhubble.setter
    def xhubble(self, value):
        if value <= 0:
            raise ValueError("xhubble must be positive")
        self._xhubble = float(value)

    @property
    def Omega_m(self):
        return self._Omega_m

    @Omega_m.setter
    def Omega_m(self, value):
        if not 0 < value <= 1:
            raise ValueError("Omega_m must be between 0 and 1")
        self._Omega_m = float(value)

    @property
    def Delta_vir(self):
        return self._Delta_vir

    @Delta_vir.setter
    def Delta_vir(self, value):
        if value <= 0:
            raise ValueError("Delta_vir must be positive")
        self._Delta_vir = float(value)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("z must be non-negative")
        self._z = float(value)

    def xH(self):
        """
        Computes H(z) in units of km/s/Mpc using cosmological parameters

        Returns
        -------
        H_z : float
            Hubble parameter at redshift z [km/s/Mpc].
        """
        import math

        Omega_m = float(self.Omega_m)
        omega_lambda = 1 - Omega_m
        xH_0 = 100 * float(self.xhubble)  # H_0 in km/s/Mpc

        z = float(self.z)

        fac = omega_lambda + (1.0 - omega_lambda - Omega_m) * (1.0 + z)**2 + Omega_m * (1.0 + z)**3

        H = xH_0 * math.sqrt(fac)

        return float(H)

    def __repr__(self):
        attrs = [
            "xhubble",
            "Omega_m",
            "Delta_vir",
            "z",
        ]
        attr_strs = [f"{attr}={getattr(self, attr)!r}" for attr in attrs]
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
