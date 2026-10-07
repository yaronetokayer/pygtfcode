import math

class SimParams:
    """
    Simulation control parameters.

    Attributes
    ----------
    sigma_m_0 : float
        Physical low-velocity isotropic-equivalent cross section per mass in cm^2/g. Must be positive.
        At low velocities the cross section per unit mass reduces to this value.
    w : float
        Physical velocity scale in km/s. Positive infinity (the default)
        selects exact velocity-independent scattering. Finite positive values
        set the scale of the velocity-dependent suppression.
    smfp_order : int
        Chapman-Enskog order for the SMFP transport factor.
        See Outmezguine et al. (2023), Appendix B.
        Options are 1 and 2. Defaults to 2. The second-order factor is
        normalized by 45/44 so b retains its constant-scattering meaning.
    alph : float
        Coefficient for interpolation scheme between lmfp and smfp regimes.
        kappa = ( kappa_smfp^-alph + kappa_lmfp^-alph )^(-1/alph). Must be positive.
    t_halt : float
        Absolute simulation halt time in t_s units. Positive infinity disables it.
    rho_c_halt : float
        Innermost-cell density rho[0]/rho_s at which to halt after t > 50 t_s.
        Positive infinity disables this limit.
    implicit_conduct : bool
        Use implicit conduction. False selects the deprecated constant-scattering
        explicit path, which has not been migrated to velocity dependence.
    a : float
        Model parameter 'a'. Must be positive.
    b : float
        Model parameter 'b'. Sets normalization of SMFP conduction scaling. Value from kinetic theory 25*sqrt(pi)/32.
        With normalized second order, b remains the constant-limit
        conductivity coefficient; the raw second-order 45/44 enhancement
        is not additionally applied. Must be positive.
    c : float
        Model parameter 'c'. Must be positive.
    """
    def __init__(
            self, 
            sigma_m_0             : float = 10.0,
            w                   : float = float("inf"),
            smfp_order           : int = 2,
            alph                : float = 1.0,
            t_halt              : float = 1e3,
            rho_c_halt          : float = 1500,
            implicit_conduct    : bool = True,
            a                   : float = 2.256758,
            b                   : float = 1.3847,
            c                   : float = 0.75
    ):
        self._sigma_m_0 = None
        self._w = None
        self._smfp_order = None
        self._alph = None
        self._t_halt = None
        self._rho_c_halt = None
        self._implicit_conduct = None
        self._a = None
        self._b = None
        self._c = None

        self.sigma_m_0 = sigma_m_0
        self.w = w
        self.smfp_order = smfp_order
        self.alph = alph
        self.t_halt = t_halt
        self.rho_c_halt = rho_c_halt
        self.implicit_conduct = implicit_conduct
        self.a = a
        self.b = b
        self.c = c

    @property
    def sigma_m_0(self):
        return self._sigma_m_0

    @sigma_m_0.setter
    def sigma_m_0(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("sigma_m_0 must be positive")
        self._sigma_m_0 = float(value)

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        if isinstance(value, bool) or math.isnan(value) or value <= 0:
            raise ValueError("w must be positive")
        self._w = float(value)

    @property
    def smfp_order(self):
        return self._smfp_order

    @smfp_order.setter
    def smfp_order(self, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("smfp_order must be an integer")
        if value not in [1, 2]:
            raise ValueError("smfp_order must be either 1 or 2")
        self._smfp_order = value

    @property
    def alph(self):
        return self._alph

    @alph.setter
    def alph(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("alph must be positive")
        self._alph = float(value)

    @property
    def t_halt(self):
        return self._t_halt

    @t_halt.setter
    def t_halt(self, value):
        if isinstance(value, bool) or math.isnan(value) or value <= 0:
            raise ValueError("t_halt must be positive")
        self._t_halt = float(value)

    @property
    def rho_c_halt(self):
        return self._rho_c_halt

    @rho_c_halt.setter
    def rho_c_halt(self, value):
        if isinstance(value, bool) or math.isnan(value) or value <= 0:
            raise ValueError("rho_c_halt must be positive")
        self._rho_c_halt = float(value)

    @property
    def implicit_conduct(self):
        return self._implicit_conduct
    
    @implicit_conduct.setter
    def implicit_conduct(self, value):
        if not isinstance(value, bool):
            raise ValueError("implicit_conduct must be a boolean")
        self._implicit_conduct = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("a must be positive")
        self._a = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("b must be positive")
        self._b = float(value)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("c must be positive")
        self._c = float(value)

    def __repr__(self):
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        ]
        attr_strs = []
        for attr in attrs:
            value = getattr(self, attr)
            attr_strs.append(f"{attr}={repr(value)}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
