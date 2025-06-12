from pygtfcode.parameters.constants import Constants
from pygtfcode.parameters.io_params import IOParams
from pygtfcode.parameters.grid_params import GridParams
from pygtfcode.parameters.init_params import InitParams, NFWParams
from pygtfcode.parameters.prec_params import PrecisionParams
from pygtfcode.parameters.sim_params import SimParams


class Config:
    """
    Central container for all static simulation parameters.

    Each attribute is an instance of a parameter class, created with
    default values unless explicitly overridden.
    """

    def __init__(
        self,
        io_params: IOParams = None,
        grid_params: GridParams = None,
        init_params: InitParams = None,
        sim_params: SimParams = None,
        prec_params: PrecisionParams = None,
        constants: type = Constants,  # note: this is a class, not an instance
    ):
        self.io = io_params or IOParams()
        self.grid = grid_params or GridParams(rmin=1e-3, rmax=1e2, Ngrid=300)
        self.init = init_params or NFWParams()
        self.sim = sim_params or SimParams()
        self.prec = prec_params or PrecisionParams()
        self.constants = constants

    def __repr__(self):
        return (
            f"Config(\n"
            f"  io={self.io},\n"
            f"  grid={self.grid},\n"
            f"  init={self.init},\n"
            f"  sim={self.sim},\n"
            f"  prec={self.prec},\n"
            f"  constants={self.constants.__name__}\n"
            f")"
        )
