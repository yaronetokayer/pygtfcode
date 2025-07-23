import os

class IOParams:
    """
    Stores parameters related to input/output behavior.

    Attributes
    ----------
    model_no : int
        Integer identifier for the model run (must be 0 <= model_no < 1000).
    base_dir : str
        Path to the main directory where output files are written.
    model_dir : str
        Subdirectory named 'ModelXXX', where XXX is zero-padded model number.
    tlog : int
        Timesteps between logging output.
    drho_prof : float
        Change in log of central density to trigger writing profiles to disk.
    drho_tevol : float
        Change in log of central density to trigger writing time evolution data to disk.
    overwrite : bool
        Whether to overwrite existing output files.
    chatter : bool
        Whether to print status messages during execution.
    """

    def __init__(self, 
                 model_no: int = 0, 
                 base_dir: str = None, 
                 tlog: int = 100000,
                 drho_prof : float = 0.1,
                 drho_tevol : float = 0.01,
                 overwrite: bool = True,
                 chatter: bool = True,
                 ):
        self._model_no = None
        self._base_dir = None
        self._tlog = tlog
        self._drho_prof = drho_prof
        self._drho_tevol = drho_tevol
        self._overwrite = None
        self._chatter = None

        self.model_no = model_no
        self.base_dir = base_dir or os.getcwd()
        self.tlog = tlog
        self.drho_prof = drho_prof
        self.drho_tevol = drho_tevol
        self.overwrite = overwrite
        self.chatter = chatter

    @property
    def model_no(self):
        return self._model_no

    @model_no.setter
    def model_no(self, value):
        if not isinstance(value, int):
            raise TypeError("model_no must be an integer")
        if not (0 <= value < 1000):
            raise ValueError("model_no must be between 0 and 999 (inclusive)")
        self._model_no = value

    @property
    def model_dir(self):
        """Returns the subdirectory name as 'ModelXXX' (with leading zeros)."""
        return f"Model{self.model_no:03d}"
    
    @property
    def logpath(self):
        """Returns the full path to the log file."""
        return os.path.join(self.base_dir, self.model_dir, "logfile.txt")

    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value):
        if not isinstance(value, str):
            raise TypeError("base_dir must be a string")
        self._base_dir = value

    @property
    def tlog(self):
        return self._tlog

    @tlog.setter
    def tlog(self, value):
        if not isinstance(value, int):
            raise TypeError("tlog must be an integer")
        self._tlog = value

    @property
    def drho_prof(self):
        return self._drho_prof
    
    @drho_prof.setter
    def drho_prof(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("drho_prof must be a number")
        if value <= 0:
            raise ValueError("drho_prof must be positive")
        self._drho_prof = float(value)

    @property
    def drho_tevol(self):
        return self._drho_tevol
    
    @drho_tevol.setter
    def drho_tevol(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("drho_tevol must be a number")
        if value <= 0:
            raise ValueError("drho_tevol must be positive")
        self._drho_tevol = float(value)

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value):
        if not isinstance(value, bool):
            raise TypeError("overwrite must be a boolean")
        self._overwrite = value

    @property
    def chatter(self):
        return self._chatter

    @chatter.setter
    def chatter(self, value):
        if not isinstance(value, bool):
            raise TypeError("chatter must be a boolean")
        self._chatter = value

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
