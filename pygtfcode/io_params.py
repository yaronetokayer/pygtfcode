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
    overwrite : bool
        Whether to overwrite existing output files.
    """

    def __init__(self, model_no: int = 1, base_dir: str = None, overwrite: bool = False):
        self._model_no = None
        self._base_dir = None
        self._overwrite = None

        self.model_no = model_no
        self.base_dir = base_dir or os.getcwd()
        self.overwrite = overwrite

    @property
    def model_no(self):
        return self._model_no

    @model_no.setter
    def model_no(self, value):
        if not (0 <= value < 1000):
            raise ValueError("model_no must be between 0 and 999 (inclusive)")
        self._model_no = int(value)

    @property
    def model_dir(self):
        return f"Model{self.model_no:03d}"

    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value):
        if not isinstance(value, str):
            raise TypeError("base_dir must be a string path")
        self._base_dir = value

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value):
        if not isinstance(value, bool):
            raise TypeError("overwrite must be a boolean")
        self._overwrite = value

    def __repr__(self):
        return (
            f"IOParams(model_no={self.model_no}, "
            f"base_dir='{self.base_dir}', "
            f"model_dir='{self.model_dir}', "
            f"overwrite={self.overwrite})"
        )
