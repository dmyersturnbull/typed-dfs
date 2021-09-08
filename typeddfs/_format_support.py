from __future__ import annotations

from typing import Mapping

pyarrow = None
fastparquet = None
tables = None
openpyxl = None
pyxlsb = None


def _import():
    global pyarrow, fastparquet, tables, openpyxl, pyxlsb
    try:
        import pyarrow
    except ImportError:  # pragma: no cover
        pyarrow = None

    try:
        import fastparquet
    except ImportError:  # pragma: no cover
        fastparquet = None

    try:
        import tables
    except ImportError:  # pragma: no cover
        tables = None

    try:
        import openpyxl
    except ImportError:  # pragma: no cover
        openpyxl = None

    try:
        import pyxlsb
    except ImportError:  # pragma: no cover
        pyxlsb = None


class _DfFormatSupport:
    """
    Records the presence of required packages.
    Records whether file formats are supported as per whether a required
    package is available/installed.
    This is used by :py.class:`FileFormat`
    and thereby :py.meth:`typeddfs.abs_df.read_file`
    and :py.meth:`typeddfs.abs_df.write_file`.

    Example:
        if not DfFormatSupport.has_hdf5:
            print("No HDF5")
    """

    def __init__(self):
        _import()
        self._has_feather = pyarrow is not None
        self._has_parquet = pyarrow is not None or fastparquet is not None
        self._has_hdf5 = tables is not None
        self._has_xlsx = openpyxl is not None
        self._has_xls = openpyxl is not None
        self._has_ods = openpyxl is not None
        self._has_xlsb = pyxlsb is not None

    @property
    def has_feather(self) -> bool:
        return self._has_feather

    @property
    def has_parquet(self) -> bool:
        return self._has_parquet

    @property
    def has_hdf5(self) -> bool:
        return self._has_hdf5

    @property
    def has_xlsx(self) -> bool:
        return self._has_xlsx

    @property
    def has_xls(self) -> bool:
        return self._has_xls

    @property
    def has_ods(self) -> bool:
        return self._has_ods

    @property
    def has_xlsb(self) -> bool:
        return self._has_xlsb

    @classmethod
    def reload(cls) -> None:
        """
        Retry importing the packages.
        Some supported formats may appear while others may disappear.
        This is a global operation.
        """
        _import()

    @property
    def support_map(self) -> Mapping[str, bool]:
        """
        Returns the optional formats and whether they are supported.
        """
        return {
            attr.replace("has_", ""): getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("has_")
        }


DfFormatSupport = _DfFormatSupport()


__all__ = ["DfFormatSupport"]
