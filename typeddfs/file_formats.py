"""
File formats for reading/writing to/from DFs.
"""
from __future__ import annotations
import enum
from collections import defaultdict
from typing import Set, Union
from typing import Mapping
from warnings import warn

from typeddfs._utils import PathLike
from typeddfs.df_errors import FilenameSuffixError

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
    import tabulate

    try:
        import wcwidth  # noqa: F401
    except ImportError:
        warn("wcwidth is not installed")
except ImportError:  # pragma: no cover
    tabulate = None


class _SuffixMap:
    def __init__(self):
        self._map = defaultdict(set)

    def text(self, name: str, suffix: str):
        self._map[name].update({suffix + c for c in {".gz", ".zip", ".bz2", ".xz", ""}})
        return self

    def other(self, name: str, suffix: str):
        self._map[name].add(suffix)
        return self

    def done(self):
        return self._map

    def inverse(self) -> Mapping[str, str]:
        dct = {}
        for name, suffixes in self._map.items():
            for suffix in suffixes:
                dct[suffix] = name
        return dct


_format_map = (
    _SuffixMap()
    .text("csv", ".csv")
    .text("tsv", ".tsv")
    .text("tsv", ".tab")
    .text("json", ".json")
    .text("lines", ".lines")
    .text("lines", ".txt")
    .text("lines", ".list")
    .text("flexwf", ".flexwf")
    .other("fwf", ".fwf")
    .other("excel", ".xls")
    .other("excel", ".xlsx")
    .other("feather", ".feather")
    .other("parquet", ".snappy")
    .other("parquet", ".parquet")
    .other("hdf", ".h5")
    .other("hdf", ".hdf5")
    .other("hdf", ".hdf")
)
_valid_formats = _format_map.done()
_rev_valid_formats = _format_map.inverse()


class DfFormatSupport:
    has_feather = pyarrow is not None
    has_parquet = fastparquet is not None or pyarrow is not None
    has_hdf5 = tables is not None
    has_tabulate = tabulate is not None


_support_map = {
    "parquet": DfFormatSupport.has_parquet,
    "feather": DfFormatSupport.has_feather,
    "hdf": DfFormatSupport.has_hdf5,
    "flexwf": DfFormatSupport.has_tabulate,
}


class FileFormat(enum.Enum):
    """ """

    csv = enum.auto()
    tsv = enum.auto()
    json = enum.auto()
    lines = enum.auto()
    fwf = enum.auto()
    flexwf = enum.auto()
    feather = enum.auto()
    parquet = enum.auto()
    hdf = enum.auto()
    excel = enum.auto()

    @classmethod
    def of(cls, t: Union[str, FileFormat]) -> FileFormat:
        if isinstance(t, FileFormat):
            return t
        return FileFormat[str(t).strip().lower()]

    @classmethod
    def from_path(cls, path: PathLike) -> FileFormat:
        for suffix, fmt in _rev_valid_formats.items():
            if str(path).endswith(suffix):
                return FileFormat[fmt]
        raise FilenameSuffixError(f"Suffix for {path} not recognized. Is an extra package needed?")

    @classmethod
    def all_readable(cls) -> Set[FileFormat]:
        return {f for f in cls if f.can_read}

    @classmethod
    def all_writable(cls) -> Set[FileFormat]:
        return {f for f in cls if f.can_write}

    @property
    def suffixes(self) -> Set[str]:
        return _valid_formats[self.name]

    @property
    def can_read(self) -> bool:
        return _support_map.get(self.name, True)

    @property
    def can_write(self) -> bool:
        return _support_map.get(self.name, True)

    @classmethod
    def from_suffix(cls, suffix: str) -> FileFormat:
        try:
            return FileFormat[_rev_valid_formats[suffix]]
        except KeyError:
            msg = f"Suffix {suffix} not recognized. Is an extra package needed?"
            raise FilenameSuffixError(msg) from None


__all__ = ["FileFormat", "DfFormatSupport"]
