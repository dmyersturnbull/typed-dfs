"""
File formats for reading/writing to/from DFs.
"""
from __future__ import annotations

import enum
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Set, Union

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
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None

try:
    import pyxlsb
except ImportError:  # pragma: no cover
    pyxlsb = None


compression_suffixes = {".gz", ".zip", ".bz2", ".xz", ""}


class _SuffixMap:
    def __init__(self):
        self._map = defaultdict(set)

    def text(self, name: str, suffix: str):
        self._map[name].update({suffix + c for c in compression_suffixes})
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
    .text("xml", ".xml")
    .text("lines", ".lines")
    .text("lines", ".txt")
    .text("lines", ".list")
    .text("flexwf", ".flexwf")
    .other("fwf", ".fwf")
    .other("pickle", ".pkl")
    .other("pickle", ".pickle")
    .other("xlsx", ".xlsx")
    .other("xls", ".xls")
    .other("xlsb", ".xlsb")
    .other("ods", ".odf")
    .other("ods", ".ods")
    .other("ods", ".odt")
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
    has_parquet = pyarrow is not None or fastparquet is not None
    has_hdf5 = tables is not None
    has_xlsx = openpyxl is not None
    has_xls = openpyxl is not None
    has_ods = openpyxl is not None
    has_xlsb = pyxlsb is not None


_support_map = {
    "parquet": DfFormatSupport.has_parquet,
    "feather": DfFormatSupport.has_feather,
    "hdf": DfFormatSupport.has_hdf5,
    "xlsx": DfFormatSupport.has_xlsx,
    "xls": DfFormatSupport.has_xls,
    "xlsb": DfFormatSupport.has_xlsb,
    "ods": DfFormatSupport.has_ods,
}


class FileFormat(enum.Enum):
    """ """

    csv = enum.auto()
    tsv = enum.auto()
    json = enum.auto()
    xml = enum.auto()
    lines = enum.auto()
    fwf = enum.auto()
    flexwf = enum.auto()
    feather = enum.auto()
    hdf = enum.auto()
    ods = enum.auto()
    parquet = enum.auto()
    pickle = enum.auto()
    xls = enum.auto()
    xlsb = enum.auto()
    xlsx = enum.auto()

    @property
    def supports_encoding(self) -> bool:
        return self in {
            FileFormat.csv,
            FileFormat.tsv,
            FileFormat.json,
            FileFormat.xml,
            FileFormat.lines,
            FileFormat.fwf,
            FileFormat.flexwf,
            FileFormat.xlsx,
            FileFormat.ods,
            FileFormat.xls,
            FileFormat.xlsb,
        }

    @property
    def is_text(self) -> bool:
        return self in {
            FileFormat.csv,
            FileFormat.tsv,
            FileFormat.json,
            FileFormat.xml,
            FileFormat.lines,
            FileFormat.fwf,
            FileFormat.flexwf,
        }

    @classmethod
    def of(cls, t: Union[str, FileFormat]) -> FileFormat:
        if isinstance(t, FileFormat):
            return t
        return FileFormat[str(t).strip().lower()]

    @classmethod
    def from_path(
        cls, path: PathLike, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> FileFormat:
        if format_map is None:
            format_map = _rev_valid_formats
        path = str(path)
        for c in compression_suffixes:
            path = path.replace(c, "")
        path = Path(path)
        return cls.from_suffix(path.suffix, format_map=format_map)

    @classmethod
    def from_suffix(
        cls, suffix: str, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> FileFormat:
        if format_map is None:
            format_map = _rev_valid_formats
        try:
            return FileFormat.of(format_map[suffix])
        except KeyError:
            msg = f"Suffix {suffix} not recognized. Is an extra package needed?"
            raise FilenameSuffixError(msg) from None

    @classmethod
    def all_readable(cls) -> Set[FileFormat]:
        return {f for f in cls if f.can_read}

    @classmethod
    def all_writable(cls) -> Set[FileFormat]:
        return {f for f in cls if f.can_write}

    @classmethod
    def suffix_map(cls) -> Dict[str, FileFormat]:
        return {k: v for k, v in _rev_valid_formats.items()}

    def compressed_variants(self, suffix: str) -> Set[str]:
        # Pandas's fwf currently does not support compression
        if self.is_text and self is not FileFormat.fwf:
            return {suffix + c for c in compression_suffixes}
        else:
            return {suffix}

    @property
    def suffixes(self) -> Set[str]:
        return _valid_formats[self.name]

    @property
    def can_read(self) -> bool:
        return _support_map.get(self.name, True)

    @property
    def can_write(self) -> bool:
        return _support_map.get(self.name, True)


__all__ = ["FileFormat", "DfFormatSupport"]
