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


class CompressionFormat(enum.Enum):
    """
    A compression scheme or no compression: gzip, zip, bz2, xz, and none.
    These are the formats supported by Pandas for read and write.
    Provides a few useful functions for calling code.

    Example:
        CompressionFormat.strip("my_file.csv.gz")  # Path("my_file.csv")
        CompressionFormat.from_path("myfile.csv")  # CompressionFormat.none
    """

    gz = enum.auto()
    zip = enum.auto()
    bz2 = enum.auto()
    xz = enum.auto()
    none = enum.auto()

    @classmethod
    def list(cls) -> Set[CompressionFormat]:
        """
        Returns the set of CompressionFormats.
        Works with static type analysis.
        """
        return set(cls)

    @classmethod
    def of(cls, t: Union[str, CompressionFormat]) -> CompressionFormat:
        """
        Returns a FileFormat from a name (e.g. "gz" or "gzip").
        Case-insensitive.

        Example:
            CompressionFormat.of("gzip").suffix  # ".gz"
        """
        if isinstance(t, CompressionFormat):
            return t
        try:
            return CompressionFormat[str(t).strip().lower()]
        except KeyError:
            for f in CompressionFormat.list():
                if t == f.full_name:
                    return f
            raise

    @property
    def full_name(self) -> str:
        """
        Returns a more-complete name of this format.
        For example, "gzip" "bzip2", "xz", and "none".
        """
        return {CompressionFormat.gz: "gzip", CompressionFormat.bz2: "bzip2"}.get(self, self.name)

    @property
    def is_compressed(self) -> bool:
        """
        Shorthand for ``fmt is not CompressionFormat.none``.
        """
        return self is not CompressionFormat.none

    @classmethod
    def all_suffixes(cls) -> Set[str]:
        """
        Returns all suffixes for all compression formats.
        """
        return {c.suffix for c in cls}

    @property
    def suffix(self) -> str:
        """
        Returns the single Pandas-recognized suffix for this format.
        This is just "" for CompressionFormat.none.
        """
        return "" if self is CompressionFormat.none else "." + self.name

    @classmethod
    def strip_suffix(cls, path: PathLike) -> Path:
        """
        Returns a path with any recognized compression suffix (e.g. ".gz") stripped.
        """
        path = Path(path)
        for c in CompressionFormat:
            if path.name.endswith(c.suffix):
                return path.parent / path.name[: -len(c.suffix)]
        return path

    @classmethod
    def from_path(cls, path: PathLike) -> CompressionFormat:
        """
        Returns the compression scheme from a path suffix.
        """
        path = Path(path)
        if path.name.startswith(".") and path.name.count(".") == 1:
            suffix = path.name
        else:
            suffix = path.suffix
        return cls.from_suffix(suffix)

    @classmethod
    def from_suffix(cls, suffix: str) -> CompressionFormat:
        """
        Returns the recognized compression scheme from a suffix.
        """
        for c in CompressionFormat:
            if suffix == c.suffix:
                return c
        return CompressionFormat.none


class _SuffixMap:
    def __init__(self):
        self._map = defaultdict(set)

    def text(self, name: str, suffix: str):
        self._map[name].update({suffix + c for c in CompressionFormat.all_suffixes()})
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
    """
    A computer-readable format for reading **and** writing of DataFrames in typeddfs.
    This includes CSV, Parquet, ODT, etc. Some formats also include compressed variants.
    E.g. a ".csg.gz" will map to ``FileFormat.csv``.
    This is used internally by :py.meth:`typeddfs.abs_df.read_file`
    and :py.meth:`typeddfs.abs_df.write_file`, but it may be useful to calling code directly.

    Example:
        FileFormat.from_path("my_file.csv.gz").is_text()   # True
        FileFormat.from_path("my_file.csv.gz").can_read()  # always True
        FileFormat.from_path("my_file.xlsx").can_read()    # true if required package is installed
    """

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

    @classmethod
    def list(cls) -> Set[FileFormat]:
        """
        Returns the set of FileFormats.
        Works with static type analysis.
        """
        return set(cls)

    @property
    def supports_encoding(self) -> bool:
        """
        Returns whether this format supports a text encoding of some sort.
        This may not correspond to an ``encoding=`` parameter, and the format may be binary.
        For example, XLS and XML support encodings.
        """
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
        """
        Returns whether this format is text-encoded.
        Note that this does *not* consider whether the file is compressed.
        """
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
        """
        Returns a FileFormat from an exact name (e.g. "csv").
        """
        if isinstance(t, FileFormat):
            return t
        return FileFormat[str(t).strip().lower()]

    @classmethod
    def from_path_or_none(
        cls, path: PathLike, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> Optional[FileFormat]:
        """
        Same as :py.meth:`from_path`, but returns None if not found.
        """
        try:
            return cls.from_path(path, format_map=format_map)
        except FilenameSuffixError:
            return None

    @classmethod
    def from_path(
        cls, path: PathLike, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> FileFormat:
        """
        Guesses a FileFormat from a filename.

        Args:
            path: A string or :py.class:`pathlib.Path` to a file.
            format_map: A mapping from suffixes to formats;
                        if ``None``, uses :meth:`suffix_map`.

        Raises:
            typeddfs.df_errors.FilenameSuffixError: If not found
        """
        if format_map is None:
            format_map = _rev_valid_formats
        path = str(path)
        for c in CompressionFormat.all_suffixes():
            path = path.replace(c, "")
        path = Path(path)
        return cls.from_suffix(path.suffix, format_map=format_map)

    @classmethod
    def from_suffix_or_none(
        cls, suffix: str, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> Optional[FileFormat]:
        """
        Same as :py.meth:`from_suffix`, but returns None if not found.
        """
        try:
            return cls.from_suffix(suffix, format_map=format_map)
        except FilenameSuffixError:
            return None

    @classmethod
    def from_suffix(
        cls, suffix: str, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> FileFormat:
        """
        Returns the FileFormat corresponding to a filename suffix.

        Args:
            suffix: E.g. ".csv.gz" or ".feather"
            format_map: A mapping from suffixes to formats;
                        if ``None``, uses :meth:`suffix_map`.

        Raises:
            typeddfs.df_errors.FilenameSuffixError: If not found
        """
        if format_map is None:
            format_map = _rev_valid_formats
        try:
            return FileFormat.of(format_map[suffix])
        except KeyError:
            msg = f"Suffix {suffix} not recognized. Is an extra package needed?"
            raise FilenameSuffixError(msg) from None

    @classmethod
    def all_readable(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be read on this system.
        Note that the result may depend on whether supporting packages are installed.
        """
        return {f for f in cls if f.can_read}

    @classmethod
    def all_writable(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be written to on this system.
        Note that the result may depend on whether supporting packages are installed.
        """
        return {f for f in cls if f.can_write}

    @classmethod
    def suffix_map(cls) -> Dict[str, FileFormat]:
        """
        Returns a mapping from all suffixes to their respective formats.
        See :meth:`suffixes`.
        """
        return {k: v for k, v in _rev_valid_formats.items()}

    def compressed_variants(self, suffix: str) -> Set[str]:
        """
        Returns all allowed suffixes.

        Example:
            FileFormat.json.compressed_variants(".json")
            # {".json", ".json.gz", ".json.zip", ...}
        """
        # Pandas's fwf currently does not support compression
        if self.is_text and self is not FileFormat.fwf:
            return {suffix + c for c in CompressionFormat.all_suffixes()}
        else:
            return {suffix}

    @property
    def suffixes(self) -> Set[str]:
        """
        Returns the suffixes that are tied to this format.
        These will not overlap with the suffixes for any other format.
        For example, .txt is for ``FileFormat.lines``, although it could
        be treated as tab- or space-separated.
        """
        return _valid_formats[self.name]

    @property
    def is_secure(self) -> bool:
        """
        Returns whether the format does NOT have serious security issues.
        These issues only apply to reading files, not writing.
        Excel formats that support Macros are not considered secure.
        This includes .xlsm, .xltm, and .xls. These can simply be replaced with xlsx.
        Note that .xml is treated as secure: Although some parsers are subject to
        entity expansion attacks, good ones are not.
        """
        macros = {".xlsm", ".xltm", ".xls"}
        return self is not FileFormat.pickle and not any([s in macros for s in self.suffixes])

    @property
    def can_always_read(self) -> bool:
        """
        Returns whether this format can be read as long as typeddfs is installed.
        In other words, regardless of any optional packages.
        """
        return self.name not in _support_map

    @property
    def can_always_write(self) -> bool:
        """
        Returns whether this format can be written to as long as typeddfs is installed.
        In other words, regardless of any optional packages.
        """
        return self.name not in _support_map

    @property
    def can_read(self) -> bool:
        """
        Returns whether this format can be read.
        Note that the result may depend on whether supporting packages are installed.
        """
        return _support_map.get(self.name, True)

    @property
    def can_write(self) -> bool:
        """
        Returns whether this format can be written.
        Note that the result may depend on whether supporting packages are installed.
        """
        return _support_map.get(self.name, True)


__all__ = ["FileFormat", "CompressionFormat", "DfFormatSupport"]
