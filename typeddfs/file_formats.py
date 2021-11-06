"""
File formats for reading/writing to/from DFs.
"""
from __future__ import annotations

import enum
from collections import defaultdict
from pathlib import Path
from typing import Mapping, MutableMapping, NamedTuple, Optional, Set, Union

from typeddfs.df_errors import FilenameSuffixError
from typeddfs.utils._format_support import DfFormatSupport
from typeddfs.utils._utils import PathLike


class BaseFormatCompression(NamedTuple):
    base: Path
    format: Optional[FileFormat]
    compression: CompressionFormat


class BaseCompression(NamedTuple):
    base: Path
    compression: CompressionFormat


class _Enum(enum.Enum):
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class CompressionFormat(_Enum):
    """
    A compression scheme or no compression: gzip, zip, bz2, xz, and none.
    These are the formats supported by Pandas for read and write.
    Provides a few useful functions for calling code.

    Examples:
        - ``CompressionFormat.strip("my_file.csv.gz")  # Path("my_file.csv")``
        - ``CompressionFormat.from_path("myfile.csv")  # CompressionFormat.none``
    """

    gz = ()
    zip = ()
    bz2 = ()
    xz = ()
    none = ()

    @classmethod
    def list(cls) -> Set[CompressionFormat]:
        """
        Returns the set of CompressionFormats.
        Works with static type analysis.
        """
        return set(cls)

    @classmethod
    def list_non_empty(cls) -> Set[CompressionFormat]:
        """
        Returns the set of CompressionFormats, except for ``none``.
        Works with static type analysis.
        """
        return {c for c in cls if c is not cls.none}

    @classmethod
    def of(cls, t: Union[str, CompressionFormat]) -> CompressionFormat:
        """
        Returns a FileFormat from a name (e.g. "gz" or "gzip").
        Case-insensitive.

        Example:
            ``CompressionFormat.of("gzip").suffix  # ".gz"``
        """
        if isinstance(t, CompressionFormat):
            return t
        try:
            return CompressionFormat[str(t).strip().lower()]
        except KeyError:
            for f in CompressionFormat.list():
                if t == f.full_name:
                    return f
            raise  # pragma: no cover

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
        for c in CompressionFormat.list_non_empty():
            if path.name.endswith(c.suffix):
                return path.parent / path.name[: -len(c.suffix)]
        return path

    @classmethod
    def split(cls, path: PathLike) -> BaseCompression:
        path = str(path)
        for c in CompressionFormat.list_non_empty():
            if path.endswith(c.suffix):
                return BaseCompression(Path(path[: -len(c.suffix)]), c)
        return BaseCompression(Path(path), CompressionFormat.none)

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
    .text("properties", ".properties")
    .text("toml", ".toml")
    .text("ini", ".ini")
    .text("lines", ".lines")
    .text("lines", ".txt")
    .text("lines", ".list")
    .text("flexwf", ".flexwf")
    .other("fwf", ".fwf")
    .other("pickle", ".pkl")
    .other("pickle", ".pickle")
    .other("xlsx", ".xlam")
    .other("xlsx", ".xlsx")
    .other("xlsx", ".xlsm")
    .other("xlsx", ".xltm")
    .other("xlsx", ".xltx")
    .other("xls", ".xla")
    .other("xls", ".xlam")
    .other("xls", ".xlm")
    .other("xls", ".xls")
    .other("xls", ".xlt")
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


class FileFormat(_Enum):
    """
    A computer-readable format for reading **and** writing of DataFrames in typeddfs.
    This includes CSV, Parquet, ODT, etc. Some formats also include compressed variants.
    E.g. a ".csg.gz" will map to ``FileFormat.csv``.
    This is used internally by :meth:`typeddfs.abs_df.read_file`
    and :meth:`typeddfs.abs_df.write_file`, but it may be useful to calling code directly.

    Examples:
        - ``FileFormat.from_path("my_file.csv.gz").is_text()   # True``
        - ``FileFormat.from_path("my_file.csv.gz").can_read()  # always True``
        - ``FileFormat.from_path("my_file.xlsx").can_read()    # true if required package is installed``
    """

    csv = ()
    tsv = ()
    json = ()
    xml = ()
    toml = ()
    ini = ()
    properties = ()
    lines = ()
    fwf = ()
    flexwf = ()
    feather = ()
    hdf = ()
    ods = ()
    parquet = ()
    pickle = ()
    xls = ()
    xlsb = ()
    xlsx = ()

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
            FileFormat.properties,
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
    def is_binary(self) -> bool:
        """
        Returns whether this format is text-encoded.
        Note that this does *not* consider whether the file is compressed.
        """
        return not self.is_text

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
            FileFormat.ini,
            FileFormat.toml,
            FileFormat.properties,
            FileFormat.lines,
            FileFormat.fwf,
            FileFormat.flexwf,
        }

    @classmethod
    def of(cls, t: Union[str, FileFormat]) -> FileFormat:
        """
        Returns a FileFormat from an exact name (e.g. "csv").

        See Also:
            :meth:`from_suffix`
            :meth:`from_path`
        """
        if isinstance(t, FileFormat):
            return t
        return FileFormat[str(t).strip().lower()]

    @classmethod
    def strip(
        cls, path: PathLike, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> Path:
        """
        Strips a recognized, optionally compressed, suffix from ``path``.

        See Also:
            :meth:`split`

        Example:
            .. code-block::

                FileFormat.strip("abc/xyz.csv.gz")  # Path("abc") / "xyz"
        """
        base, _, _ = cls.split(path, format_map=format_map)
        return base

    @classmethod
    def split(
        cls, path: PathLike, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> BaseFormatCompression:
        """
        Splits a path into the base path, format, and compression.

        See Also:
            :meth:`split_or_none`
            :meth:`strip`
            :meth:`from_path`

        Raises:
            FilenameSuffixError: If the suffix is not found

        Returns:
            A 3-tuple of (base base excluding suffixes, file format, compression format)
        """
        p, fmt, comp = cls.split_or_none(path, format_map=format_map)
        if fmt is None:
            raise FilenameSuffixError(f"Suffix for {path} not recognized")
        return BaseFormatCompression(p, fmt, comp)

    @classmethod
    def split_or_none(
        cls, path: PathLike, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> BaseFormatCompression:
        """
        Splits a path into the base path, format, and compression.

        See Also:
            :meth:`split`
            :meth:`strip`
            :meth:`from_path`

        Returns:
            A 3-tuple of (base base excluding suffixes, file format, compression format)
        """
        if format_map is None:
            format_map = _rev_valid_formats
        format_map = {k: FileFormat.of(v) for k, v in format_map.items()}
        path, comp = CompressionFormat.split(path)
        if not isinstance(comp, CompressionFormat):
            raise TypeError(f"{comp} is {type(comp)}")
        path = str(path)
        fmt = None
        for f0, f1 in format_map.items():
            if path.endswith(f0):
                path = path[: -len(f0)]
                fmt = f1
        return BaseFormatCompression(Path(path), fmt, comp)

    @classmethod
    def from_path_or_none(
        cls, path: PathLike, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> Optional[FileFormat]:
        """
        Same as :meth:`from_path`, but returns None if not found.
        """
        try:
            return cls.from_path(path, format_map=format_map)
        except FilenameSuffixError:
            return None

    @classmethod
    def from_path(
        cls, path: PathLike, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> FileFormat:
        """
        Guesses a FileFormat from a filename.

        See Also:
            :meth:`from_suffix`

        Args:
            path: A string or :class:`pathlib.Path` to a file.
            format_map: A mapping from suffixes to formats;
                        if ``None``, uses :meth:`suffix_map`.

        Raises:
            typeddfs.df_errors.FilenameSuffixError: If not found
        """
        _, fmt, _ = cls.split(path, format_map=format_map)
        return fmt

    @classmethod
    def from_suffix_or_none(
        cls, suffix: str, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> Optional[FileFormat]:
        """
        Same as :meth:`from_suffix`, but returns None if not found.
        """
        try:
            return cls.from_suffix(suffix, format_map=format_map)
        except FilenameSuffixError:
            return None

    @classmethod
    def from_suffix(
        cls, suffix: str, *, format_map: Optional[Mapping[str, Union[FileFormat, str]]] = None
    ) -> FileFormat:
        """
        Returns the FileFormat corresponding to a filename suffix.

        See Also:
            :meth:`from_path`

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
            raise FilenameSuffixError(msg, key=suffix) from None

    @classmethod
    def all_readable(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be read on this system.
        Note that the result may depend on whether supporting packages are installed.
        Includes insecure and discouraged formats.
        """
        return {f for f in cls if f.can_read}

    @classmethod
    def all_writable(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be written to on this system.
        Note that the result may depend on whether supporting packages are installed.
        Includes insecure and discouraged formats.
        """
        return {f for f in cls if f.can_write}

    @classmethod
    def suffix_map(cls) -> MutableMapping[str, FileFormat]:
        """
        Returns a mapping from all suffixes to their respective formats.
        See :meth:`suffixes`.
        """
        return {k: v for k, v in _rev_valid_formats.items()}

    def compressed_variants(self, suffix: str) -> Set[str]:
        """
        Returns all allowed suffixes.

        Example:
            .. code-block::

            FileFormat.json.compressed_variants(".json")
            # {".json", ".json.gz", ".json.zip", ...}
        """
        # Pandas's fwf currently does not support compression
        if self.is_text and self is not FileFormat.fwf:
            return {suffix + c for c in CompressionFormat.all_suffixes()}
        else:
            return {suffix}

    def matches(self, *, supported: bool, secure: bool, recommended: bool) -> bool:
        """
        Returns whether this format meets some requirements.

        Args:
            supported: :attr:`can_read` and :attr:`can_write` are True
            secure: :attr:`is_secure` is True
            recommended: :attr:`is_recommended` is True
        """
        return (
            (not secure or self.is_secure)
            and (not recommended or self.is_recommended)
            and (not supported or self.can_read and self.can_write)
        )

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
    def is_recommended(self) -> bool:
        """
        Returns whether the format is good.
        Includes CSV, TSV, Parquet, etc.
        Excludes all insecure formats along with fixed-width, INI, properties, TOML, and HDF5.
        """
        return self not in {
            FileFormat.fwf,
            FileFormat.xls,
            FileFormat.xlsb,
            FileFormat.hdf,
            FileFormat.ini,
            FileFormat.toml,
            FileFormat.properties,
        }

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
        macros = {".xlsm", ".xltm", ".xls", ".xlm", ".xlam", ".xla"}
        return self is not FileFormat.pickle and not any([s in macros for s in self.suffixes])

    @property
    def can_always_read(self) -> bool:  # pragma: no cover
        """
        Returns whether this format can be read as long as typeddfs is installed.
        In other words, regardless of any optional packages.
        """
        return self.name not in DfFormatSupport.support_map

    @property
    def can_always_write(self) -> bool:  # pragma: no cover
        """
        Returns whether this format can be written to as long as typeddfs is installed.
        In other words, regardless of any optional packages.
        """
        return self.name not in DfFormatSupport.support_map

    @property
    def can_read(self) -> bool:
        """
        Returns whether this format can be read.
        Note that the result may depend on whether supporting packages are installed.
        """
        return DfFormatSupport.support_map.get(self.name, True)

    @property
    def can_write(self) -> bool:
        """
        Returns whether this format can be written.
        Note that the result may depend on whether supporting packages are installed.
        """
        return DfFormatSupport.support_map.get(self.name, True)


__all__ = [
    "FileFormat",
    "CompressionFormat",
    "DfFormatSupport",
    "BaseCompression",
    "BaseFormatCompression",
]
