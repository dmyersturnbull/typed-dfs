"""
Information about how DataFrame subclasses should be handled.
"""
from __future__ import annotations

from copy import deepcopy as _copy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

import pandas as pd

from typeddfs._core_dfs import CoreDf
from typeddfs.file_formats import FileFormat
from typeddfs.utils import Utils

# noinspection PyUnresolvedReferences
from typeddfs.utils._utils import _FLEXWF_SEP, _TOML_AOT

T = TypeVar("T", bound=CoreDf, covariant=True)


def _opt_list(x):
    return [] if x is None else list(x)


def _opt_set(x):
    return set() if x is None else set(x)


def _opt_dict(x):
    return {} if x is None else dict(x)


@dataclass(frozen=True, repr=True)
class IoTyping(Generic[T]):
    _hash_alg: Optional[str] = "sha256"
    _save_hash_file: bool = False
    _save_hash_dir: bool = False
    _remap_suffixes: Optional[Mapping[str, FileFormat]] = None
    _text_encoding: str = "utf-8"
    _read_kwargs: Optional[Mapping[FileFormat, Mapping[str, Any]]] = None
    _write_kwargs: Optional[Mapping[FileFormat, Mapping[str, Any]]] = None
    _secure: bool = False
    _recommended: bool = False
    _hdf_key: str = "df"
    _attrs_suffix: str = ".attrs.json"
    _use_attrs: bool = False
    _attrs_json_kwargs: Optional[Mapping[str, Any]] = None
    _remapped_read_kwargs: Optional[Mapping[str, Any]] = None
    _remapped_write_kwargs: Optional[Mapping[str, Any]] = None
    _custom_readers: Optional[Mapping[str, Callable[[Path], pd.DataFrame]]] = None
    _custom_writers: Optional[Mapping[str, Callable[[pd.DataFrame, Path], None]]] = None

    def copy(self, **kwargs) -> IoTyping:
        x = _copy(self)
        for k, v in kwargs.items():
            if not hasattr(x, k):
                raise AttributeError(f"No attribute {k}")
            setattr(x, k, v)
        return x

    @property
    def use_attrs(self) -> bool:
        """
        Whether to read and write ``pd.DataFrame.attrs`` when passing ``attrs=None``.
        """
        return self._use_attrs

    @property
    def attrs_suffix(self) -> str:
        """
        File filename suffix detailing where to save/load per-DataFrame "attrs" (metadata).
        Will be appended to the DataFrame filename.
        """
        return self._attrs_suffix

    @property
    def attrs_json_kwargs(self) -> Mapping[str, Any]:
        """
        Keyword arguments for :class:`typeddfs.json_utils.JsonUtils.encoder`.
        Used when writing attrs.
        """
        return self._attrs_json_kwargs

    @property
    def hdf_key(self) -> str:
        """
        The default key used in :meth:`typeddfs.abs_df.AbsDf.to_hdf`.
        The key is also used in :meth:`typeddfs.abs_df.AbsDf.read_hdf.`
        """
        return self._hdf_key

    @property
    def toml_aot(self) -> str:
        """
        The name of the Array of Tables (AoT) used when reading TOML.

        .. caution::
            Only checks the read keyword arguments, not write
        """
        return self._read_kwargs.get(FileFormat.toml, {}).get("aot", _TOML_AOT)

    @property
    def flexwf_sep(self) -> str:
        """
        The delimiter used when reading "flex-width" format.

        .. caution::
            Only checks the read keyword arguments, not write
        """
        return self._read_kwargs.get(FileFormat.flexwf, {}).get("sep", _FLEXWF_SEP)

    @property
    def secure(self) -> bool:
        """
        Whether to forbid insecure operations and formats.
        """
        return self._secure

    @property
    def recommended(self) -> bool:
        """
        Whether to forbid discouraged formats like fixed-width and HDF5.
        Excludes all insecure formats.
        """
        return self._recommended

    @property
    def hash_algorithm(self) -> Optional[str]:
        """
        The hash algorithm used for checksums.
        """
        return self._hash_alg

    @property
    def file_hash(self) -> bool:
        """
        Whether to save per-file hash files by default.
        Specifically, in :meth:`typeddfs.abs_df.AbsDf.write_file`.
        """
        return self._save_hash_file

    @property
    def dir_hash(self) -> bool:
        """
        Whether to save (append) to per-directory hash files by default.
        Specifically, in :meth:`typeddfs.abs_df.AbsDf.write_file`.
        """
        return self._save_hash_file

    @property
    def remap_suffixes(self) -> Mapping[str, FileFormat]:
        """
        Returns filename formats that have been re-mapped to file formats.
        These are used in ``read_file`` and ``write_file``.

        Note:
            This should rarely be needed.
            An exception might be ``.txt`` to tsv rather than lines; Excel uses this.
        """
        return _opt_dict(self._remap_suffixes)

    @property
    def text_encoding(self) -> str:
        """
        Can be an exact encoding like utf-8, "platform", "utf8(bom)" or "utf16(bom)".
        See the docs in ``TypedDfs.typed().encoding`` for details.
        """
        return self._text_encoding

    @property
    def is_text_encoding_utf(self) -> bool:
        return Utils.get_encoding(self._text_encoding) in ["utf-8", "utf-16", "utf-32"]

    @property
    def read_kwargs(self) -> Mapping[FileFormat, Mapping[str, Any]]:
        """
        Passes kwargs into read functions from ``read_file``.
        These are keyword arguments that are automatically added into
        specific ``read_`` methods when called by ``read_file``.

        Note:
            This should rarely be needed
        """
        return _opt_dict(self._read_kwargs)

    @property
    def write_kwargs(self) -> Mapping[FileFormat, Mapping[str, Any]]:
        """
        Passes kwargs into write functions from ``to_file``.
        These are keyword arguments that are automatically added into
        specific ``to_`` methods when called by ``write_file``.

        Note:
            This should rarely be needed
        """
        return _opt_dict(self._write_kwargs)

    @property
    def read_suffix_kwargs(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Per-suffix kwargs into read functions from ``read_file``.
        Modulo compression (e.g. .tsv is equivalent to .tsv.gz).
        """
        return _opt_dict(self._remapped_read_kwargs)

    @property
    def write_suffix_kwargs(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Per-suffix kwargs into read functions from ``write_file``.
        Modulo compression (e.g. .tsv is equivalent to .tsv.gz).
        """
        return _opt_dict(self._remapped_write_kwargs)

    @property
    def custom_readers(self) -> Mapping[str, Callable[[Path], pd.DataFrame]]:
        """
        Mapping from filename suffixes (module compression) to custom reading methods.
        """
        return _opt_dict(self._custom_readers)

    @property
    def custom_writers(self) -> Mapping[str, Callable[[pd.DataFrame, Path], None]]:
        """
        Mapping from filename suffixes (module compression) to custom reading methods.
        """
        return _opt_dict(self._custom_writers)


FINAL_IO_TYPING = IoTyping()


@dataclass(frozen=True, repr=True)
class DfTyping:
    """
    Contains all information about how to type a DataFrame subclass.
    """

    _io_typing: IoTyping = FINAL_IO_TYPING
    _post_processing: Optional[Callable[[T], Optional[T]]] = None
    _verifications: Optional[Sequence[Callable[[T], Union[None, bool, str]]]] = None
    _column_series_name: Union[bool, None, str] = None
    _index_series_name: Union[bool, None, str] = None
    _more_columns_allowed: bool = True
    _more_index_names_allowed: bool = True
    _required_columns: Optional[Sequence[str]] = None
    _reserved_columns: Optional[Sequence[str]] = None
    _required_index_names: Optional[Sequence[str]] = None
    _reserved_index_names: Optional[Sequence[str]] = None
    _auto_dtypes: Optional[Mapping[str, Type[Any]]] = None
    _columns_to_drop: Optional[Set[str]] = None
    _value_dtype: Optional[Type[Any]] = None
    _order_dclass: bool = True

    def copy(self, **kwargs) -> DfTyping:
        x = _copy(self)
        for k, v in kwargs.items():
            if not hasattr(x, k):
                raise AttributeError(f"No attribute {k}")
            setattr(x, k, v)
        return x

    @property
    def io(self) -> IoTyping:
        return self._io_typing

    @property
    def is_strict(self) -> bool:
        """
        Returns True if this allows unspecified index levels **or** columns.
        """
        return not self._more_columns_allowed and not self._more_index_names_allowed

    @property
    def order_dataclass(self) -> bool:
        """
        Whether the corresponding dataclass can be sorted (has ``__lt__``).
        """
        return self._order_dclass

    @property
    def index_series_name(self) -> Union[bool, None, str]:
        """
        Intelligently returns ``df.index.name``.
        Returns a value that will be forced into ``df.index.name`` on calling ``convert``,
        *only if* the DataFrame is multi-index.
        If ``None``, will set ``df.index.name = None`` if ``df.index.names != [None]``.
        If ``False``, will not set. (``True`` is treated the same as ``None``.)
        """
        return self._index_series_name

    @property
    def column_series_name(self) -> Union[bool, None, str]:
        """
        Intelligently returns ``df.columns.name``.
        Returns a value that will be forced into ``df.columns.name`` on calling ``convert``.
        If ``None``, will set ``df.columns.name = None``.
        If ``False``, will not set. (``True`` is treated the same as ``None``.)
        """
        return self._column_series_name

    @property
    def more_indices_allowed(self) -> bool:
        """
        Returns whether the DataFrame allows index levels that are neither reserved nor required.
        """
        return self._more_index_names_allowed

    @property
    def more_columns_allowed(self) -> bool:
        """
        Returns whether the DataFrame allows columns that are not reserved or required.
        """
        return self._more_columns_allowed

    @property
    def required_columns(self) -> Sequence[str]:
        """
        Returns the list of required column names.
        """
        return _opt_list(self._required_columns)

    @property
    def reserved_columns(self) -> Sequence[str]:
        """
        Returns the list of reserved (optional) column names.
        """
        return _opt_list(self._reserved_columns)

    @property
    def required_index_names(self) -> Sequence[str]:
        """
        Returns the list of required column names.
        """
        return _opt_list(self._required_index_names)

    @property
    def reserved_index_names(self) -> Sequence[str]:
        """
        Returns the list of reserved (optional) index levels.
        """
        return _opt_list(self._reserved_index_names)

    @property
    def known_column_names(self) -> Sequence[str]:
        """
        Returns all columns that are required or reserved.
        The sort order positions required columns first.
        """
        return [*self.required_columns, *self.reserved_columns]

    @property
    def known_index_names(self) -> Sequence[str]:
        """
        Returns all index levels that are required or reserved.
        The sort order positions required columns first.
        """
        return [*self.required_index_names, *self.reserved_index_names]

    @property
    def required_names(self) -> Sequence[str]:
        """
        Returns all index and column names that are required.
        The sort order is: required index, required columns.
        """
        return [*self.required_index_names, *self.required_columns]

    @property
    def reserved_names(self) -> Sequence[str]:
        """
        Returns all index and column names that are **not** required.
        The sort order is: reserved index, reserved columns.
        """
        return [*self.reserved_index_names, *self.reserved_columns]

    @property
    def known_names(self) -> Sequence[str]:
        """
        Returns all index and column names that are required or reserved.
        The sort order is: required index, reserved index, required columns, reserved columns.
        """
        return [
            *self.required_index_names,
            *self.reserved_index_names,
            *self.required_columns,
            *self.reserved_columns,
        ]

    @property
    def value_dtype(self) -> Optional[Type[Any]]:
        """
        A type for "values" in a simple DataFrame.
        Typically numeric.
        """
        return self._value_dtype

    @property
    def auto_dtypes(self) -> Mapping[str, Type[Any]]:
        """
        A mapping from column/index names to the expected dtype.
        These are used via ``pd.Series.as_type`` for automatic conversion.
        An error will be raised if a ``as_type`` call fails.
        Note that Pandas frequently just does not perform the conversion,
        rather than raising an error.
        The keys should be contained in ``known_names``, but this is not strictly required.
        """
        return _opt_dict(self._auto_dtypes)

    @property
    def columns_to_drop(self) -> Set[str]:
        """
        Returns the list of columns that are automatically dropped by ``convert``.
        This does NOT include "level_0" and "index, which are ALWAYS dropped.
        """
        return _opt_set(self._columns_to_drop)

    @property
    def post_processing(self) -> Optional[Callable[[T], Optional[T]]]:
        """
        A function to be called at the final stage of ``convert``.
        It is called immediately before ``verifications`` are checked.
        The function takes a copy of the input ``BaseDf`` and returns a new copy.

        Note:
            Although a copy is passed as input, the function should not modify it.
            Technically, doing so will cause problems only if the DataFrame's internal values
            are modified. The value passed is a *shallow* copy (see ``pd.DataFrame.copy``).
        """
        return self._post_processing

    @property
    def verifications(self) -> Sequence[Callable[[T], Union[None, bool, str]]]:
        """
        Additional requirements for the DataFrame to be conformant.

        Returns:
            A sequence of conditions that map the DF to None or True if the condition passes,
            or False or the string of an error message if it fails
        """
        return _opt_list(self._verifications)


FINAL_DF_TYPING = DfTyping()


__all__ = ["IoTyping", "DfTyping", "FINAL_IO_TYPING", "FINAL_DF_TYPING"]
