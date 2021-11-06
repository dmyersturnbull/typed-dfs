"""
Combines various IO mixins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd
from tabulate import TableFormat

from typeddfs._mixins._csv_like_mixin import _CsvLikeMixin
from typeddfs._mixins._excel_mixins import _ExcelMixin
from typeddfs._mixins._feather_parquet_hdf_mixin import _FeatherParquetHdfMixin
from typeddfs._mixins._flexwf_mixin import _FlexwfMixin
from typeddfs._mixins._formatted_mixin import _FormattedMixin
from typeddfs._mixins._fwf_mixin import _FwfMixin
from typeddfs._mixins._ini_like_mixin import _IniLikeMixin
from typeddfs._mixins._json_xml_mixin import _JsonXmlMixin
from typeddfs._mixins._lines_mixin import _LinesMixin
from typeddfs._mixins._pickle_mixin import _PickleMixin
from typeddfs.df_errors import (
    FilenameSuffixError,
    FormatDiscouragedError,
    FormatInsecureError,
    UnsupportedOperationError,
)
from typeddfs.file_formats import CompressionFormat, FileFormat
from typeddfs.utils import Utils
from typeddfs.utils._utils import PathLike


class _FullIoMixin(
    _CsvLikeMixin,
    _ExcelMixin,
    _FeatherParquetHdfMixin,
    _FlexwfMixin,
    _FormattedMixin,
    _FwfMixin,
    _IniLikeMixin,
    _JsonXmlMixin,
    _LinesMixin,
    _PickleMixin,
):
    def pretty_print(
        self,
        fmt: Union[None, str, TableFormat] = None,
        *,
        to: Optional[PathLike] = None,
        mode: str = "w",
        **kwargs,
    ) -> str:
        """
        Outputs a pretty table using the `tabulate <https://pypi.org/project/tabulate/>`_ package.

        Args:
            fmt: A tabulate format; if None, chooses according to ``to``, falling back to ``"plain"``
            to: Write to this path (.gz, .zip, etc. is inferred)
            mode: Write mode: 'w', 'a', or 'x'
            kwargs: Passed to tabulate

        Returns:
            The formatted string
        """
        fmt = Utils.choose_table_format(path=to, fmt=fmt)
        s = self._tabulate(fmt, **kwargs)
        if to is not None:
            Utils.write(to, s, mode=mode)
        return s

    @classmethod
    def _call_read(
        cls,
        clazz,
        path: Union[Path, str],
    ) -> pd.DataFrame:
        fmt = cls._get_fmt(path)
        # noinspection HttpUrlsUsage
        if str(path).startswith("http://"):
            raise UnsupportedOperationError("Cannot read from http with .secure() enabled")
        cls._check_io_ok(path, fmt)
        kwargs = cls._get_read_kwargs(fmt, path)
        fn = cls._get_io(clazz, path, fmt, cls._get_write_kwargs(fmt, path), "read_")
        return fn(path, **kwargs)

    def _call_write(self, path: Union[Path, str]) -> Optional[str]:
        cls = self.__class__
        fmt = self._get_fmt(path)
        cls._check_io_ok(path, fmt)
        kwargs = cls._get_write_kwargs(fmt, path)
        fn = self._get_io(self, path, fmt, cls._get_write_kwargs(fmt, path), "to_")
        return fn(path, **kwargs)

    @classmethod
    def _get_read_kwargs(cls, fmt: Optional[FileFormat], path: Path) -> Mapping[str, Any]:
        t = cls.get_typing().io
        real_suffix = CompressionFormat.strip_suffix(path).suffix
        kwargs = t.read_kwargs.get(fmt, {})
        kwargs.update(t.read_suffix_kwargs.get(real_suffix, {}))
        if fmt in [
            FileFormat.csv,
            FileFormat.tsv,
            FileFormat.properties,
            FileFormat.lines,
            FileFormat.flexwf,
            FileFormat.fwf,
            FileFormat.json,
        ]:
            encoding = kwargs.get("encoding", t.text_encoding)
            kwargs["encoding"] = Utils.get_encoding(encoding)
        return kwargs

    @classmethod
    def _get_write_kwargs(cls, fmt: Optional[FileFormat], path: Path) -> Mapping[str, Any]:
        t = cls.get_typing().io
        real_suffix = CompressionFormat.strip_suffix(path).suffix
        kwargs = t.write_kwargs.get(fmt, {})
        kwargs.update(t.write_suffix_kwargs.get(real_suffix, {}))
        if fmt is FileFormat.json:
            # not perfect, but much better than the alternative of failing
            # I don't see a better solution anyway
            kwargs["force_ascii"] = False
        elif (
            fmt is not None and fmt.supports_encoding
        ):  # and IS NOT JSON -- it doesn't use "encoding="
            encoding = kwargs.get("encoding", t.text_encoding)
            kwargs["encoding"] = Utils.get_encoding(encoding)
        return kwargs

    @classmethod
    def _get_fmt(cls, path: Path) -> Optional[FileFormat]:
        t = cls.get_typing().io
        mp = FileFormat.suffix_map()
        mp.update(t.remap_suffixes)
        return FileFormat.from_path_or_none(path, format_map=mp)

    @classmethod
    def _check_io_ok(cls, path: Path, fmt: Optional[FileFormat]):
        t = cls.get_typing().io
        if fmt is not None:
            if t.secure and not fmt.is_secure:
                raise FormatInsecureError(
                    f"Insecure format {fmt} forbidden by typing", key=fmt.name
                )
            if t.recommended and not fmt.is_recommended:
                raise FormatDiscouragedError(
                    f"Discouraged format {fmt} forbidden by typing", key=fmt.name
                )

    @classmethod
    def _get_io(cls, on, path: Path, fmt: FileFormat, custom, prefix: str):
        if fmt is not None:
            return getattr(on, prefix + fmt.name)
        real_suffix = CompressionFormat.strip_suffix(path).suffix
        try:
            return custom[real_suffix]
        except KeyError:
            raise FilenameSuffixError(f"No format found for suffix (path: {path})") from None


__all__ = ["_FullIoMixin"]
