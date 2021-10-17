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
    FormatDiscouragedError,
    FormatInsecureError,
    UnsupportedOperationError,
)
from typeddfs.file_formats import FileFormat
from typeddfs.utils import Utils


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
    """
    DataFrame that supports
    """

    def pretty_print(self, fmt: Union[str, TableFormat] = "plain", **kwargs) -> str:
        """
        Outputs a pretty table using the `tabulate <https://pypi.org/project/tabulate/>`_ package.
        """
        return self._tabulate(fmt, **kwargs)

    @classmethod
    def _call_read(
        cls,
        clazz,
        path: Union[Path, str],
    ) -> pd.DataFrame:
        t = cls.get_typing().io
        mp = FileFormat.suffix_map()
        mp.update(t.remap_suffixes)
        fmt = FileFormat.from_path(path, format_map=mp)
        # noinspection HttpUrlsUsage
        if isinstance(path, str) and path.startswith("http://"):
            raise UnsupportedOperationError("Cannot read from http with .secure() enabled")
        cls._check_io_ok(path, fmt)
        kwargs = cls._get_read_kwargs(fmt)
        fn = getattr(clazz, "read_" + fmt.name)
        return fn(path, **kwargs)

    def _call_write(
        self,
        path: Union[Path, str],
    ) -> Optional[str]:
        cls = self.__class__
        t = cls.get_typing().io
        mp = FileFormat.suffix_map()
        mp.update(t.remap_suffixes)
        fmt = FileFormat.from_path(path, format_map=mp)
        self._check_io_ok(path, fmt)
        kwargs = cls._get_write_kwargs(fmt)
        fn = getattr(self, "to_" + fmt.name)
        return fn(path, **kwargs)

    @classmethod
    def _get_read_kwargs(cls, fmt: FileFormat) -> Mapping[str, Any]:
        t = cls.get_typing().io
        kwargs = t.read_kwargs.get(fmt, {})
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
    def _check_io_ok(cls, path: Path, fmt: FileFormat):
        t = cls.get_typing().io
        if t.secure and not fmt.is_secure:
            raise FormatInsecureError(f"Insecure format {fmt} forbidden by typing", key=fmt.name)
        if t.recommended and not fmt.is_recommended:
            raise FormatDiscouragedError(
                f"Discouraged format {fmt} forbidden by typing", key=fmt.name
            )

    @classmethod
    def _get_write_kwargs(cls, fmt: FileFormat) -> Mapping[str, Any]:
        t = cls.get_typing().io
        kwargs = t.write_kwargs.get(fmt, {})
        if fmt is FileFormat.json:
            # not perfect, but much better than the alternative of failing
            # I don't see a better solution anyway
            kwargs["force_ascii"] = False
        elif fmt.supports_encoding:  # and IS NOT JSON -- it doesn't use "encoding="
            encoding = kwargs.get("encoding", t.text_encoding)
            kwargs["encoding"] = Utils.get_encoding(encoding)
        return kwargs


__all__ = ["_FullIoMixin"]
