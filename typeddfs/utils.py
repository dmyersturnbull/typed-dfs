"""
Tools that could possibly be used outside of typed-dfs.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

from pandas.io.common import get_handle

# noinspection PyProtectedMember
from tabulate import DataRow, TableFormat, _table_formats


class Utils:
    @classmethod
    def write(cls, path_or_buff, content, mode: str = "w", **kwargs) -> Optional[str]:
        """
        Writes using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        if path_or_buff is None:
            return content
        with get_handle(path_or_buff, mode, **kwargs) as f:
            f.handle.write(content)

    @classmethod
    def read(cls, path_or_buff, content, mode: str = "w", **kwargs) -> Optional[str]:
        """
        Reads using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        if path_or_buff is None:
            return content
        with get_handle(path_or_buff, mode, **kwargs) as f:
            f.handle.write(content)

    @classmethod
    def get_encoding(cls, encoding: str = "utf-8") -> str:
        encoding = encoding.lower().replace("-", "")
        if encoding == "platform":
            encoding = sys.getdefaultencoding()
        if encoding == "utf8(bom)":
            encoding = "utf-8-sig" if os.name == "nt" else "utf-8"
        if encoding == "utf16(bom)":
            encoding = "utf-16-sig" if os.name == "nt" else "utf-16"
        return encoding

    @classmethod
    def table_formats(cls) -> Sequence[str]:
        return _table_formats.keys()

    @classmethod
    def table_format(cls, fmt: str) -> TableFormat:
        return _table_formats[fmt]

    @classmethod
    def plain_table_format(cls, sep: str = " ", **kwargs):
        defaults = dict(
            lineabove=None,
            linebelowheader=None,
            linebetweenrows=None,
            linebelow=None,
            headerrow=DataRow("", sep, ""),
            datarow=DataRow("", sep, ""),
            padding=0,
            with_header_hide=None,
        )
        kwargs = {**defaults, **kwargs}
        return TableFormat(**kwargs)


__all__ = ["Utils"]
