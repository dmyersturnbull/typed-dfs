"""
Mixin for Excel/ODF IO.
"""
from __future__ import annotations

from pathlib import Path, PurePath
from typing import Optional, Sequence, Union

import pandas as pd

_SheetNamesOrIndices = Union[Sequence[Union[int, str]], int, str]


class _ExcelMixin:
    @classmethod
    def read_excel(cls, io, sheet_name: _SheetNamesOrIndices = 0, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_excel(io, sheet_name, *args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        # This only applies for .xlsb -- the others don't have this problem
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        return cls._convert_typed(df)

    # noinspection PyFinal,PyMethodOverriding
    def to_excel(self, excel_writer, *args, **kwargs) -> Optional[str]:
        kwargs = dict(kwargs)
        df = self.vanilla_reset()
        if isinstance(excel_writer, (str, PurePath)) and Path(excel_writer).suffix in [
            ".xls",
            ".ods",
            ".odt",
            ".odf",
        ]:
            # Pandas's defaults for XLS and ODS are so buggy that we should never use them
            # that is, unless the user actually passes engine=
            kwargs.setdefault("engine", "openpyxl")
        return df.to_excel(excel_writer, *args, **kwargs)

    @classmethod
    def read_xlsx(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads XLSX Excel files.
        Prefer this method over :meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_xlsx(self, excel_writer, *args, **kwargs) -> Optional[str]:
        """
        Writes XLSX Excel files.
        Prefer this method over :meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(excel_writer, *args, **kwargs)

    @classmethod
    def read_xls(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads legacy XLS Excel files.
        Prefer this method over :meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_xls(self, excel_writer, *args, **kwargs) -> Optional[str]:
        """
        Reads legacy XLS Excel files.
        Prefer this method over :meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(excel_writer, *args, **kwargs, engine="openpyxl")

    @classmethod
    def read_xlsb(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads XLSB Excel files.
        This is a relatively uncommon format.
        Prefer this method over :meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_xlsb(self, excel_writer, *args, **kwargs) -> Optional[str]:
        """
        Writes XLSB Excel files.
        This is a relatively uncommon format.
        Prefer this method over :meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(excel_writer, *args, **kwargs)

    @classmethod
    def read_ods(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads OpenDocument ODS/ODT files.
        Prefer this method over :meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_ods(self, ods_writer, *args, **kwargs) -> Optional[str]:
        """
        Writes OpenDocument ODS/ODT files.
        Prefer this method over :meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(ods_writer, *args, **kwargs)


__all__ = ["_ExcelMixin"]
