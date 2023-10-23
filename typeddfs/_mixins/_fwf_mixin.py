# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Mixin for fixed-width format.
"""
from __future__ import annotations

import csv
from collections.abc import Sequence
from typing import Union

import pandas as pd

from typeddfs.utils import Utils

_SheetNamesOrIndices = Union[Sequence[int | str], int, str]


class _FwfMixin:
    @classmethod
    def read_fwf(cls, *args, **kwargs) -> __qualname__:
        try:
            return cls._convert_typed(pd.read_fwf(*args, **kwargs))
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()

    def to_fwf(
        self,
        path_or_buff=None,
        mode: str = "w",
        colspecs: Sequence[tuple[int, int]] | None = None,
        widths: Sequence[int] | None = None,
        na_rep: str | None = None,
        float_format: str | None = None,
        date_format: str | None = None,
        decimal: str = ".",
        **kwargs,
    ) -> str | None:
        """
        Writes a fixed-width text format.
        See ``read_fwf`` and ``to_flexwf`` for more info.

        .. warning:

            This method is a preview. Not all options are complete, and
            behavior is subject to change in a future (major) version.
            Notably, Pandas may eventually introduce a method with the same name.

        Args:
            path_or_buff: Path or buffer
            mode: write or append (w/a)
            colspecs: A list of tuples giving the extents of the fixed-width fields of each line
                      as half-open intervals (i.e., [from, to[ )
            widths: A list of field widths which can be used instead of ``colspecs``
                   if the intervals are contiguous
            na_rep: Missing data representation
            float_format: Format string for floating point numbers
            date_format: Format string for datetime objects
            decimal: Character recognized as decimal separator. E.g. use `,` for European data.
            kwargs: Passed to :meth:`typeddfs.utils.Utils.write`

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        if colspecs is not None and widths is not None:
            msg = "Both widths and colspecs passed"
            raise ValueError(msg)
        if widths is not None:
            colspecs = []
            at = 0
            for w in widths:
                colspecs.append((at, at + w))
                at += w
        # if colspecs is None:
        if True:
            # TODO: use format, etc.
            content = self._tabulate(Utils.plain_table_format(sep=" "), disable_numparse=True)
        else:
            df = self.vanilla_reset()
            if len(df.columns) != len(colspecs):
                msg = f"{colspecs} column intervals for {len(df.columns)} columns"
                raise ValueError(msg)
            for col, (start, end) in zip(df.columns, colspecs):
                width = end - start
                mx = df[col].map(str).map(len).max()
                if mx > width:
                    msg = f"Column {col} has max length {mx} > {end - start}"
                    raise ValueError(msg)
            _number_format = {
                "na_rep": na_rep,
                "float_format": float_format,
                "date_format": date_format,
                "quoting": csv.QUOTE_NONE,
                "decimal": decimal,
            }
            res = df._mgr.to_native_types(**_number_format)
            [res.iget_values(i) for i in range(len(res.items))]
            content = None  # TODO
        if path_or_buff is None:
            return content
        _encoding = {"encoding": kwargs.get("encoding")} if "encoding" in kwargs else {}
        _compression = {"encoding": kwargs.get("compression")} if "compression" in kwargs else {}
        Utils.write(path_or_buff, content, mode=mode, **_encoding, **_compression)


__all__ = ["_FwfMixin"]
