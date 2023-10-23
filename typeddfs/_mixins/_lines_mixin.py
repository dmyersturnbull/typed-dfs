# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Mixin for line-by-line text files.
"""
from __future__ import annotations

import csv

import pandas as pd
from tabulate import TableFormat, tabulate

from typeddfs.df_errors import NotSingleColumnError
from typeddfs.utils._utils import _FAKE_SEP


class _LinesMixin:
    def to_lines(
        self,
        path_or_buff=None,
        *,
        mode: str = "w",
        **kwargs,
    ) -> str | None:
        r"""
        Writes a file that contains one row per line and 1 column per line.
        Associated with ``.lines`` or ``.txt``.

        .. caution::

            For technical reasons, values cannot contain a 6-em space (U+2008).
            Their presence will result in undefined behavior.

        Args:
            path_or_buff: Path or buffer
            mode: Write ('w') or append ('a')
            kwargs: Passed to ``pd.DataFrame.to_csv``

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("header", True)
        df = self.vanilla_reset()
        if len(df.columns) != 1:
            msg = f"Cannot write {len(df.columns)} columns ({df}) to lines"
            raise NotSingleColumnError(msg)
        return df.to_csv(
            path_or_buff,
            mode=mode,
            index=False,
            sep=_FAKE_SEP,
            quoting=csv.QUOTE_NONE,
            **kwargs,
        )

    @classmethod
    def read_lines(cls, path_or_buff, **kwargs) -> __qualname__:
        r"""
        Reads a file that contains 1 row and 1 column per line.
        Skips lines that are blank after trimming whitespace.
        Also skips comments if ``comment`` is set.

        .. caution::

            For technical reasons, values cannot contain a 6-em space (U+2008).
            Their presence will result in undefined behavior.

        Args:
            path_or_buff: Path or buffer
            kwargs: Passed to ``pd.DataFrame.read_csv``
                    E.g. 'comment', 'encoding', 'skip_blank_lines', and 'line_terminator'
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("skip_blank_lines", True)
        kwargs.setdefault("header", 0)
        kwargs.setdefault("engine", "python")
        try:
            df = pd.read_csv(
                path_or_buff,
                sep=_FAKE_SEP,
                index_col=False,
                quoting=csv.QUOTE_NONE,
                **kwargs,
            )
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        if len(df.columns) > 1:
            msg = f"Read multiple columns on {path_or_buff}"
            raise NotSingleColumnError(msg)
        return cls._convert_typed(df)

    @classmethod
    def _lines_files_apply(cls) -> bool:
        # CAN apply as long as we don't REQUIRE more than 1 column
        return len(cls.get_typing().required_names) <= 1

    def _tabulate(self, fmt: str | TableFormat, **kwargs) -> str:
        df = self.vanilla_reset()
        return tabulate(df.to_numpy().tolist(), list(df.columns), tablefmt=fmt, **kwargs)


__all__ = ["_LinesMixin"]
