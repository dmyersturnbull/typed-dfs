"""
Mixin for line-by-line text files.
"""
from __future__ import annotations

import csv
from typing import Optional, Union

import pandas as pd
from tabulate import TableFormat, tabulate

from typeddfs.df_errors import NotSingleColumnError
from typeddfs.utils import IoUtils, MiscUtils
from typeddfs.utils._utils import _FAKE_SEP


class _LinesMixin:
    def to_lines(
        self,
        path_or_buff=None,
        mode: str = "w",
        **kwargs,
    ) -> Optional[str]:
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
            raise NotSingleColumnError(f"Cannot write {len(df.columns)} columns ({df}) to lines")
        return df.to_csv(
            path_or_buff, mode=mode, index=False, sep=_FAKE_SEP, quoting=csv.QUOTE_NONE, **kwargs
        )

    @classmethod
    def read_lines(
        cls,
        path_or_buff,
        **kwargs,
    ) -> __qualname__:
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
            raise NotSingleColumnError(f"Read multiple columns on {path_or_buff}")
        return cls._convert_typed(df)

    @classmethod
    def read_fwf(cls, *args, **kwargs) -> __qualname__:
        try:
            return cls._convert_typed(pd.read_fwf(*args, **kwargs))
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()

    @classmethod
    def read_flexwf(
        cls,
        path_or_buff,
        sep: str = r"\|\|\|",
        **kwargs,
    ) -> __qualname__:
        r"""
        Reads a "flexible-width format".
        The delimiter (``sep``) is important.
        **Note that ``sep`` is a regex pattern if it contains more than 1 char.**

        These are designed to read and write (``to_flexwf``) as though they
        were fixed-width. Specifically, all of the columns line up but are
        separated by a possibly multi-character delimiter.

        The files ignore blank lines, strip whitespace,
        always have a header, never quote values, and have no default index column
        unless given by ``required_columns()``, etc.

        Args:
            path_or_buff: Path or buffer
            sep: The delimiter, a regex pattern
            kwargs: Passed to ``read_csv``; may include 'comment' and 'skip_blank_lines'
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("skip_blank_lines", True)
        try:
            df = pd.read_csv(
                path_or_buff,
                sep=sep,
                index_col=False,
                quoting=csv.QUOTE_NONE,
                engine="python",
                header=0,
                **kwargs,
            )
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            try:
                df[c] = df[c].str.strip()
            except AttributeError:
                pass
        return cls._convert_typed(df)

    def to_flexwf(
        self, path_or_buff=None, sep: str = "|||", mode: str = "w", **kwargs
    ) -> Optional[str]:
        """
        Writes a fixed-width formatter, optionally with a delimiter, which can be multiple characters.

        See ``read_flexwf`` for more info.

        Args:
            path_or_buff: Path or buffer
            sep: The delimiter, 0 or more characters
            mode: write or append (w/a)
            kwargs: Passed to ``Utils.write``; may include 'encoding'

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        fmt = MiscUtils.plain_table_format(" " + sep + " ")
        content = self._tabulate(fmt, disable_numparse=True)
        if path_or_buff is None:
            return content
        IoUtils.write(path_or_buff, content, mode=mode, **kwargs)

    @classmethod
    def _lines_files_apply(cls) -> bool:
        # CAN apply as long as we don't REQUIRE more than 1 column
        return len(cls.get_typing().required_names) <= 1

    def _tabulate(self, fmt: Union[str, TableFormat], **kwargs) -> str:
        df = self.vanilla_reset()
        return tabulate(df.to_numpy().tolist(), list(df.columns), tablefmt=fmt, **kwargs)


__all__ = ["_LinesMixin"]
