"""
Mixin for flex-wf.
"""
from __future__ import annotations

import csv
from typing import Optional

import pandas as pd

from typeddfs.utils import IoUtils, MiscUtils


class _FlexwfMixin:
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
        fmt = MiscUtils.plain_table_format(sep=" " + sep + " ")
        content = self._tabulate(fmt, disable_numparse=True)
        if path_or_buff is None:
            return content
        IoUtils.write(path_or_buff, content, mode=mode, **kwargs)


__all__ = ["_FlexwfMixin"]
