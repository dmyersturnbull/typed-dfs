"""
Defines DataFrames with convenience methods but that do not enforce invariants.
"""
from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from typeddfs.base_dfs import BaseDf
from typeddfs.df_typing import IoTyping, DfTyping, FINAL_DF_TYPING

_empty_io_typing = IoTyping()


class UntypedDf(BaseDf):
    """
    A concrete DataFrame that does not require columns or enforce conditions.
    Overrides a number of DataFrame methods that preserve the subclass.
    For example, calling ``df.reset_index()`` will return a ``UntypedDf`` of the same type as ``df``.
    """

    @classmethod
    def get_typing(cls) -> DfTyping:
        return FINAL_DF_TYPING

    @classmethod
    def new_df(
        cls, rows: int = 0, cols: Union[int, Sequence[str]] = 0, fill: Any = 0
    ) -> __qualname__:
        """
        Creates a new, semi-arbitrary DataFrame of the specified rows and columns.
        The DataFrame will have no index.

        Arguments:
            rows: Number of rows
            cols: Number of columns or a sequence of column labels
            fill: Fill every cell with this value
        """
        if isinstance(cols, int):
            cols = [str(c) for c in range(cols)]
        a = np.ndarray(shape=(rows, len(cols)))
        a.fill(fill)
        df = pd.DataFrame(a, columns=cols)
        return UntypedDf.convert(df)


__all__ = ["UntypedDf"]
