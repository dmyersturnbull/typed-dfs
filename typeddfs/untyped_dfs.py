"""
Defines DataFrames with convenience methods but that do not enforce invariants.
"""
from __future__ import annotations

from pathlib import PurePath
from typing import Optional, Union

import pandas as pd

from typeddfs.base_dfs import BaseDf

PathLike = Union[str, PurePath]


class UntypedDf(BaseDf):
    """
    A concrete DataFrame that does not require columns or enforce conditions.
    Overrides a number of DataFrame methods that preserve the subclass.
    For example, calling ``df.reset_index()`` will return a ``UntypedDf`` of the same type as ``df``.
    """

    @classmethod
    def read_csv(cls, *args, **kwargs) -> __qualname__:
        """
        Reads from CSV, converting to this type.
        Using to_csv() and read_csv() from BaseFrame, this property holds::

            df.to_csv(path)
            df.__class__.read_csv(path) == df
        """
        index_col = kwargs.get("index_col", False)
        df = pd.read_csv(*args, index_col=index_col)
        return cls._check_and_change(df)

    def to_csv(self, path: PathLike, *args, **kwargs) -> Optional[str]:
        """
        Writes CSV.
        Using to_csv() and read_csv() from BaseFrame, this property holds::

            df.to_csv(path)
            df.__class__.read_csv(path) == df
        """
        if "index" in kwargs:
            return super().to_csv(path, *args, **kwargs)
        else:
            df = self.vanilla().reset_index(drop=list(self.index.names) == [None])
            return df.to_csv(path, *args, index=False, **kwargs)


__all__ = ["UntypedDf"]
