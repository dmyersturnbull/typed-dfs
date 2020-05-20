from __future__ import annotations
from pathlib import PurePath
from typing import Union, Optional
import pandas as pd

from typeddfs.base_dfs import *

PathLike = Union[str, PurePath]


class UntypedDf(BaseDf):
    """
    A concrete BaseFrame that does not require special columns.
    Overrides a number of DataFrame methods to convert before returning.
    """

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.index.names:
            return self.index.get_level_values(item)
        else:
            return super().__getitem__(item)

    @classmethod
    def read_csv(cls, *args, **kwargs):
        """
        Reads from CSV, converting to this type.
        Using to_csv() and read_csv() from BaseFrame, this property holds:
            ```
            df.to_csv(path)
            df.__class__.read_csv(path) == df
            ```
        """
        index_col = kwargs.get("index_col", False)
        df = pd.read_csv(*args, index_col=index_col)
        return cls._check_and_change(df)

    def to_csv(self, path: PathLike, *args, **kwargs) -> Optional[str]:
        """
        Writes CSV.
        Using to_csv() and read_csv() from BaseFrame, this property holds:
            ```
            df.to_csv(path)
            df.__class__.read_csv(path) == df
            ```
        """
        if "index" in kwargs:
            return super().to_csv(path, *args, **kwargs)
        else:
            df = self.vanilla().reset_index(drop=list(self.index.names) == [None])
            return df.to_csv(path, *args, index=False, **kwargs)


__all__ = ["UntypedDf"]
