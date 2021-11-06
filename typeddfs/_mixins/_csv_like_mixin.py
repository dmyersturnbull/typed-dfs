"""
Mixin for CSV and TSV.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class _CsvLikeMixin:
    @classmethod
    def read_tsv(cls, *args, **kwargs) -> __qualname__:
        """
        Reads tab-separated data.
        See  :meth:`read_csv` for more info.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "sep"}
        return cls.read_csv(*args, sep="\t", **kwargs)

    @classmethod
    def read_csv(cls, *args, **kwargs) -> __qualname__:
        """
        Reads from CSV, converting to this type.
        Using to_csv() and read_csv() from BaseFrame, this property holds::

            df.to_csv(path)
            df.__class__.read_csv(path) == df

        Passing ``index`` on ``to_csv`` or ``index_col`` on ``read_csv``
        explicitly will break this invariant.

        Args:
            args: Passed to ``pd.read_csv``; should start with a path or buffer
            kwargs: Passed to ``pd.read_csv``.
        """
        kwargs = dict(kwargs)
        # we want to set index=False, but we also want to let the user override
        # checking for index in the positional args
        # this is a really good case against positional arguments in languages
        # 'index_col' is in the 6th positional slot
        # that's ONLY IF we don't list the path as the first arg though!!!
        # if we added path_or_buf before `*args`, this would need to be < 5
        if len(args) < 6:
            kwargs.setdefault("index_col", False)
        try:
            df = pd.read_csv(*args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        return cls._convert_typed(df)

    def to_tsv(self, *args, **kwargs) -> Optional[str]:
        """
        Writes tab-separated data.
        See :meth:`to_csv` for more info.
        """
        return self.to_csv(*args, sep="\t", **kwargs)

    # noinspection PyFinal
    def to_csv(self, *args, **kwargs) -> Optional[str]:
        kwargs = dict(kwargs)
        kwargs.setdefault("index", False)
        df = self.vanilla_reset()
        return df.to_csv(*args, **kwargs)


__all__ = ["_CsvLikeMixin"]
