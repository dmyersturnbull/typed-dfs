# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Mixin for CSV and TSV.
"""
from __future__ import annotations

import pandas as pd


class _CsvLikeMixin:
    @classmethod
    def read_tsv(cls, path_or_buff, **kwargs) -> __qualname__:
        """
        Reads tab-separated data.
        See  :meth:`read_csv` for more info.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "sep"}
        return cls.read_csv(path_or_buff, sep="\t", **kwargs)

    @classmethod
    def read_csv(cls, path_or_buff, **kwargs) -> __qualname__:
        """
        Reads from CSV, converting to this type.
        Using to_csv() and read_csv() from BaseFrame, this property holds::

            df.to_csv(path)
            df.__class__.read_csv(path) == df

        Passing ``index`` on ``to_csv`` or ``index_col`` on ``read_csv``
        explicitly will break this invariant.

        Args:
            path_or_buff: Passed to ``pd.read_csv`
            kwargs: Passed to ``pd.read_csv``.
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("index_col", False)
        try:
            df = pd.read_csv(path_or_buff, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        return cls._convert_typed(df)

    def to_tsv(self, path_or_buff, **kwargs) -> str | None:
        """
        Writes tab-separated data.
        See :meth:`to_csv` for more info.
        """
        return self.to_csv(path_or_buff, sep="\t", **kwargs)

    # noinspection PyFinal
    def to_csv(self, path_or_buff=None, **kwargs) -> str | None:
        kwargs = dict(kwargs)
        kwargs.setdefault("index", False)
        df = self.vanilla_reset()
        return df.to_csv(path_or_buff, **kwargs)


__all__ = ["_CsvLikeMixin"]
