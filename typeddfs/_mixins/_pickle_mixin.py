"""
Mixin for pickle.
"""
from __future__ import annotations

import pandas as pd


class _PickleMixin:
    @classmethod
    def read_pickle(cls, filepath_or_buffer, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_pickle(filepath_or_buffer, *args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        return cls._convert_typed(df)

    # noinspection PyFinal
    def to_pickle(self, path, *args, **kwargs) -> None:
        df = self.vanilla()
        return df.to_pickle(path, *args, **kwargs)


__all__ = ["_PickleMixin"]
