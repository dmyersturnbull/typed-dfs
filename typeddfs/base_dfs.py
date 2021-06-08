"""
Defines the superclasses of the types ``TypedDf`` and ``UntypedDf``.
"""
from __future__ import annotations
import abc

import pandas as pd

from typeddfs.abs_df import AbsDf


class BaseDf(AbsDf, metaclass=abc.ABCMeta):
    """
    A subclass of ``AbsDf`` with ``convert()`` and ``vanilla()`` methods.
    This is an abstract version of ``TypedDf`` that has no declaration of what "typed" means,
    only a method ``BaseDf.convert`` to be overridden.
    Note that ``UntypedDf`` also inherits from this class; it simply does not override ``convert``.
    You can add your own implementation if ``TypedDf`` is missing a feature you need.
    """

    def __getitem__(self, item) -> __qualname__:
        if isinstance(item, str) and item in self.index.names:
            return self.index.get_level_values(item)
        else:
            return super().__getitem__(item)

    @classmethod
    def convert(cls, df: pd.DataFrame) -> __qualname__:
        """
        Converts a vanilla Pandas DataFrame to cls.
        Sets the index.

        Args:
            df: The Pandas DataFrame or member of cls; will have its __class_ change but will otherwise not be affected

        Returns:
            A copy
        """
        df = df.copy()
        df.__class__ = cls
        return df


__all__ = ["BaseDf"]
