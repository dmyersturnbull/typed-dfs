"""
Defines the superclasses of the types ``TypedDf`` and ``UntypedDf``.
"""
from __future__ import annotations

import abc
from typing import Callable, Optional, Sequence, Union

import pandas as pd

from typeddfs.abs_df import AbsDf
from typeddfs.df_errors import InvalidDfError, VerificationFailedError


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
    def of(cls, df: pd.DataFrame) -> __qualname__:
        """
        Converts a vanilla Pandas DataFrame to cls.
        See ``convert`` for more info.

        This is normally an exact alias for ``convert``.
        Occasionally, it may accept more values, as long as it always falls back to ``convert``.
        This is intended to facilitate fast lookups.
        For example, ``Customers.of("john")`` could return a DataFrame for a database customer,
        or return the result of ``Customers.convert(...)`` if a DataFrame instance is provided.
        """
        return cls.convert(df)

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
        # noinspection PyTypeChecker
        df = cls._post_process(df)
        return df

    @classmethod
    def post_processing(cls) -> Optional[Callable[[BaseDf], Optional[BaseDf]]]:
        """
        A function to be called at the final stage of ``convert``.
        It is called immediately before ``verifications`` are checked.
        The function takes a copy of the input ``BaseDf`` and returns a new copy.

        Note:
            Although a copy is passed as input, the function should not modify it.
            Technically, doing so will cause problems only if the DataFrame's internal values
            are modified. The value passed is a *shallow* copy (see ``pd.DataFrame.copy``).
        """
        return None

    @classmethod
    def verifications(cls) -> Sequence[Callable[[BaseDf], Union[None, bool, str]]]:
        """
        Additional requirements for the DataFrame to be conformant.

        Returns:
            A sequence of conditions that map the DF to None or True if the condition passes,
            or False or the string of an error message if it fails
        """
        return []

    @classmethod
    def is_valid(cls, df: pd.DataFrame) -> bool:
        try:
            cls._check(df)
        except InvalidDfError:
            return False
        return True

    @classmethod
    def _check(cls, df) -> None:
        for req in cls.verifications():
            value = req(df)
            if value is not None and value is not True:
                raise VerificationFailedError(str(value))

    @classmethod
    def _post_process(cls, df) -> pd.DataFrame:
        if cls.post_processing() is not None:
            df = cls.post_processing()(df)
        return df


__all__ = ["BaseDf"]
