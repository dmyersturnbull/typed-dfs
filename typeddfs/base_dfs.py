"""
Defines the superclasses of the types ``TypedDf`` and ``UntypedDf``.
"""
from __future__ import annotations

import abc

import pandas as pd

from typeddfs.abs_dfs import AbsDf


class BaseDf(AbsDf, metaclass=abc.ABCMeta):
    """
    An abstract DataFrame type that has a way to convert and de-convert.
    A subclass of :py.class:`typeddfs.abs_dfs.AbsDf`,
    it has methods :py.meth:`convert` and :py.meth:`vanilla`.
    but no implementation or enforcement of typing.
    """

    def __getitem__(self, item):
        """
        Finds an index level or or column, returning the Series, DataFrame, or value.
        Note that typeddfs forbids duplicate column names, as well as column names and
        index levels sharing names.
        """
        if isinstance(item, str) and item in self.index.names:
            return self.index.get_level_values(item)
        else:
            return super().__getitem__(item)

    @classmethod
    def of(cls, df, *args, **kwargs) -> __qualname__:
        """
        Construct or convert a DataFrame, returning this type.
        Delegates to :py.meth:`convert` for DataFrames,
        or tries first constructing a DataFrame by calling ``pd.DataFrame(df)``.

        May be overridden to accept more types, such as a string for database lookup.
        For example, ``Customers.of("john")`` could return a DataFrame for a database customer,
        or return the result of ``Customers.convert(...)`` if a DataFrame instance is provided.

        Returns:
            A new DataFrame; see :py.meth:`convert` for more info.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, *args, **kwargs)
        return cls.convert(df)

    def retype(self) -> __qualname__:
        """
        Calls ``self.__class__.convert`` on this DataFrame.
        This is useful to call at the end of a chain of DataFrame functions, where the type
        is preserved but the DataFrame may no longer be valid under this type's rules.
        This can occur because, for performance, typeddfs does not call ``convert`` on most calls.

        Example:
            df = MyDf(data).apply(my_fn, axis=1).retype()  # make sure it's still valid
            df = MyDf(data).groupby(...).retype()  # we maybe changed the index; fix it

        Returns:
            A copy
        """

    @classmethod
    def convert(cls, df: pd.DataFrame) -> __qualname__:
        """
        Converts a vanilla Pandas DataFrame to cls.

        .. note::

            The argument ``df`` will have its ``__class__`` changed to ``cls``
            but will otherwise be unaffected.

        Returns:
            A copy
        """
        df = df.copy()
        df.__class__ = cls
        # noinspection PyTypeChecker
        return df


__all__ = ["BaseDf"]
