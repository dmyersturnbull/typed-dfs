"""
Defines the superclasses of the types ``TypedDf`` and ``UntypedDf``.
"""
from __future__ import annotations

import abc
from typing import Iterable, Optional

import pandas as pd

from typeddfs.abs_dfs import AbsDf


class BaseDf(AbsDf, metaclass=abc.ABCMeta):
    """
    An abstract DataFrame type that has a way to convert and de-convert.
    A subclass of :class:`typeddfs.abs_dfs.AbsDf`,
    it has methods :meth:`convert` and :meth:`vanilla`.
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
    def of(cls, df, *args, keys: Optional[Iterable[str]] = None, **kwargs) -> __qualname__:
        """
        Construct or convert a DataFrame, returning this type.
        Delegates to :meth:`convert` for DataFrames,
        or tries first constructing a DataFrame by calling ``pd.DataFrame(df)``.
        If ``df`` is a list (``Iterable``) of DataFrames, will call ``pd.concat`` on them;
        for this, ``ignore_index=True`` is passed.
        If the list is empty, will return :meth:`new_df`.

        May be overridden to accept more types, such as a string for database lookup.
        For example, ``Customers.of("john")`` could return a DataFrame for a database customer,
        or return the result of ``Customers.convert(...)`` if a DataFrame instance is provided.
        You may add and process keyword arguments, but keyword args for ``pd.DataFrame.__init__``
        should be passed along to that constructor.

        Args:
            df: A DataFrame, list of DataFrames, or something to be passed to ``pd.DataFrame``.
            keys: Labels for the DataFrames (if passed a sequence of them) to use as attr keys;
                  if None, attrs will be empty (``{}``) if concatenating
            kwargs: Passed to ``pd.DataFrame.__init__``; can be handled directly by this method
                    for specialized construction, database lookup, etc.

        Returns:
            A new DataFrame; see :meth:`convert` for more info.
        """
        dfs = None
        if isinstance(df, Iterable):
            dfs_ = list(df)  # make sure we can iter multiple times
            if all((isinstance(d, pd.DataFrame) for d in df)):
                dfs = dfs_
                if keys is not None:
                    keys = list(keys)
                    if len(keys) != len(dfs):
                        raise ValueError(f"Got {len(dfs)} DataFrames but {len(keys)} keys")
        if dfs is not None:
            if len(dfs) == 0:
                return cls.new_df()
            df = pd.concat(dfs, ignore_index=True, copy=False)
            if keys is not None and any((len(d.attrs) > 0 for d in dfs)):
                df.attrs = {s: d.attrs for s, d in zip(keys, dfs)}
        elif not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, *args, **kwargs)
        return cls.convert(df)

    def retype(self) -> __qualname__:
        """
        Calls ``self.__class__.convert`` on this DataFrame.
        This is useful to call at the end of a chain of DataFrame functions, where the type
        is preserved but the DataFrame may no longer be valid under this type's rules.
        This can occur because, for performance, typeddfs does not call ``convert`` on most calls.

        Examples:
            - ``df = MyDf(data).apply(my_fn, axis=1).retype()  # make sure it's still valid``
            - ``df = MyDf(data).groupby(...).retype()  # we maybe changed the index; fix it``

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
