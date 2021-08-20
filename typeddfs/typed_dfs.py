"""
Defines DataFrames with convenience methods and that enforce invariants.
"""
from __future__ import annotations

import abc
from typing import Sequence, Union

import pandas as pd

from typeddfs._pretty_dfs import PrettyDf
from typeddfs.base_dfs import BaseDf
from typeddfs.df_errors import (
    MissingColumnError,
    UnexpectedColumnError,
    UnexpectedIndexNameError,
    VerificationFailedError,
)
from typeddfs.df_typing import DfTyping, FINAL_DF_TYPING
from typeddfs.untyped_dfs import UntypedDf


class Df(UntypedDf):
    """
    An UntypedDf that shouldn't be overridden.
    """


class TypedDf(BaseDf, metaclass=abc.ABCMeta):
    """
    A concrete BaseFrame that enforces conditions.
    Each subclass has required and reserved (optional) columns and index names.
    They may or may not permit additional columns or index names.

    The constructor will require the conditions to pass but will not rearrange columns and indices.
    To do that, call ``convert``.

    Overrides a number of DataFrame methods that preserve the subclass.
    For example, calling ``df.reset_index()`` will return a ``TypedDf`` of the same type as ``df``.
    If a condition would then fail, call ``untyped()`` first.

    For example, suppose ``MyTypedDf`` has a required index name called "xyz".
    Then this will be fine as long as ``df`` has a column or index name called ``xyz``: ``MyTypedDf.convert(df)``.
    But calling ``MyTypedDf.convert(df).reset_index()`` will fail.
    You can put the column "xyz" back into the index using ``convert``: ``MyTypedDf.convert(df.reset_index())``.
    Or, you can get a plain DataFrame (UntypedDf) back: ``MyTypedDf.convert(df).untyped().reset_index()``.

    To summarize: Call ``untyped()`` before calling something that would result in anything invalid.
    """

    @classmethod
    def get_typing(cls) -> DfTyping:
        return FINAL_DF_TYPING  # just a default -- should be overridden

    @classmethod
    def convert(cls, df: pd.DataFrame) -> __qualname__:
        """
        Converts a vanilla Pandas DataFrame (or any subclass) to ``cls``.
        Explicitly sets the new copy's __class__ to cls.
        Rearranges the columns and index names.
        For example, if a column in ``df`` is in ``self.reserved_index_names()``, it will be moved to the index.

        The new index names will be, in order:
            - ``required_index_names()``, in order
            - ``reserved_index_names()``, in order
            - any extras in ``df``, if ``more_indices_allowed`` is True

        Similarly, the new columns will be, in order:
            - ``required_columns()``, in order
            - ``reserved_columns()``, in order
            - any extras in ``df`` in the original, if ``more_columns_allowed`` is True

        NOTE:
            Any column called ``index`` or ``level_0`` will be dropped automatically.

        Args:
            df: The Pandas DataFrame or member of cls; will have its __class_ change but will otherwise not be affected

        Returns:
            A copy

        Raises:
            InvalidDfError: If a condition such as a required column or symmetry fails (specific subclasses)
            TypeError: If ``df`` is not a DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Can't convert {type(df)} to {cls.__name__}")
        # first always reset the index so we can manage what's in the index vs columns
        # index_names() will return [] if no named indices are found
        df = df.copy()
        df.__class__ = PrettyDf
        original_index_names = df.index_names()
        df = df.reset_index()
        # remove trash columns
        df.__class__ = cls
        t = cls.get_typing()
        df = df.drop_cols(["index", "level_0", "Unnamed: 0"])  # these MUST be dropped
        df = df.drop_cols(t.columns_to_drop)
        # now let's convert the dtypes
        for c, dt in t.auto_dtypes.items():
            if c in df.columns:
                df[c] = df[c].astype(dt)
        # set index columns and used preferred order
        new_index_names = []
        # here we keep the order of reserved
        for c in list(t.required_index_names) + list(t.reserved_index_names):
            if c not in new_index_names and c in df.columns:
                new_index_names.append(c)
        # if the original index names are reserved columns, add them to the columns
        # otherwise, stick them at the end of the index
        all_reserved = t.known_names
        # if it doesn't get added in here, it just stays in the columns -- which will be kept
        new_index_names.extend([s for s in original_index_names if s not in all_reserved])
        if len(new_index_names) > 0:  # raises an error otherwise
            df = df.set_index(new_index_names)
        # now set the regular column order
        new_columns = []  # re-use the same variable name
        for c in list(t.required_columns) + list(t.reserved_columns):
            if c not in new_columns and c in df.columns:
                new_columns.append(c)
        # set the index/column series name(s)
        df: BaseDf = df
        col_series = t.column_series_name
        if col_series is not False:
            if col_series is True:
                col_series = None
            df.columns.name = col_series
        ind_series = t.index_series_name
        if df.is_multindex() and ind_series is not False:
            if ind_series is True:
                ind_series = None
            df.index.name = ind_series
        # this lets us keep whatever extra columns
        df = df.cfirst(new_columns)
        # call post-processing
        if t.post_processing is not None:
            df = t.post_processing(df)
        # check that it has every required column and index name
        cls._check(df)
        # now change the class
        df.__class__ = cls
        return df

    @classmethod
    def new_df(cls, *, reserved: Union[bool, Sequence[str]] = False) -> __qualname__:
        """
        Returns a DataFrame that is empty but has the correct columns and indices.

        Arguments:
            reserved: Include reserved index/column names as well as required.
                      If True, adds all reserved index levels and columns;
                      You can also specify the exact list of columns and index names.

        Raises:
            InvalidDfError: If a function in ``verifications`` fails (returns False or a string).
        """
        t = cls.get_typing()
        if reserved:
            req = t.known_names
        else:
            req = [*t.required_index_names, *t.required_columns]
        df = pd.DataFrame({r: [] for r in req})
        return cls.convert(df)

    def untyped(self) -> UntypedDf:
        """
        Makes a copy that's an UntypedDf.
        It won't have enforced requirements but will still have the convenience functions.

        Returns:
            A shallow copy with its __class__ set to an UntypedDf

        See:
            :py.meth:`vanilla`
        """
        df = self.copy()
        df.__class__ = Df
        return df

    def meta(self) -> __qualname__:
        """
        Drops the columns, returning only the index but as the same type.

        Returns:
            A copy

        Raises:
            InvalidDfError: If the result does not pass the typing of this class
        """
        if len(self.columns) == 0:
            return self
        else:
            df = self[[self.columns[0]]]
            df = df.drop(self.columns[0], axis=1)
            return self.__class__.convert(df)

    @classmethod
    def _check(cls, df) -> None:
        cls._check_has_required(df)
        cls._check_has_unexpected(df)
        for req in cls.get_typing().verifications:
            value = req(df)
            if value is not None:
                raise VerificationFailedError(value)

    @classmethod
    def _check_has_required(cls, df: pd.DataFrame) -> None:
        t = cls.get_typing()
        for c in set(t.required_index_names):
            if c not in set(df.index.names):
                raise MissingColumnError(
                    f"Missing index name {c} (indices are: {set(df.index.names)}; cols are: {set(df.columns.names)}))"
                )
        for c in set(t.required_columns):
            if c not in set(df.columns):
                raise MissingColumnError(
                    f"Missing column {c} (cols are: {set(df.columns.names)}; indices are: {set(df.index.names)})"
                )

    @classmethod
    def _check_has_unexpected(cls, df: pd.DataFrame) -> None:
        df = PrettyDf(df)
        t = cls.get_typing()
        if not t.more_columns_allowed:
            for c in df.column_names():
                if c not in t.required_columns and c not in t.reserved_columns:
                    raise UnexpectedColumnError(f"Unexpected column {c}")
        if not t.more_indices_allowed:
            for c in df.index_names():
                if c not in t.required_index_names and c not in t.reserved_index_names:
                    raise UnexpectedIndexNameError(f"Unexpected index name {c}")


__all__ = ["TypedDf"]
