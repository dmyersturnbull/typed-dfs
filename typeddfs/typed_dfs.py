"""
Defines DataFrames with convenience methods and that enforce invariants.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Type, Any, Mapping

import pandas as pd

from typeddfs.base_dfs import BaseDf
from typeddfs.df_errors import (
    AsymmetricDfError,
    ExtraConditionFailedError,
    InvalidDfError,
    MissingColumnError,
    UnexpectedColumnError,
    UnexpectedIndexNameError,
)
from typeddfs._pretty_dfs import PrettyDf
from typeddfs.untyped_dfs import UntypedDf


class Df(UntypedDf):
    """
    An UntypedDf that shouldn't be overridden.
    """


class TypedDf(BaseDf):
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
        df = df.drop_cols(["index", "level_0"])  # these MUST be dropped
        df = df.drop_cols(cls.columns_to_drop())
        # now let's convert the dtypes
        for c, dt in cls.auto_dtypes().items():
            if c in df.columns:
                df[c] = df[c].astype(dt)
        # set index columns and used preferred order
        new_index_names = []
        # here we keep the order of reserved
        for c in list(cls.required_index_names()) + list(cls.reserved_index_names()):
            if c not in new_index_names and c in df.columns:
                new_index_names.append(c)
        # if the original index names are reserved columns, add them to the columns
        # otherwise, stick them at the end of the index
        all_reserved = cls.known_names()
        # if it doesn't get added in here, it just stays in the columns -- which will be kept
        new_index_names.extend([s for s in original_index_names if s not in all_reserved])
        if len(new_index_names) > 0:  # raises an error otherwise
            df = df.set_index(new_index_names)
        # now set the regular column order
        new_columns = []  # re-use the same variable name
        for c in list(cls.required_columns()) + list(cls.reserved_columns()):
            if c not in new_columns and c in df.columns:
                new_columns.append(c)
        # this lets us keep whatever extra columns
        df = df.cfirst(new_columns)
        # call post-processing
        if cls.post_processing() is not None:
            df = cls.post_processing()(df)
        # check that it has every required column and index name
        cls._check(df)
        # now change the class
        df.__class__ = cls
        return df

    def untyped(self) -> UntypedDf:
        """
        Makes a copy that's an UntypedDf.
        It won't have enforced requirements but will still have the convenience functions.

        Returns:
            A shallow copy with its __class__ set to an UntypedDf

        See:
            ``vanilla``
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
    def is_valid(cls, df: pd.DataFrame) -> bool:
        """
        Returns True if all of the required conditions pass.
        This is provided as a sanity check only: It should always return True.
        """
        try:
            cls.convert(df)
            return True
        except InvalidDfError:
            return False

    @classmethod
    def more_columns_allowed(cls) -> bool:
        """
        Returns whether the DataFrame allows columns that are not reserved or required.
        """
        return True

    @classmethod
    def more_indices_allowed(cls) -> bool:
        """
        Returns whether the DataFrame allows index names that are not reserved or required.
        """
        return True

    @classmethod
    def required_columns(cls) -> Sequence[str]:
        """
        Returns the list of required column names.
        """
        return []

    @classmethod
    def reserved_columns(cls) -> Sequence[str]:
        """
        Returns the list of reserved (optional) column names.
        """
        return []

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        """
        Returns the list of required column names.
        """
        return []

    @classmethod
    def reserved_index_names(cls) -> Sequence[str]:
        """
        Returns the list of reserved (optional) index names.
        """
        return []

    @classmethod
    def known_column_names(cls) -> Sequence[str]:
        """
        Returns all columns that are required or reserved.
        The sort order positions required columns first.
        """
        return [*cls.required_columns(), *cls.reserved_columns()]

    @classmethod
    def known_index_names(cls) -> Sequence[str]:
        """
        Returns all index names that are required or reserved.
        The sort order positions required columns first.
        """
        return [*cls.required_index_names(), *cls.reserved_index_names()]

    @classmethod
    def known_names(cls) -> Sequence[str]:
        """
        Returns all index and column names that are required or reserved.
        The sort order is: required index, reserved index, required columns, reserved columns.
        """
        return [
            *cls.required_index_names(),
            *cls.reserved_index_names(),
            *cls.required_columns(),
            *cls.reserved_columns(),
        ]

    @classmethod
    def auto_dtypes(cls) -> Mapping[str, Type[Any]]:
        """
        Returns a mapping from column/index names to the expected dtype.
        These are used via ``pd.Series.as_type`` for automatic conversion.
        An error will be raised if a ``as_type`` call fails.
        Note that Pandas frequently just does not perform the conversion,
        rather than raising an error.
        The keys should be contained in ``known_names``, but this is not strictly required.
        """
        return {}

    @classmethod
    def must_be_symmetric(cls) -> bool:
        """
        Returns whether the (single only) index values must match the column names.
        """
        return False

    @classmethod
    def columns_to_drop(cls) -> Sequence[str]:
        """
        Returns the list of columns that are automatically dropped by ``convert``.
        This does NOT include "level_0" and "index, which are ALWAYS dropped.
        """
        return []

    @classmethod
    def post_processing(cls) -> Optional[Callable[[BaseDf], Optional[BaseDf]]]:
        """
        A function to be called at the final stage of ``convert``.
        It is called immediately before ``extra_conditions`` are checked.
        The function takes a copy of the input ``BaseDf`` and returns a new copy.

        Note:
            Although a copy is passed as input, the function should not modify it.
            Technically, doing so will cause problems only if the DataFrame's internal values
            are modified. The value passed is a *shallow* copy (see ``pd.DataFrame.copy``).
        """
        return None

    @classmethod
    def extra_conditions(cls) -> Sequence[Callable[[BaseDf], Optional[str]]]:
        """
        Additional requirements for the DataFrame to be conformant.

        Returns:
            A sequence of conditions that map the DF to None if the condition passes,
            or the string of an error message if it fails
        """
        return []

    @classmethod
    def _check(cls, df) -> None:
        cls._check_has_required(df)
        cls._check_has_unexpected(df)
        if cls.must_be_symmetric():
            cls._check_symmetric(df)
        for req in cls.extra_conditions():
            value = req(df)
            if value is not None:
                raise ExtraConditionFailedError(value)

    @classmethod
    def _check_has_required(cls, df: pd.DataFrame) -> None:
        for c in set(cls.required_index_names()):
            if c not in set(df.index.names):
                raise MissingColumnError(
                    f"Missing index name {c} (indices are: {set(df.index.names)}; columns are: {set(df.columns.names)}))"
                )
        for c in set(cls.required_columns()):
            if c not in set(df.columns):
                raise MissingColumnError(
                    f"Missing column {c} (columns are: {set(df.columns.names)}; indices are: {set(df.index.names)})"
                )

    @classmethod
    def _check_has_unexpected(cls, df: pd.DataFrame) -> None:
        df = PrettyDf(df)
        if not cls.more_columns_allowed():
            for c in df.column_names():
                if c not in cls.required_columns() and c not in cls.reserved_columns():
                    raise UnexpectedColumnError(f"Unexpected column {c}")
        if not cls.more_indices_allowed():
            for c in df.index_names():
                if c not in cls.required_index_names() and c not in cls.reserved_index_names():
                    raise UnexpectedIndexNameError(f"Unexpected index name {c}")

    @classmethod
    def _check_symmetric(cls, df: pd.DataFrame) -> None:
        if isinstance(df.index, pd.MultiIndex):
            raise AsymmetricDfError(
                f"The {cls.__name__} cannot be symmetric because it's multi-index"
            )
        if list(df.index) != list(df.columns):
            raise AsymmetricDfError(
                f"The indices are {list(df.index)} but the rows are {list(df.columns)}"
            )

    @classmethod
    def _lines_files_apply(cls) -> bool:
        return len(cls.known_names()) == 1


__all__ = ["TypedDf"]
