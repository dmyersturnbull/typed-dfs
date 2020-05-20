from __future__ import annotations
from pathlib import Path, PurePath
from warnings import warn
from typing import Sequence, Union, Optional, Callable
import pandas as pd

from typeddfs.base_dfs import (
    BaseDf,
    MissingColumnError,
    UnexpectedColumnError,
    ExtraConditionError,
    AsymmetricDfError,
)
from typeddfs.untyped_dfs import UntypedDf

PathLike = Union[str, PurePath]


class Sentinel:
    pass


_sentinel = Sentinel()


class Df(UntypedDf):
    """
    An UntypedDf that shouldn't be overridden.
    """


class TypedDf(BaseDf):
    """
    A concrete BaseFrame that has required columns and index names.
    """

    @classmethod
    def convert(cls, df: pd.DataFrame, require_full: bool = _sentinel) -> TypedDf:
        """
        Converts a vanilla Pandas DataFrame to cls.
        Sets the index appropriately, permitting the required columns and index names to be either columns or index names.
        Explicitly sets the new copy's __class__ to cls.
        :param df: The Pandas DataFrame or member of cls; will have its __class_ change but will otherwise not be affected
        :param require_full: Raise a InvalidExtendedDataFrameError if a required column or index name is missing
        :return: A copy
        """
        if require_full is _sentinel:
            require_full = True
        else:
            warn(
                "Passing require_full to OrganizingFrame.convert is deprecated.",
                DeprecationWarning,
            )
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Can't convert {} to {}".format(type(df), cls.__name__))
        # first always reset the index so we can manage what's in the index vs columns
        df = df.reset_index()
        # remove trash columns
        df.__class__ = cls
        df = df.drop_cols(["index", "level_0"])  # these MUST be dropped
        df = df.drop_cols(cls.columns_to_drop())
        # set index columns and used preferred order
        res = []
        # here we keep the order of reserved if it contains all of required
        for c in list(cls.required_index_names()) + list(cls.reserved_index_names()):
            if c not in res and c in df.columns:
                res.append(c)
        if len(res) > 0:  # raises an error otherwise
            df = df.set_index(res)
        # now set the regular column order
        res = []  # re-use the same variable name
        for c in list(cls.required_columns()) + list(cls.reserved_columns()):
            if c not in res and c in df.columns:
                res.append(c)
        # check that it has every required column and index name
        cls._check(df, require_full=require_full)
        # now change the class
        df.__class__ = cls
        df = df.cfirst(res)
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

    @classmethod
    def read_csv(cls, path: PathLike, *args, **kwargs):
        if "index_col" in kwargs:
            warn("index_col={} in OrganizingFrame.read_csv is ignored".format(kwargs["index_col"]))
            kwargs = {k: v for k, v in kwargs if k != "index_col"}
        df = pd.read_csv(Path(path), index_col=False, **kwargs)
        return cls.convert(df)

    def to_csv(self, path: PathLike, *args, **kwargs) -> Optional[str]:
        if "index" in kwargs:
            warn("index={} in OrganizingFrame.to_csv is ignored".format(kwargs["index"]))
        df = self.vanilla().reset_index()
        return df.to_csv(path, index=False)

    @classmethod
    def is_valid(cls, df: pd.DataFrame, require_full: bool = True) -> bool:
        try:
            cls.convert(df, require_full=require_full)
            return True
        except (MissingColumnError, UnexpectedColumnError):
            return False

    @classmethod
    def more_columns_allowed(cls) -> bool:
        return True

    @classmethod
    def more_indices_allowed(cls) -> bool:
        return True

    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return []

    @classmethod
    def reserved_columns(cls) -> Sequence[str]:
        return []

    @classmethod
    def reserved_index_names(cls) -> Sequence[str]:
        return []

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return []

    @classmethod
    def must_be_symmetric(cls) -> bool:
        return False

    @classmethod
    def columns_to_drop(cls) -> Sequence[str]:
        return []

    @classmethod
    def extra_conditions(cls) -> Sequence[Callable[[pd.DataFrame], Optional[str]]]:
        """
        Additional requirements for the DataFrame to be conformant.

        Returns:
            A sequence of conditions that map the DF to None if the condition passes, or the string of an error message if it fails
        """
        return []

    @classmethod
    def _check(cls, df, require_full):
        if require_full:
            cls._check_has_required(df)
        cls._check_has_unexpected(df)
        if cls.must_be_symmetric():
            cls._check_symmetric(df)
        for req in cls.extra_conditions():
            value = req(df)
            if value is not None:
                raise ExtraConditionError(value)

    @classmethod
    def _check_has_required(cls, df: pd.DataFrame):
        for c in set(cls.required_index_names()):
            if c not in set(df.index.names):
                raise MissingColumnError("Missing index name {}".format(c))
        for c in set(cls.required_columns()):
            if c not in set(df.columns):
                raise MissingColumnError("Missing column {}".format(c))

    @classmethod
    def _check_has_unexpected(cls, df: pd.DataFrame):
        if not cls.more_columns_allowed():
            for c in df.columns:
                if c not in cls.required_columns() and c not in cls.reserved_columns():
                    raise UnexpectedColumnError("Unexpected column {}".format(c))
        if not cls.more_indices_allowed() and list(df.index.names) != ["None"]:
            for c in df.index.names:
                if (
                    c is None
                    or c not in cls.required_index_names()
                    and c not in cls.reserved_index_names()
                ):
                    raise UnexpectedColumnError("Unexpected column {}".format(c))

    @classmethod
    def _check_symmetric(cls, df: pd.DataFrame):
        if isinstance(df.index, pd.MultiIndex):
            raise AsymmetricDfError(
                "The {} cannot be symmetric because it's multi-index".format(cls.__name__)
            )
        if list(df.index) == [None]:
            raise AsymmetricDfError(
                "The {} cannot be symmetric because it lacks a named index".format(cls.__name__)
            )
        if list(df.index) != list(df.columns):
            raise AsymmetricDfError(
                "The indices are {} but the rows are {}".format(list(df.index), list(df.columns))
            )


__all__ = ["TypedDf"]
