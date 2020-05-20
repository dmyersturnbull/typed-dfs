from __future__ import annotations
from pathlib import Path, PurePath
import abc
from warnings import warn
from typing import Sequence, List, Any, Union, Iterable, Optional, Callable, Tuple as Tup
import pandas as pd
from natsort import ns, natsorted
from pandas.core.frame import DataFrame as _InternalDataFrame

PathLike = Union[str, PurePath]


class InvalidDfError(Exception):
    pass


class ExtraConditionError(InvalidDfError):
    pass


class MissingColumnError(InvalidDfError):
    pass


class AsymmetricDfError(InvalidDfError):
    pass


class UnexpectedColumnError(InvalidDfError):
    pass


class Sentinel:
    pass


_sentinel = Sentinel()


class PrettyDf(_InternalDataFrame, metaclass=abc.ABCMeta):
    """
    A DataFrame with an overridden _repr_html_ that shows the dimensions at the top.
    """

    @property
    def _constructor_expanddim(self):
        # this raises a NotImplementedError in _InternalDataFrame, so let's override it here to prevent tools and IDEs from complaining
        raise ValueError()

    def _repr_html_(self):
        """
        Renders HTML for display() in Jupyter notebooks.
        Jupyter automatically uses this function.
        Returns:
            Just a string, which will be wrapped in HTML
        """
        return "<strong>{}: {}</strong>\n{}".format(
            self.__class__.__name__, self._dims(), super()._repr_html_(), len(self)
        )

    def _dims(self) -> str:
        """
        Returns:
            A text description of the dimensions of this DataFrame
        """
        # we could handle multi-level columns, but they're quite rare, and the number of rows is probably obvious when looking at it
        if len(self.index.names) > 1:
            return "{} rows × {} columns, {} index columns".format(
                len(self), len(self.columns), len(self.index.names)
            )
        else:
            return "{} rows × {} columns".format(len(self), len(self.columns))


class AbsDf(PrettyDf, metaclass=abc.ABCMeta):
    """
    An abstract Pandas DataFrame subclass with additional methods.
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self._check_and_change(self)

    def column_names(self) -> List[str]:
        return list(self.columns)

    def index_names(self) -> List[str]:
        lst = list(self.index.names)
        if lst == [None]:
            return []
        else:
            return lst

    def is_multindex(self) -> bool:
        return isinstance(self.index, pd.MultiIndex)

    def n_rows(self) -> int:
        return len(self)

    def n_columns(self) -> int:
        return len(self.columns)

    def n_indices(self) -> int:
        return len(self.index.names)

    def only(self, column: str) -> Any:
        """
        Returns the single unique value in a column.
        Raises an error if zero or more than one value is in the column.
        :param column: The name of the column
        :return: The value
        """
        x = self[column].unique()
        if len(x) > 0:
            raise ValueError("Multiple values for {}".format(column))
        return x[0]

    def cfirst(self, cols: Union[str, int, Sequence[str]]):
        """
        Returns a new DataFrame with the specified columns appearing first.
        :param cols: A list of columns
        :return: A non-copy
        """
        if isinstance(cols, str) or isinstance(cols, int):
            cols = [cols]
        if len(self) == 0:  # will break otherwise
            return self
        else:
            return self._check_and_change(self[cols + [c for c in self.columns if c not in cols]])

    def sort_natural(self, column: str, alg: int = ns.INT):
        """
        Calls natsorted on a single column.
        :param alg Input as the ``alg`` argument to ``natsorted``.
        """
        df = self.copy().reset_index()
        zzz = natsorted([s for s in df[column]], alg=alg)
        df["__sort"] = df[column].map(lambda s: zzz.index(s))
        df.__class__ = self.__class__
        df = df.sort_values("__sort").drop_cols(
            ["__sort"]
        )  # .drop_cols(['__sort', 'level_0', 'index'])
        return self._check_and_change(df)

    def sort_natural_index(self, alg: int = ns.INT):
        """
        Calls natsorted on this index.
        (Works for multi-index too.)
        :param alg Input as the ``alg`` argument to ``natsorted``.
        """
        df = self.copy().reset_index()
        zzz = natsorted([s for s in df.index], alg=alg)
        df["__sort"] = df.index.map(lambda s: zzz.index(s))
        df.__class__ = self.__class__
        df = df.sort_values("__sort").drop_cols(
            ["__sort"]
        )  # .drop_cols(['__sort', 'level_0', 'index'])
        return self._check_and_change(df)

    def drop_cols(self, cols: Union[str, Iterable[str]]):
        """
        Drops columns, ignoring those that are not present.
        """
        df = self.copy()
        if isinstance(cols, str):
            cols = [cols]
        for c in cols:
            if c in self.columns:
                df = df.drop(c, axis=1)
        return self._check_and_change(df)

    @classmethod
    def read_csv(cls, *args, **kwargs):
        return cls._check_and_change(pd.read_csv(*args, **kwargs))

    # noinspection PyMethodOverriding
    def to_csv(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        return self.to_vanilla().to_csv(path_or_buf, *args, **kwargs)

    @classmethod
    def read_hdf(cls, *args, key: str = "df", **kwargs):
        """
        Reads from HDF with ``key`` as the default, converting to this type.
        """
        # noinspection PyTypeChecker
        df: pd.DataFrame = pd.read_hdf(*args, key=key, **kwargs)
        return cls._check_and_change(df)

    def to_hdf(self, path: PathLike, key: str = "df", **kwargs) -> None:
        """
        Writes to HDF with ``key`` as the default. Calling pd.to_hdf on this would error.
        """
        path = str(Path(path))
        x = self.to_vanilla()
        x.to_hdf(path, key, **kwargs)

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

    def vanilla(self) -> pd.DataFrame:
        """
        Makes a copy that's a normal Pandas DataFrame.
        You might want ``untyped`` instead.
        Returns:
            A shallow copy with its __class__ set to pd.DataFrame
        See:
            ``untyped``
        """
        df = self.copy()
        df.__class__ = pd.DataFrame
        return df

    @classmethod
    def _make_vanilla(cls, df: AbsDf) -> pd.DataFrame:
        """
        Make vanilla in-place.
        DEPRECATED. Use ``vanilla`` instead.
        Args:
            df: The ConvertibleFrame or member of cls; will have its __class_ change but will otherwise not be affected
        Returns:
            A shallow copy with its __class__ set to pd.DataFrame
        """
        df = df
        df.__class__ = pd.DataFrame
        return df

    def drop_duplicates(self, **kwargs):
        return self._check_and_change(super().drop_duplicates(**kwargs))

    def reindex(self, *args, **kwargs):
        return self._check_and_change(super().reindex(*args, **kwargs))

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        **kwargs,
    ):
        return self._check_and_change(
            super().sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                inplace=inplace,
                kind=kind,
                na_position=na_position,
                **kwargs,
            )
        )

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
        return self._check_and_change(
            super().reset_index(
                level=level, drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill,
            )
        )

    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        return self._check_and_change(
            super().set_index(
                keys=keys,
                drop=drop,
                append=append,
                inplace=inplace,
                verify_integrity=verify_integrity,
            )
        )

    def dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False):
        return self._check_and_change(
            super().dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace)
        )

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
        **kwargs,
    ):
        return self._check_and_change(
            super().fillna(
                value=value,
                method=method,
                axis=axis,
                inplace=inplace,
                limit=limit,
                downcast=downcast,
                **kwargs,
            )
        )

    def copy(self, deep: bool = False):
        return self._check_and_change(super().copy(deep=deep))

    def append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        return self._check_and_change(
            super().append(
                other, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort
            )
        )

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        return self._check_and_change(
            super().ffill(axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        )

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        return self._check_and_change(
            super().bfill(axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        )

    def abs(self):
        return self._check_and_change(super().abs())

    def rename(self, *args, **kwargs):
        return self._check_and_change(super().rename(*args, **kwargs))

    def replace(
        self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method="pad",
    ):
        return self._check_and_change(
            super().replace(
                to_replace=to_replace,
                value=value,
                inplace=inplace,
                limit=limit,
                regex=regex,
                method=method,
            )
        )

    def applymap(self, func):
        return self._check_and_change(super().applymap(func))

    def astype(self, dtype, copy=True, errors="raise"):
        return self._check_and_change(super().astype(dtype=dtype, copy=copy, errors=errors))

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        return self._check_and_change(
            super().drop(
                labels=labels,
                axis=axis,
                index=index,
                columns=columns,
                level=level,
                inplace=inplace,
                errors=errors,
            )
        )

    @classmethod
    def _check_and_change(cls, df):
        df.__class__ = cls
        return df

    @classmethod
    def _change(cls, df):
        df.__class__ = cls
        return df


class BaseDf(AbsDf, metaclass=abc.ABCMeta):
    """
    An extended DataFrame with convert() and vanilla() methods.
    """

    @classmethod
    def convert(cls, df: pd.DataFrame):
        """
        Converts a vanilla Pandas DataFrame to cls.
        Sets the index appropriately, permitting the required columns and index names to be either columns or index names.
        :param df: The Pandas DataFrame or member of cls; will have its __class_ change but will otherwise not be affected
        :return: A non-copy
        """
        df = df.copy()
        df.__class__ = cls
        return df


class UntypedDf(BaseDf):
    """
    A concrete BaseFrame that does not require special columns.
    Overrides a number of DataFrame methods to convert before returning.
    """

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.index.names:
            return self.index.get_level_values(item)
        else:
            return super().__getitem__(item)

    @classmethod
    def read_csv(cls, *args, **kwargs):
        """
        Reads from CSV, converting to this type.
        Using to_csv() and read_csv() from BaseFrame, this property holds:
            ```
            df.to_csv(path)
            df.__class__.read_csv(path) == df
            ```
        """
        index_col = kwargs.get("index_col", False)
        df = pd.read_csv(*args, index_col=index_col)
        return cls._check_and_change(df)

    def to_csv(self, path: PathLike, *args, **kwargs) -> Optional[str]:
        """
        Writes CSV.
        Using to_csv() and read_csv() from BaseFrame, this property holds:
            ```
            df.to_csv(path)
            df.__class__.read_csv(path) == df
            ```
        """
        if "index" in kwargs:
            return super().to_csv(path, *args, **kwargs)
        else:
            df = self.to_vanilla().reset_index(drop=list(self.index.names) == [None])
            return df.to_csv(path, *args, index=False, **kwargs)


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
        df = self.to_vanilla().reset_index()
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
            for c in set(cls.required_index_names()):
                if c not in set(df.index.names):
                    raise MissingColumnError("Missing index name {}".format(c))
            for c in set(cls.required_columns()):
                if c not in set(df.columns):
                    raise MissingColumnError("Missing column {}".format(c))
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
        if cls.must_be_symmetric():
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
                    "The indices are {} but the rows are {}".format(
                        list(df.index), list(df.columns)
                    )
                )
        for req in cls.extra_conditions():
            value = req(df)
            if value is not None:
                raise InvalidDfError(value)


__all__ = [
    "BaseDf",
    "UntypedDf",
    "TypedDf",
    "InvalidDfError",
    "MissingColumnError",
    "UnexpectedColumnError",
    "AsymmetricDfError",
    "ExtraConditionError",
]
