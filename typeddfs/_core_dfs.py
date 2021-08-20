import abc
from typing import Any, Iterable, Mapping, Sequence, Union, Generator, Tuple

import pandas as pd
from pandas.core.frame import DataFrame as _InternalDataFrame
from natsort import natsorted, ns

from typeddfs._pretty_dfs import PrettyDf
from typeddfs.df_errors import NoValueError, UnsupportedOperationError, ValueNotUniqueError


class CoreDf(PrettyDf, metaclass=abc.ABCMeta):
    """
    An abstract Pandas DataFrame subclass with additional methods.
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        # noinspection PyTypeChecker
        self.__class__._change(self)

    @classmethod
    def new_df(cls, *args, **kwargs) -> __qualname__:
        """
        Creates a new, somewhat arbitrary DataFrame of this type.

        Arguments:
            *args: These should be narrowed by the overriding method as needed.
            **kwargs: These should be narrowed by the overriding method as needed.

        Raises:
            UnsupportedOperationError: Can be raised if a valid DataFrame is too difficult to create.
            InvalidDfError: May be raised if the type requires specific constraints
                            and did not overload this method to account for them.
                            While programmers using the type should be aware of this possibility,
                            consuming code, in general, should assume that ``new_df`` will always work.
        """
        raise NotImplementedError()

    def iter_row_col(self) -> Generator[Tuple[Tuple[int, int], Any], None, None]:
        """
        Iterates over ``((row, col), value)`` tuples.
        The row and column are the row and column numbers, 1-indexed.
        """
        for row in range(len(self)):
            for col in range(len(self.columns)):
                yield (row, col), self.iat[row, col]

    def only(self, column: str, exclude_na: bool = False) -> Any:
        """
        Returns the single unique value in a column.
        Raises an error if zero or more than one value is in the column.

        Args:
            column: The name of the column
            exclude_na: Exclude null values
        """
        x = set(self[column].unique())
        if exclude_na:
            x = {k for k in x if not pd.isna(k)}
        if len(x) > 1:
            raise ValueNotUniqueError(f"Multiple values for {column}")
        if len(x) == 0:
            raise NoValueError(
                f"No values for {column}" + " (excluding null)" if exclude_na else ""
            )
        return next(iter(x))

    def cfirst(self, cols: Union[str, int, Sequence[str]]) -> __qualname__:
        """
        Returns a new DataFrame with the specified columns appearing first.

        Args:
            cols: A list of columns, or a single column or column index
        """
        if isinstance(cols, str) or isinstance(cols, int):
            cols = [cols]
        return self.__class__._change(self[cols + [c for c in self.columns if c not in cols]])

    def sort_natural(self, column: str, alg: int = ns.INT) -> __qualname__:
        """
        Calls ``natsorted`` on a single column.

        Args:
            column: The name of the (single) column to sort by
            alg: Input as the ``alg`` argument to ``natsorted``
        """
        df = self.vanilla_reset()
        zzz = natsorted([s for s in df[column]], alg=alg)
        df["__sort"] = df[column].map(lambda s: zzz.index(s))
        df.__class__ = self.__class__
        df = df.sort_values("__sort").drop_cols(["__sort"])
        return self.__class__._change(df)

    def sort_natural_index(self, alg: int = ns.INT) -> __qualname__:
        """
        Calls natsorted on this index. Works for multi-index too.

        Args:
            alg: Input as the ``alg`` argument to ``natsorted``
        """
        df = self.copy()
        zzz = natsorted([s for s in df.index], alg=alg)
        df["__sort"] = df.index.map(lambda s: zzz.index(s))
        df.__class__ = self.__class__
        df = df.sort_values("__sort").drop_cols(["__sort"])
        return self.__class__._change(df)

    def drop_cols(self, cols: Union[str, Iterable[str]]) -> __qualname__:
        """
        Drops columns, ignoring those that are not present.

        Args:
            cols: A single column name or a list of column names
        """
        df = self.copy()
        if isinstance(cols, str):
            cols = [cols]
        for c in cols:
            if c in self.columns:
                df = df.drop(c, axis=1)
        return self.__class__._change(df)

    def vanilla_reset(self) -> pd.DataFrame:
        """
        Same as :py.meth:`vanilla`, but resets the index -- but dropping the index if it has no name.
        This means that an effectively index-less dataframe will not end up with an extra column
        called "index".
        """
        if len(self.index_names()) > 0:
            return self.vanilla().reset_index()
        else:
            return self.vanilla().reset_index(drop=True)

    def vanilla(self) -> pd.DataFrame:
        """
        Makes a copy that's a normal Pandas DataFrame.

        Returns:
            A shallow copy with its __class__ set to pd.DataFrame
        """
        df = self.copy()
        df.__class__ = pd.DataFrame
        return df

    def __add__(self, other):
        x = super().__add__(other)
        return self._change_if_df(x)

    def __radd__(self, other):
        x = super().__radd__(other)
        return self._change_if_df(x)

    def __sub__(self, other):
        x = super().__sub__(other)
        return self._change_if_df(x)

    def __rsub__(self, other):
        x = super().__rsub__(other)
        return self._change_if_df(x)

    def __mul__(self, other):
        x = super().__mul__(other)
        return self._change_if_df(x)

    def __rmul__(self, other):
        x = super().__rmul__(other)
        return self._change_if_df(x)

    def __truediv__(self, other):
        x = super().__truediv__(other)
        return self._change_if_df(x)

    def __rtruediv__(self, other):
        x = super().__rtruediv__(other)
        return self._change_if_df(x)

    def __divmod__(self, other):
        x = super().__divmod__(other)
        return self._change_if_df(x)

    def __rdivmod__(self, other):
        x = super().__rdivmod__(other)
        return self._change_if_df(x)

    def __mod__(self, other):
        x = super().__mod__(other)
        return self._change_if_df(x)

    def __rmod__(self, other):
        x = super().__rmod__(other)
        return self._change_if_df(x)

    def __pow__(self, other):
        x = super().__pow__(other)
        return self._change_if_df(x)

    def __rpow__(self, other):
        x = super().__rpow__(other)
        return self._change_if_df(x)

    def drop_duplicates(self, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        return self.__class__._change(super().drop_duplicates(**kwargs))

    def reindex(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        return self.__class__._change(super().reindex(*args, **kwargs))

    def sort_values(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().sort_values(*args, **kwargs)
        return self.__class__._change(df)

    def reset_index(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().reset_index(*args, **kwargs)
        return self.__class__._change(df)

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ) -> __qualname__:
        self._no_inplace(dict(inplace=inplace))
        if len(keys) == 0 and append:
            return self
        elif len(keys) == 0:
            return self.__class__._change(super().reset_index(drop=drop))
        df = super().set_index(
            keys=keys,
            drop=drop,
            append=append,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )
        return self.__class__._change(df)

    def dropna(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().dropna(*args, **kwargs)
        return self.__class__._change(df)

    def fillna(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().fillna(*args, **kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def copy(self, deep: bool = False) -> __qualname__:
        df = super().copy(deep=deep)
        return self.__class__._change(df)

    def assign(self, **kwargs) -> __qualname__:
        df = self.vanilla_reset()
        df = df.assign(**kwargs)
        return self.__class__._change(df)

    def append(self, *args, **kwargs) -> __qualname__:
        df = super().append(*args, **kwargs)
        return self.__class__._change(df)

    def transpose(self, *args, copy: bool = False) -> __qualname__:
        df = super().transpose(*args, copy=copy)
        return self.__class__._change(df)

    # noinspection PyFinal
    def ffill(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().ffill(*args, **kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def bfill(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().bfill(*args, **kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def abs(self) -> __qualname__:
        return self.__class__._change(super().abs())

    def rename(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().rename(*args, **kwargs)
        return self.__class__._change(df)

    def replace(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().replace(*args, **kwargs)
        return self.__class__._change(df)

    def applymap(self, *args, **kwargs) -> __qualname__:
        df = super().applymap(*args, **kwargs)
        return self.__class__._change(df)

    def astype(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().astype(*args, **kwargs)
        return self.__class__._change(df)

    def drop(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().drop(*args, **kwargs)
        return self.__class__._change(df)

    def st(
        self, *array_conditions: Sequence[bool], **dict_conditions: Mapping[str, Any]
    ) -> __qualname__:
        """
        Short for "such that" -- an alternative to slicing with ``.loc``.

        Args:
            array_conditions: Conditions like ``df["score"]<2``
            dict_conditions: Equality conditions, mapping column names to their values (ex ``score=2``)

        Returns:
            A new DataFrame of the same type
        """
        df = self.vanilla()
        for condition in array_conditions:
            df = df.loc[condition]
        for key, value in dict_conditions.items():
            df = df.loc[df[key] == value]
        return self.__class__._change(df)

    @classmethod
    def _convert_typed(cls, df: pd.DataFrame):
        # not great, but works ok
        # if this is a BaseDf, use convert
        # otherwise, just use check_and_change
        if hasattr(cls, "convert"):
            return cls.convert(df)
        else:
            return cls._change(df)

    @classmethod
    def _change_if_df(cls, df):
        if isinstance(df, _InternalDataFrame):
            df.__class__ = cls
        return df

    @classmethod
    def _change(cls, df) -> __qualname__:
        df.__class__ = cls
        return df

    def _no_inplace(self, kwargs):
        if kwargs.get("inplace") is True:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")


__all__ = ["CoreDf"]
