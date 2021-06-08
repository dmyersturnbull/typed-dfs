import abc
from typing import Any, Union, Sequence, Iterable, Optional, Mapping

import pandas as pd
from natsort import natsorted, ns

from typeddfs._pretty_dfs import PrettyDf
from typeddfs.df_errors import ValueNotUniqueError, NoValueError, UnsupportedOperationError


class CoreDf(PrettyDf, metaclass=abc.ABCMeta):
    """
    An abstract Pandas DataFrame subclass with additional methods.
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        # noinspection PyTypeChecker
        self.__class__._check_and_change(self)

    def only(self, column: str, exclude_na: bool = False) -> Any:
        """
        Returns the single unique value in a column.
        Raises an error if zero or more than one value is in the column.

        Args:
            column: The name of the column
            exclude_na: Exclude null values

        Returns:
            The value
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
            cols: A list of columns

        Returns:
            A non-copy
        """
        if isinstance(cols, str) or isinstance(cols, int):
            cols = [cols]
        return self.__class__._check_and_change(
            self[cols + [c for c in self.columns if c not in cols]]
        )

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
        return self.__class__._check_and_change(df)

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
        return self.__class__._check_and_change(df)

    def drop_cols(self, cols: Union[str, Iterable[str]]) -> __qualname__:
        """
        Drops columns, ignoring those that are not present.

        Args:
            A single column name or a list of column names

        Returns:
            The new dataframe, which has the same class
        """
        df = self.copy()
        if isinstance(cols, str):
            cols = [cols]
        for c in cols:
            if c in self.columns:
                df = df.drop(c, axis=1)
        return self.__class__._check_and_change(df)

    def vanilla_reset(self) -> pd.DataFrame:
        """
        Same as ``vanilla``, but resets the index -- but dropping the index if it has no name.
        This means that an effectively index-less dataframe will not end up with an extra column
        called 'index'.
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

    def drop_duplicates(self, **kwargs) -> __qualname__:
        if "inplace" in kwargs:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(super().drop_duplicates(**kwargs))

    def reindex(self, *args, **kwargs) -> __qualname__:
        if "inplace" in kwargs:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(super().reindex(*args, **kwargs))

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        **kwargs,
    ) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
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

    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
            super().reset_index(
                level=level,
                drop=drop,
                inplace=inplace,
                col_level=col_level,
                col_fill=col_fill,
            )
        )

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        if len(keys) == 0 and append:
            return self
        elif len(keys) == 0:
            # TODO what happens to the other args?
            return self.__class__._check_and_change(super().reset_index(drop=drop))
        return self.__class__._check_and_change(
            super().set_index(
                keys=keys,
                drop=drop,
                append=append,
                inplace=inplace,
                verify_integrity=verify_integrity,
            )
        )

    def dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
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
    ) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
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

    # noinspection PyFinal
    def copy(self, deep: bool = False) -> __qualname__:
        return self.__class__._check_and_change(super().copy(deep=deep))

    def assign(self, **kwargs) -> __qualname__:
        df = self.vanilla_reset()
        df = df.assign(**kwargs)
        return self.__class__._convert(df)

    def append(self, other, ignore_index=False, verify_integrity=False, sort=False) -> __qualname__:
        return self.__class__._check_and_change(
            super().append(
                other, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort
            )
        )

    # noinspection PyFinal
    def ffill(self, axis=None, inplace=False, limit=None, downcast=None) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
            super().ffill(axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        )

    # noinspection PyFinal
    def bfill(self, axis=None, inplace=False, limit=None, downcast=None) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
            super().bfill(axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        )

    # noinspection PyFinal
    def abs(self) -> __qualname__:
        return self.__class__._check_and_change(super().abs())

    def rename(self, *args, **kwargs) -> __qualname__:
        if "inplace" in kwargs:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(super().rename(*args, **kwargs))

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
            super().replace(
                to_replace=to_replace,
                value=value,
                inplace=inplace,
                limit=limit,
                regex=regex,
                method=method,
            )
        )

    def applymap(self, func, na_action: Optional[str] = None) -> __qualname__:
        return self.__class__._check_and_change(super().applymap(func, na_action=na_action))

    def astype(self, dtype, copy=True, errors="raise") -> __qualname__:
        return self.__class__._check_and_change(
            super().astype(dtype=dtype, copy=copy, errors=errors)
        )

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ) -> __qualname__:
        if inplace:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")
        return self.__class__._check_and_change(
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
        return self.__class__._check_and_change(df)

    @classmethod
    def _convert(cls, df: pd.DataFrame):
        # not great, but works ok
        # if this is a BaseDf, use convert
        # otherwise, just use check_and_change
        if hasattr(cls, "convert"):
            return cls.convert(df)
        else:
            return cls._check_and_change(df)

    @classmethod
    def _check_and_change(cls, df) -> __qualname__:
        df.__class__ = cls
        return df

    @classmethod
    def _change(cls, df) -> __qualname__:
        df.__class__ = cls
        return df


__all__ = ["CoreDf"]
