"""
Mixin with misc new DataFrame methods.
"""
from typing import Any, Generator, Iterable, Mapping, Sequence, Set, Tuple, Union

import pandas as pd
from natsort import natsorted

from typeddfs.df_errors import NoValueError, ValueNotUniqueError
from typeddfs.utils import Utils


class _NewMethodsMixin:
    def strip_control_chars(self) -> __qualname__:
        """
        Removes all control characters (Unicode group 'C') from all string-typed columns.
        """
        df = self.vanilla_reset()
        for c in df.columns:
            if Utils.is_string_dtype(df[c]):
                df[c] = df[c].map(Utils.strip_control_chars)
        return self.__class__._convert_typed(df)

    def set_attrs(self, **attrs) -> __qualname__:
        """
        Sets ``pd.DataFrame.attrs``, returning a copy.
        """
        df = self.copy()
        df.attrs.update(attrs)
        return df

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
            exclude_na: Exclude None/pd.NA values
        """
        x = set(self[column].unique())
        if exclude_na:
            x = {k for k in x if not pd.isna(k)}
        if len(x) > 1:
            raise ValueNotUniqueError(f"Multiple values for {column}", key=column, values=set(x))
        if len(x) == 0:
            raise NoValueError(
                f"No values for {column}" + " (excluding null)" if exclude_na else "", key=column
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

    def sort_natural(
        self, column: str, *, alg: Union[None, int, Set[str]] = None, reverse: bool = False
    ) -> __qualname__:
        """
        Calls ``natsorted`` on a single column.

        Args:
            column: The name of the (single) column to sort by
            alg: Input as the ``alg`` argument to ``natsorted``
                 If ``None``, the "best" algorithm is chosen from the dtype of ``column``
                 via :meth:`typeddfs.utils.Utils.guess_natsort_alg`.
                 Otherwise, :meth:typeddfs.utils.Utils.exact_natsort_alg`
                 is called with ``Utils.exact_natsort_alg(alg)``.
            reverse: Reverse the sort order (e.g. 'z' before 'a')
        """
        df = self.vanilla_reset()
        if alg is None:
            _, alg = Utils.guess_natsort_alg(self[column].dtype)
        else:
            _, alg = Utils.exact_natsort_alg(alg)
        zzz = natsorted([s for s in df[column]], alg=alg, reverse=reverse)
        df["__sort"] = df[column].map(lambda s: zzz.index(s))
        df.__class__ = self.__class__
        df = df.sort_values("__sort").drop("__sort", axis=1)
        return self.__class__._change(df)

    def sort_natural_index(self, *, alg: int = None, reverse: bool = False) -> __qualname__:
        """
        Calls natsorted on this index. Works for multi-index too.

        Args:
            alg: Input as the ``alg`` argument to ``natsorted``
                 If ``None``, the "best" algorithm is chosen from the dtype of ``column``
                 via :meth:`typeddfs.utils.Utils.guess_natsort_alg`.
                 Otherwise, :meth:typeddfs.utils.Utils.exact_natsort_alg`
                 is called with ``Utils.exact_natsort_alg(alg)``.
            reverse: Reverse the sort order (e.g. 'z' before 'a')
        """
        df = self.copy()
        if alg is None:
            # TODO: Does this work for multi-index?
            _, alg = Utils.guess_natsort_alg(self.index.dtype)
        else:
            _, alg = Utils.exact_natsort_alg(alg)
        zzz = natsorted([s for s in df.index], alg=alg)
        df["__sort"] = df.index.map(lambda s: zzz.index(s))
        df.__class__ = self.__class__
        df = df.sort_values("__sort").drop_cols(["__sort"])
        return self.__class__._change(df)

    def drop_cols(self, *cols: Union[str, Iterable[str]]) -> __qualname__:
        """
        Drops columns, ignoring those that are not present.

        Args:
            cols: A single column name or a list of column names
        """
        my_cols = set()
        for cols_ in cols:
            if isinstance(cols_, str):
                my_cols.add(cols_)
            else:
                my_cols.update(cols_)
        df = self
        for c in my_cols:
            if c in df.columns:
                df = df.drop(c, axis=1)
        return self.__class__._change(df)

    def rename_cols(self, **cols) -> __qualname__:
        """
        Shorthand for ``.rename(columns=)``.
        """
        df = self.rename(columns=cols)
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


__all__ = ["_NewMethodsMixin"]
