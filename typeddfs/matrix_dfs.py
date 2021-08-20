"""
DataFrames that are essentially n-by-m matrices.
"""
from __future__ import annotations

import abc
from copy import deepcopy
from functools import partial
from inspect import cleandoc
from typing import Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState

from typeddfs.base_dfs import BaseDf
from typeddfs.df_errors import (
    InvalidDfError,
    RowColumnMismatchError,
    VerificationFailedError,
)
from typeddfs.df_typing import DfTyping, FINAL_DF_TYPING
from typeddfs.typed_dfs import TypedDf


class LongFormMatrixDf(TypedDf):
    """
    A long-form matrix with columns "row", "column", and "value".
    """

    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["row", "column", "value"])


class _MatrixDf(BaseDf, metaclass=abc.ABCMeta):
    @classmethod
    def convert(cls, df: pd.DataFrame) -> __qualname__:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Can't convert {type(df)} to {cls.__name__}")
        # first always reset the index so we can manage what's in the index vs columns
        # index_names() will return [] if no named indices are found
        df.__class__ = cls
        t = cls.get_typing()
        # df = df.vanilla_reset()
        # df = df.set_index(t.required_index_names[0])
        if df.index.names == [None] and "row" in df.columns:
            df = df.set_index("row")
        df.columns.name = "column"
        df.index.name = "row"
        if t.value_dtype is not None:
            df = df.astype(t.value_dtype)
        # now change the class
        df.__class__ = cls
        # noinspection PyProtectedMember
        cls._check(df)
        return df

    @classmethod
    def _check(cls, df) -> None:
        t = cls.get_typing()
        # TODO: Why doesn't .dtype work?
        if [str(c) for c in df.index.names] != list(df.index.names):
            raise InvalidDfError("Some index names are non-str")
        if [str(c) for c in df.columns] != df.columns.tolist():
            raise InvalidDfError("Some columns are non-str")
        for req in t.verifications:
            value = req(df)
            if value is not None and value is not True:
                raise VerificationFailedError(str(value))

    def is_symmetric(self) -> bool:
        """
        Returns True if the matrix is fully symmetric with exact equality.
        """
        return self.rows == self.cols and np.array_equal(self.values, self.T.values)

    def sub_matrix(self, rows: Set[str], cols: Set[str]) -> __qualname__:
        """
        Returns a matrix containing only these labels.
        """
        return self.__class__(self.loc[rows][cols])

    def long_form(self) -> LongFormMatrixDf:
        """
        Melts into a long-form DataFrame with columns "row", "column", and "value".

        Consider calling ``triangle`` first if the matrix is (always) symmetric.
        """
        # TODO: melt wasn't working
        df = []
        for r, row in enumerate(self.rows):
            for c, col in enumerate(self.cols):
                df.append(pd.Series(dict(row=row, column=col, value=self.iat[r, c])))
        return LongFormMatrixDf.convert(pd.DataFrame(df))

    def triangle(self, upper: bool = False, strict: bool = False) -> __qualname__:
        """
        NaNs out the upper (or lower) triangle, returning a copy.

        Arguments:
            upper: Keep the upper triangular matrix instead of the lower
            strict: Discard the diagonal (set it to NaN)
        """
        fn = np.triu if upper else np.tril
        fn = partial(fn, k=1) if strict else fn
        return self.__class__(self.where(fn(np.ones(self.shape)).astype(bool)))

    def sort_alphabetical(self) -> __qualname__:
        """
        Sorts by the rows and columns alphabetically.
        """
        df = self.sort_natural_index()
        df = df.transpose().sort_natural_index()
        df = df.transpose()
        return df

    def shuffle(self, rand: Union[None, int, RandomState] = None) -> __qualname__:
        """
        Returns a copy with every value mapped to a new location.
        Destroys the correct links between labels and values.
        Useful for permutation tests.
        """
        cp = deepcopy(self.flatten())
        if rand is None:
            rand = np.random.RandomState()
        elif isinstance(rand, int):
            rand = np.random.RandomState(seed=rand)
        rand.shuffle(cp)
        values = cp.reshape((len(self.rows), len(self.columns)))
        return self.__class__(values, index=self.rows, columns=self.columns)

    def diagonals(self) -> np.array:
        """
        Returns an array of the diagonal elements.
        """
        return pd.Series(np.diag(self), index=[self.index, self.columns]).values

    def flatten(self) -> np.array:
        """
        Flattens the values into a 1-d array.
        """
        return self.values.flatten()

    @property
    def dim_str(self) -> str:
        """
        Returns a simple string of n_rows by n_columns.
        E.g.: ``15 × 15``.
        """
        return f"{len(self.rows)} × {len(self.columns)}"

    @property
    def dims(self) -> Tuple[int, int]:
        """
        Returns (n rows, n_columns).
        """
        return len(self.rows), len(self.columns)

    @property
    def rows(self) -> Sequence[str]:
        """
        Returns the row labels.
        """
        return self.index.tolist()

    @property
    def cols(self) -> Sequence[str]:
        """
        Returns the column labels.
        """
        return self.columns.tolist()

    def _repr_html_(self) -> str:
        cls = self.__class__
        mark = "✅" if self.__class__.is_valid(self) else "❌"
        return cleandoc(
            f"""
            <strong>{cls.name}: {self.dim} {mark}</strong>
            {pd.DataFrame._repr_html_(self)}
        """
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self.rows)} × {len(self.columns)} @ {hex(id(self))})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({len(self.rows)} × {len(self.columns)})"


class MatrixDf(_MatrixDf):
    """
    A dataframe that is best thought of as a simple matrix.
    Contains a single index level and a list of columns,
    with numerical values of a single dtype.
    """

    @classmethod
    def get_typing(cls) -> DfTyping:
        return FINAL_DF_TYPING  # default only -- should be overridden

    @classmethod
    def new_df(
        cls,
        rows: Union[int, Sequence[str]],
        cols: Union[int, Sequence[str]],
        fill: Union[int, float, complex] = 0,
    ) -> __qualname__:
        """
        Returns a DataFrame that is empty but valid.

        Arguments:
            rows: Either a number of rows or a sequence of labels.
                  If a number is given, will choose (str-type) labels '0', '1', ...
            cols: Either a number of columns or a sequence of labels.
                  If a number is given, will choose (str-type) labels '0', '1', ...
            fill: A value to fill in every cell.
                  Should match ``self.required_dtype``.
                  String values are

        Raises:
            InvalidDfError: If a function in ``verifications`` fails (returns False or a string).
            IntCastingNaNError: If ``fill`` is NaN or inf and ``self.required_dtype`` does not support it.
        """
        if isinstance(rows, int):
            rows = [str(r) for r in range(rows)]
        if isinstance(cols, int):
            cols = [str(c) for c in range(cols)]
        a = np.ndarray(shape=(len(rows), len(cols)))
        a.fill(fill)
        df = pd.DataFrame(a, columns=cols)
        return cls.convert(df)


class AffinityMatrixDf(_MatrixDf):
    """
    A similarity or distance matrix.
    The rows and columns must match, and only 1 index is allowed.
    """

    @classmethod
    def get_typing(cls) -> DfTyping:
        return FINAL_DF_TYPING  # default only -- should be overridden

    @classmethod
    def new_df(
        cls, n: Union[int, Sequence[str]], fill: Union[int, float, complex] = 0
    ) -> __qualname__:
        """
        Returns a DataFrame that is empty but valid.

        Arguments:
            n:    Either a number of rows/columns or a sequence of labels.
                  If a number is given, will choose (str-type) labels '0', '1', ...
            fill: A value to fill in every cell.
                  Should match ``self.required_dtype``.

        Raises:
            InvalidDfError: If a function in ``verifications`` fails (returns False or a string).
            IntCastingNaNError: If ``fill`` is NaN or inf and ``self.required_dtype`` does not support it.
        """
        if isinstance(n, int):
            n = [str(c) for c in range(n)]
        a = np.ndarray(shape=(len(n), len(n)))
        a.fill(fill)
        df = pd.DataFrame(a, columns=n)
        df["row"] = n
        return cls.convert(df)

    @classmethod
    def _check(cls, df: BaseDf):
        rows = df.index.tolist()
        cols = df.columns.tolist()
        t = cls.get_typing()
        if df.rows != df.cols:
            raise RowColumnMismatchError(f"Rows {rows} but columns {cols}")
        for req in t.verifications:
            value = req(df)
            if value is not None:
                raise VerificationFailedError(value)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self.rows)} × {len(self.columns)} @ {hex(id(self))})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({len(self.rows)} × {len(self.columns)})"

    def symmetrize(self) -> __qualname__:
        """
        Averages with its transpose, forcing it to be symmetric.
        """
        return self.__class__(0.5 * (self + self.T))


__all__ = ["MatrixDf", "AffinityMatrixDf", "LongFormMatrixDf"]
