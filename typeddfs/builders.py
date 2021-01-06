"""
Defines a builder pattern for ``TypedDf``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Sequence, Type, Union, Any

import pandas as pd

from typeddfs.typed_dfs import TypedDf

logger = logging.getLogger(Path(__file__).parent.name)


class TypedDfBuilder:
    """
    A builder pattern for ``TypedDf``.

    Example:
        TypedDfBuilder.typed().require("name").build()
    """

    def __init__(self, name: str, doc: Optional[str] = None):
        """
        Constructs a new builder.

        Args:
            name: The name of the resulting class
            doc: The docstring of the resulting class
        """
        self._name = name
        self._doc = doc
        self._req_meta = []
        self._res_meta = []
        self._req_cols = []
        self._res_cols = []
        self._dtypes = {}
        self._drop = []
        self._strict_meta = False
        self._strict_cols = False
        self._symmetric = False
        self._extra_reqs = []
        if not isinstance(name, str):
            raise TypeError(f"Class name {name} is a {type(name)}, not str")

    def require(
        self, *names: str, dtype: Optional[Type] = None, index: bool = False
    ) -> __qualname__:
        """
        Requires column(s) or index name(s).
        DataFrames will fail if they are missing any of these.

        Args:
            names: A varargs list of columns or index names
            dtype: An automatically applied transformation of the column values using ``.astype``
            index: If True, put these in the index

        Returns:
            self

        Raises:
            ValueError: If a name was already added, or is "level_0" or "index"
        """
        self._check(names)
        if index:
            self._req_meta.extend(names)
        else:
            self._req_cols.extend(names)
        if dtype is not None:
            for name in names:
                self._dtypes[name] = dtype
        return self

    def reserve(
        self, *names: str, dtype: Optional[Type] = None, index: bool = False
    ) -> __qualname__:
        """
        Reserves column(s) or index name(s) for optional inclusion.
        A reserved column will be accepted even if ``strict`` is set.
        A reserved index will be accepted even if ``strict`` is set;
        additionally, it will be automatically moved from the list of columns to the list of index names.

        Args:
            names: A varargs list of columns or index names
            dtype: An automatically applied transformation of the column values using ``.astype``
            index: If True, put these in the index

        Returns:
            self

        Raises:
            ValueError: If a name was already added, or is "level_0" or "index"
        """
        self._check(names)
        if index:
            self._res_meta.extend(names)
        else:
            self._res_cols.extend(names)
        if dtype is not None:
            for name in names:
                self._dtypes[name] = dtype
        return self

    def drop(self, *names: str) -> __qualname__:
        """
        Adds columns (and index names) that should be automatically dropped when calling ``convert``.

        Args:
            names: Varargs list of names

        Returns:
            self
        """
        self._drop.extend(names)
        return self

    def strict(self, index: bool = True, cols: bool = True) -> __qualname__:
        """
        Disallows any columns or index names not in the lists of reserved/required.

        Args:
            index: Disallow additional names in the index
            cols: Disallow additional columns

        Returns:
            self
        """
        self._strict_meta = index
        self._strict_cols = cols
        return self

    def symmetric(self) -> __qualname__:
        """
        Requires the DataFrame to be have the same columns as values in the index.

        Returns:
            self
        """
        self._symmetric = True
        return self

    def condition(self, *conditions: Callable[[pd.DataFrame], Optional[str]]) -> __qualname__:
        """
        Adds additional requirement(s) for the DataFrames.

        Args:
            conditions: Functions of the DataFrame that return None if the condition is met, or an error message
        """
        self._extra_reqs.extend(conditions)
        return self

    def build(self) -> Type[TypedDf]:
        """
        Final method in the chain.
        Creates a new subclass of ``TypedDf``.

        Returns:
            The new class

        Raises:
            ValueError: If ``symmetric()`` was called and there are multiple required+reserved index names.
        """
        if self._symmetric and len([*self._res_meta, *self._req_meta]) > 1:
            raise ValueError("Cannot enforce symmetry for multi-index DataFrames")

        class New(TypedDf):
            @classmethod
            def more_indices_allowed(cls) -> bool:
                return not self._strict_meta

            @classmethod
            def more_columns_allowed(cls) -> bool:
                return not self._strict_cols

            @classmethod
            def required_columns(cls) -> Sequence[str]:
                return self._req_cols

            @classmethod
            def reserved_columns(cls) -> Sequence[str]:
                return self._res_cols

            @classmethod
            def reserved_index_names(cls) -> Sequence[str]:
                return self._res_meta

            @classmethod
            def required_index_names(cls) -> Sequence[str]:
                return self._req_meta

            @classmethod
            def must_be_symmetric(cls) -> bool:
                return self._symmetric

            @classmethod
            def columns_to_drop(cls) -> Sequence[str]:
                return self._drop

            @classmethod
            def extra_conditions(cls) -> Sequence[Callable[[pd.DataFrame], Optional[str]]]:
                return self._extra_reqs

        New.__name__ = self._name
        New.__doc__ = self._doc
        return New

    def _check(self, names: Sequence[str]) -> None:
        if any([name in {"index", "level_0"} for name in names]):
            raise ValueError(
                f"Any column called 'index' or 'level_0' is automatically dropped (reserving: {names})"
            )
        for name in names:
            if name in [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]:
                raise ValueError(
                    f"Cannot add {name} to builder for {self._name}: it already exists"
                )


__all__ = ["TypedDfBuilder"]
