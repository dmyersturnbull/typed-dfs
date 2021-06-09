"""
Defines a builder pattern for ``TypedDf``.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence, Type, Mapping, Any, Union
from warnings import warn

import pandas as pd

from typeddfs import BaseDf, FileFormat
from typeddfs.typed_dfs import TypedDf
from typeddfs.df_errors import ClashError

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
        self._post_processing = None
        self._extra_reqs = []
        self._read_kwargs = defaultdict(dict)
        self._write_kwargs = defaultdict(dict)
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
            this builder for chaining

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
                   THIS PARAMETER IS **NOT IMPLEMENTED YET**
            index: If True, put these in the index

        Returns:
            this builder for chaining

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
            this builder for chaining
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
            this builder for chaining
        """
        self._strict_meta = index
        self._strict_cols = cols
        return self

    def symmetric(self) -> __qualname__:
        """
        Requires the DataFrame to be have the same columns as values in the index.

        Returns:
            this builder for chaining
        """
        self._symmetric = True
        return self

    def post(self, fn: Callable[[BaseDf], BaseDf]) -> __qualname__:
        """
        Adds a method that is called on the converted DataFrame.
        It is called immediately before final optional conditions (``verify``) are checked.
        The function must return a new DataFrame.

        Returns:
            this builder for chaining
        """
        self._post_processing = fn
        return self

    def verify(self, *conditions: Callable[[pd.DataFrame], Optional[str]]) -> __qualname__:
        """
        Adds additional requirement(s) for the DataFrames.

        Returns:
            this builder for chaining

        Args:
            conditions: Functions of the DataFrame that return None if the condition is met, or an error message
        """
        self._extra_reqs.extend(conditions)
        return self

    def condition(self, *conditions) -> __qualname__:  # pragma: no cover
        """
        Deprecated alias for ``validate``.

        Returns:
            this builder for chaining
        """
        warn("TypedDfBuilder.condition is deprecated; use verify instead", DeprecationWarning)
        return self.verify(*conditions)

    def add_read_kwargs(self, fmt: Union[FileFormat, str], **kwargs) -> __qualname__:
        """
        Adds keyword arguments that are passed to ``read_`` methods when called from ``read_file``.
        Rarely needed.

        Arguments:
            fmt: The file format (which corresponds to the delegated method)
            kwargs: key-value pairs that are used for the specified format

        Returns:
            this builder for chaining
        """
        fmt = FileFormat.of(fmt)
        for k, v in kwargs.items():
            self._read_kwargs[fmt][k] = v
        return self

    def add_write_kwargs(self, fmt: Union[FileFormat, str], **kwargs) -> __qualname__:
        """
        Adds keyword arguments that are passed to ``to_`` methods when called from ``to_file``.
        Rarely needed.

        Example:
            .. code::

                TypedDfs.typed("x").add_write_kwargs()

        Arguments:
            fmt: The file format (which corresponds to the delegated method)
            kwargs: key-value pairs that are used for the specified format

        Returns:
            this builder for chaining
        """
        fmt = FileFormat.of(fmt)
        for k, v in kwargs.items():
            self._write_kwargs[fmt][k] = v
        return self

    def build(self) -> Type[TypedDf]:
        """
        Final method in the chain.
        Creates a new subclass of ``TypedDf``.

        Returns:
            The new class

        Raises:
            ClashError: If there is a contradiction in the specification
        """
        if self._symmetric and len([*self._res_meta, *self._req_meta]) > 1:
            raise ClashError("Cannot enforce symmetry for multi-index DataFrames")

        all_names = [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]
        problem_names = [name for name in all_names if name in self._drop]
        if len(problem_names) > 0:
            raise ClashError(
                f"Required/reserved column/index names {problem_names} are auto-dropped"
            )

        class New(TypedDf):
            @classmethod
            def auto_dtypes(cls) -> Mapping[str, Type[Any]]:
                return self._dtypes

            @classmethod
            def post_processing(cls) -> Optional[Callable[[BaseDf], Optional[BaseDf]]]:
                return self._post_processing

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

            @classmethod
            def read_kwargs(cls) -> Mapping[FileFormat, Mapping[str, Any]]:
                return self._read_kwargs

            @classmethod
            def write_kwargs(cls) -> Mapping[FileFormat, Mapping[str, Any]]:
                return self._write_kwargs

        New.__name__ = self._name
        New.__doc__ = self._doc
        return New

    def _check(self, names: Sequence[str]) -> None:
        if any([name in {"index", "level_0"} for name in names]):
            raise ClashError(
                f"Any column called 'index' or 'level_0' is automatically dropped (reserving: {names})"
            )
        for name in names:
            if name in [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]:
                raise ClashError(
                    f"Cannot add {name} to builder for {self._name}: it already exists"
                )


__all__ = ["TypedDfBuilder"]
