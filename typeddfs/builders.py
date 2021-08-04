"""
Defines a builder pattern for ``TypedDf``.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type, TypeVar, Union

import pandas as pd

from typeddfs.base_dfs import BaseDf
from typeddfs.df_errors import ClashError
from typeddfs.file_formats import FileFormat
from typeddfs.matrix_dfs import MatrixDf, AffinityMatrixDf
from typeddfs.typed_dfs import TypedDf

logger = logging.getLogger(Path(__file__).parent.name)
T = TypeVar("T", bound=MatrixDf, covariant=True)


class _GenericBuilder:
    def __init__(self, name: str, doc: Optional[str] = None):
        """
        Constructs a new builder.

        Args:
            name: The name of the resulting class
            doc: The docstring of the resulting class
        """
        if not isinstance(name, str):
            raise TypeError(f"Class name {name} is a {type(name)}, not str")
        self._name = name
        self._doc = doc
        self._clazz = None
        self._remapped_suffixes = {}
        self._encoding = "utf8"
        self._read_kwargs = defaultdict(dict)
        self._write_kwargs = defaultdict(dict)
        self._methods = {}
        self._classmethods = {}
        self._post_processing = None
        self._verifications = []
        # use utf-8 by default
        self.encoding()
        # make these use an explicit version
        # the user can override if needed
        self.add_read_kwargs("pickle", protocol=5)
        self.add_write_kwargs("pickle", protocol=5)

    def subclass(self, clazz: T) -> __qualname__:
        self._clazz = clazz
        return self

    def add_methods(self, **kwargs: Callable[[BaseDf, ...], Any]) -> __qualname__:
        """
        Attaches methods to the class.

        Example:
            add_methods(summary=lambda df: f"{len(df) rows")
        """
        self._methods.update(**kwargs)
        return self

    def add_classmethods(self, **kwargs: Callable[[Type[BaseDf], ...], Any]) -> __qualname__:
        """
        Attaches classmethods to the class.
        Mostly useful for factory methods.

        Example:
            add_classmethods(flat_instance=lambda t, value: MyClass(value))
        """
        self._classmethods.update(**kwargs)
        return self

    def remap_suffixes(self, **kwargs) -> __qualname__:
        """
        Makes read_files and write_files interpret a filename suffix differently.
        Suffixes like .gz, .zip, etc. are also included for text formats that are provided.

        Arguments:
            kwargs: Pairs of (suffix, format); e.g. remap_suffixes(commas="csv")
                    These must be names recognized in ``FileFormat``;
                    See that enum for the list of formats.
        """
        for suffix, fmt in kwargs.items():
            fmt = FileFormat.of(fmt)
            for s in fmt.compressed_variants(suffix):
                self._remapped_suffixes[s] = fmt
        return self

    def encoding(self, encoding: str = "utf8") -> __qualname__:
        """
        Has pandas-defined text read/write functions use UTF-8.
        UTF-8 was the default when the builder was constructed.

        Arguments:
            encoding: Use this encoding.
                      Values are case-insensitive and ignore hyphen.
                      (i.e. ``utf-8(-bom)`` and ``utf8(bom)`` are the same.
                      Special values are ``platform`` and ``utf8(bom)``.
                      "platform" is equivalent to ``sys.getdefaultencoding()``.
                      "utf8(bom)" changes the encoding depending on the platform at the time of writing.
                      (I.e. The read/write functions will work as expected when pickled.)
                      If ``utf8(bom)``, will use utf-8-sig if the platform is Windows ('nt').
                      Some applications will otherwise assume the default encoding (and break).
                      (Note: ``utf16(bom)`` will also work.)
        """
        encoding = encoding.lower().replace("-", "")
        self._encoding = encoding
        return self

    def newline(self, char: str = os.sep) -> __qualname__:
        r"""
        Set the newline character for line-based text formats.
        Consider setting to ``\n`` explicitly.
        """
        for fn in ["csv", "tsv", "flexwf", "lines"]:
            fn = FileFormat.of(fn)
            self.add_write_kwargs(fn, newline_separator=char)
            self.add_read_kwargs(fn, newline_separator=char)
        return self

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

    def verify(self, *conditions: Callable[[pd.DataFrame], Optional[str, bool]]) -> __qualname__:
        """
        Adds additional requirement(s) for the DataFrames.

        Returns:
            this builder for chaining

        Args:
            conditions: Functions of the DataFrame that return None if the condition is met, or an error message
        """
        self._verifications.extend(conditions)
        return self

    def build(self) -> Type[T]:
        new = self._generate()
        new.__name__ = self._name
        new.__doc__ = self._doc
        for k, v in self._methods.items():
            setattr(new, k, v)
        for k, v in self._classmethods.items():
            setattr(new, k, classmethod(v))
        return new

    def _generate(self) -> Type[T]:
        raise NotImplementedError()


class MatrixDfBuilder(_GenericBuilder):
    """"""

    def __init__(self, name: str, doc: Optional[str] = None):
        super().__init__(name, doc)
        self._strict = True  # can't change, currently
        self._dtype = None
        self._clazz = MatrixDf

    def dtype(self, dt: Type[Any]) -> __qualname__:
        self._dtype = dt
        return self

    def _generate(self) -> T:
        class New(self._clazz):
            @classmethod
            def text_encoding(cls) -> str:
                return self._encoding

            @classmethod
            def is_strict(cls) -> bool:
                return self._strict

            @classmethod
            def required_dtype(cls) -> bool:
                return self._dtype

            @classmethod
            def post_processing(cls) -> Optional[Callable[[BaseDf], Optional[BaseDf]]]:
                return self._post_processing

            @classmethod
            def verifications(cls) -> Sequence[Callable[[pd.DataFrame], Optional[str]]]:
                return list(self._verifications)

        # noinspection PyTypeChecker
        return New


class AffinityMatrixDfBuilder(MatrixDfBuilder):
    """"""

    def __init__(self, name: str, doc: Optional[str] = None):
        super().__init__(name, doc)
        self._clazz = AffinityMatrixDf


class TypedDfBuilder(_GenericBuilder):
    """
    A builder pattern for ``TypedDf``.

    Example:
        TypedDfBuilder.typed().require("name").build()
    """

    def __init__(self, name: str, doc: Optional[str] = None):
        super().__init__(name, doc)
        self._clazz = TypedDf
        self._req_meta = []
        self._res_meta = []
        self._req_cols = []
        self._res_cols = []
        self._dtypes = {}
        self._drop = []
        self._strict_meta = False
        self._strict_cols = False

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

    def _generate(self) -> Type[TypedDf]:
        """
        Final method in the chain.
        Creates a new subclass of ``TypedDf``.

        Returns:
            The new class

        Raises:
            ClashError: If there is a contradiction in the specification
        """
        all_names = [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]
        problem_names = [name for name in all_names if name in self._drop]
        if len(problem_names) > 0:
            raise ClashError(
                f"Required/reserved column/index names {problem_names} are auto-dropped"
            )

        class New(TypedDf):
            @classmethod
            def auto_dtypes(cls) -> Mapping[str, Type[Any]]:
                return dict(self._dtypes)

            @classmethod
            def post_processing(cls) -> Optional[Callable[[BaseDf], Optional[BaseDf]]]:
                return self._post_processing

            @classmethod
            def verifications(cls) -> Sequence[Callable[[pd.DataFrame], Optional[str]]]:
                return list(self._verifications)

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
                return list(self._res_cols)

            @classmethod
            def reserved_index_names(cls) -> Sequence[str]:
                return list(self._res_meta)

            @classmethod
            def required_index_names(cls) -> Sequence[str]:
                return list(self._req_meta)

            @classmethod
            def columns_to_drop(cls) -> Sequence[str]:
                return list(self._drop)

            @classmethod
            def remap_suffixes(cls) -> Mapping[str, FileFormat]:
                return dict(self._remapped_suffixes)

            @classmethod
            def text_encoding(cls) -> str:
                return self._encoding

            @classmethod
            def read_kwargs(cls) -> Mapping[FileFormat, Mapping[str, Any]]:
                return {k: dict(v) for k, v in self._read_kwargs.items()}

            @classmethod
            def write_kwargs(cls) -> Mapping[FileFormat, Mapping[str, Any]]:
                return {k: dict(v) for k, v in self._write_kwargs.items()}

        return New

    def _check(self, names: Sequence[str]) -> None:
        if any([name in {"index", "level_0", "Unnamed: 0"} for name in names]):
            raise ClashError("Columns 'index', 'level_0', and 'Unnamed: 0' are auto-dropped")
        if any(
            [name in {"__xml_is_empty_", "__xml_index_", "__feather_ignore_"} for name in names]
        ):
            raise ClashError(
                "__xml_is_empty_, __xml_index_, and __feather_ignore_ are forbidden names"
            )
        for name in names:
            if name in [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]:
                raise ClashError(f"Column {name} for {self._name} already exists")


__all__ = ["TypedDfBuilder", "MatrixDfBuilder", "AffinityMatrixDfBuilder"]
