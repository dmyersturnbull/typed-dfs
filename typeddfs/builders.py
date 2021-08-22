"""
Defines a builder pattern for ``TypedDf``.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Callable, Optional, Sequence, Type, Union

import pandas as pd
from typeddfs.utils import Utils

from typeddfs.base_dfs import BaseDf
from typeddfs.df_errors import ClashError, FormatInsecureError
from typeddfs.df_typing import DfTyping, IoTyping
from typeddfs.file_formats import FileFormat
from typeddfs.matrix_dfs import MatrixDf, AffinityMatrixDf
from typeddfs.typed_dfs import TypedDf
from typeddfs._utils import _AUTO_DROPPED_NAMES, _FORBIDDEN_NAMES, _DEFAULT_HASH_ALG

logger = logging.getLogger("typeddfs")


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
        self._req_meta = []
        self._res_meta = []
        self._req_cols = []
        self._res_cols = []
        self._dtypes = {}
        self._value_dtype = None
        self._drop = []
        self._strict_meta = False
        self._strict_cols = False
        self._hash_alg = _DEFAULT_HASH_ALG
        self._hash_file = False
        self._hash_dir = False
        self._index_series_name = False
        self._column_series_name = False
        self._secure = False
        # use utf-8 by default
        self.encoding()
        # make these use an explicit version
        # the user can override if needed
        self.add_read_kwargs("pickle", protocol=5)
        self.add_write_kwargs("pickle", protocol=5)

    def series_names(
        self, index: Union[None, bool, str] = False, columns: Union[None, bool, str] = False
    ) -> __qualname__:
        """
        Sets ``pd.DataFrame.index.name`` and/or ``pd.DataFrame.columns.name``.
        Valid values are ``False`` to not set (default), ``None`` to set to ``None``,
        or a string to set to.

        Returns:
            This builder for chaining
        """
        self._index_series_name = index
        self._column_series_name = columns
        return self

    def add_methods(
        self, *args: Callable[[BaseDf, ...], Any], **kwargs: Callable[[BaseDf, ...], Any]
    ) -> __qualname__:
        """
        Attaches methods to the class.

        Args:
            args: Functions whose names are used directly
            kwargs: Mapping from function names to functions (the keys will be the method names)

        Example:
            add_methods(summary=lambda df: f"{len(df) rows")

        Returns:
            This builder for chaining
        """
        self._methods.update({m.__name__: m for m in args})
        self._methods.update(**kwargs)
        return self

    def add_classmethods(self, **kwargs: Callable[[Type[BaseDf], ...], Any]) -> __qualname__:
        """
        Attaches classmethods to the class.
        Mostly useful for factory methods.

        Example:
            add_classmethods(flat_instance=lambda t, value: MyClass(value))

        Returns:
            This builder for chaining
        """
        self._classmethods.update(**kwargs)
        return self

    def post(self, fn: Callable[[BaseDf], BaseDf]) -> __qualname__:
        """
        Adds a method that is called on the converted DataFrame.
        It is called immediately before final optional conditions (``verify``) are checked.
        The function must return a new DataFrame.

        Returns:
            This builder for chaining
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

        Returns:
            This builder for chaining
        """
        self._verifications.extend(conditions)
        return self

    def remap_suffixes(self, **kwargs) -> __qualname__:
        """
        Makes read_files and write_files interpret a filename suffix differently.
        Suffixes like .gz, .zip, etc. are also included for text formats that are provided.

        Arguments:
            kwargs: Pairs of (suffix, format); e.g. remap_suffixes(commas="csv")
                    These must be names recognized in ``FileFormat``;
                    See that enum for the list of formats.

        Returns:
            This builder for chaining
        """
        for suffix, fmt in kwargs.items():
            fmt = FileFormat.of(fmt)
            for s in fmt.compressed_variants(suffix):
                self._remapped_suffixes[s] = fmt
        return self

    def hash(self, alg: str = "sha256", file: bool = True, directory: bool = False) -> __qualname__:
        """
        Write a hash file (e.g. .sha256) alongside files.
        Performed when calling :py.meth:`typeddfs.abs_dfs.AbsDf.write_file`.
        The hash files will be in the `sha1sum <https://en.wikipedia.org/wiki/Sha1sum>`_ format,
        with a the filename, followed by ``" *"``, followed by the filename.

        Note that this affects the default behavior of :py.meth:`typeddfs.abs_dfs.AbsDf.write_file`,
        which can be called with ``file_hash=False`` and/or ``dir_hash=False``.

        Args:
            alg: The name of the algorithm in :py:mod:`hashlib`;
                 The final name will ignore any hyphens and be converted to lowercase,
                 and the suffix will be ``"." + alg``.
            file: Alongside a file ``"my_file.csv.gz"``,
                  write a file ``"my_file.csv.gz."+alg`` alongside.
            directory: Alongside a file ``"my_file.csv.gz"`` in ``"my_dir"``,
                       append to a file ``"my_dir/my_dir"+alg``,
                       which presumably should contain hashes for files in that directory.

        Returns:
            This builder for chaining
        """
        self._hash_alg = Utils.get_algorithm(alg)
        self._hash_file = file
        self._hash_dir = directory
        return self

    def secure(self) -> __qualname__:
        """
        Bans IO with insecure formats.
        This includes Pickle and Excel formats that support macros.

        Returns:
            This builder for chaining
        """
        self._secure = True
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

        Returns:
            This builder for chaining
        """
        encoding = encoding.lower().replace("-", "")
        self._encoding = encoding
        return self

    def add_read_kwargs(self, fmt: Union[FileFormat, str], **kwargs) -> __qualname__:
        """
        Adds keyword arguments that are passed to ``read_`` methods when called from ``read_file``.
        Rarely needed.

        Arguments:
            fmt: The file format (which corresponds to the delegated method)
            kwargs: key-value pairs that are used for the specified format

        Returns:
            This builder for chaining
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
            This builder for chaining
        """
        fmt = FileFormat.of(fmt)
        for k, v in kwargs.items():
            self._write_kwargs[fmt][k] = v
        return self

    def _build(self) -> Type[BaseDf]:
        if self._secure and self._hash_alg in Utils.insecure_hash_functions():
            raise FormatInsecureError(f"Hash algorithm {self._hash_alg} forbidden by .secure()")
        self._check_final()

        _io_typing = IoTyping(
            _remap_suffixes=dict(self._remapped_suffixes),
            _text_encoding=self._encoding,
            _read_kwargs=dict(self._read_kwargs),
            _write_kwargs=dict(self._write_kwargs),
            _hash_alg=self._hash_alg,
            _save_hash_file=self._hash_file,
            _save_hash_dir=self._hash_dir,
            _secure=self._secure,
        )

        _typing = DfTyping(
            _io_typing=_io_typing,
            _auto_dtypes=dict(self._dtypes),
            _post_processing=self._post_processing,
            _verifications=self._verifications,
            _more_index_names_allowed=not self._strict_meta,
            _more_columns_allowed=not self._strict_cols,
            _required_columns=list(self._req_cols),
            _required_index_names=list(self._req_meta),
            _reserved_columns=list(self._res_cols),
            _reserved_index_names=list(self._res_meta),
            _columns_to_drop=set(self._drop),
            _index_series_name=self._index_series_name,
            _column_series_name=self._column_series_name,
            _value_dtype=self._value_dtype,
        )

        class New(self._clazz):
            @classmethod
            def get_typing(cls) -> DfTyping:
                return _typing

        New.__name__ = self._name
        New.__doc__ = self._doc
        for k, v in self._methods.items():
            setattr(New, k, v)
        for k, v in self._classmethods.items():
            setattr(New, k, classmethod(v))
        return New

    def _check_final(self) -> None:
        raise NotImplementedError()


class MatrixDfBuilder(_GenericBuilder):
    """
    A builder pattern for :py.class:`typeddfs.matrix_dfs.MatrixDf`.
    """

    def __init__(self, name: str, doc: Optional[str] = None):
        super().__init__(name, doc)
        self._clazz = MatrixDf
        self._index_series_name = "row"
        self._column_series_name = "column"
        self._req_meta.append("row")

    def build(self) -> Type[MatrixDf]:
        """
        Builds this type.

        Returns:
            A newly created subclass of :py.class:`typeddfs.matrix_dfs.MatrixDf`.

        Raises:
            ClashError: If there is a contradiction in the specification
            FormatInsecureError: If :py.meth:`hash` set an insecure
                                 hash format and :py.meth:`secure` was set.

        .. note ::

            Copies, so this builder can be used to create more types without interference.
        """
        # self.add_read_kwargs(FileFormat.csv, index_col=0)
        # self.add_read_kwargs(FileFormat.tsv, index_col=0)
        # noinspection PyTypeChecker
        return self._build()

    def subclass(self, clazz: Type[MatrixDf]) -> __qualname__:
        """
        Make the type subclass some subclass of :py.class:`typeddfs.matrix_dfs.MatrixDf`.

        Returns:
            This builder for chaining

        Raises:
            ValueError: If ``clazz`` is not a subclass of ``MatrixDf``.
        """
        if not issubclass(clazz, MatrixDf):
            raise ValueError(f"{clazz.__name__} is not a subclass of {MatrixDf.__name__}")
        self._clazz = clazz
        return self

    def dtype(self, dt: Type[Any]) -> __qualname__:
        self._value_dtype = dt
        return self

    def _check_final(self) -> None:
        pass


class AffinityMatrixDfBuilder(MatrixDfBuilder):
    """
    A builder pattern for :py.class:`typeddfs.matrix_dfs.AffinityMatrixDf`.
    """

    def __init__(self, name: str, doc: Optional[str] = None):
        super().__init__(name, doc)
        self._clazz = AffinityMatrixDf

    def build(self) -> Type[AffinityMatrixDf]:
        """
        Builds this type.

        Returns:
            A newly created subclass of :py.class:`typeddfs.matrix_dfs.AffinityMatrixDf`.

        Raises:
            typeddfs.df_errors.ClashError: If there is a contradiction in the specification
            typeddfs.df_errors.FormatInsecureError: If :py.meth:`hash` set an insecure
                                                    hash format and :py.meth:`secure` was set.

        .. note ::

            Copies, so this builder can be used to create more types without interference.
        """
        # noinspection PyTypeChecker
        return self._build()

    def subclass(self, clazz: Type[AffinityMatrixDf]) -> __qualname__:
        """
        Make the type subclass some subclass of :py.class:`typeddfs.matrix_dfs.AffinityMatrixDf`.

        Returns:
            This builder for chaining

        Raises:
            ValueError: If ``clazz`` is not a subclass of ``AffinityMatrixDf``.
        """
        if not issubclass(clazz, AffinityMatrixDf):
            raise ValueError(f"{clazz.__name__} is not a subclass of {AffinityMatrixDf.__name__}")
        self._clazz = clazz
        return self


class TypedDfBuilder(_GenericBuilder):
    """
    A builder pattern for :py.class:`typeddfs.typed_dfs.TypedDf`.

    Example:
        TypedDfBuilder.typed().require("name").build()
    """

    def __init__(self, name: str, doc: Optional[str] = None):
        super().__init__(name, doc)
        self._clazz = TypedDf

    def build(self) -> Type[TypedDf]:
        """
        Builds this type.

        Returns:
            A newly created subclass of :py.class:`typeddfs.typed_dfs.TypedDf`.

        Raises:
            ClashError: If there is a contradiction in the specification
            FormatInsecureError: If :py.meth:`hash` set an insecure
                                 hash format and :py.meth:`secure` was set.

        .. note ::

            Copies, so this builder can be used to create more types without interference.
        """
        # noinspection PyTypeChecker
        return self._build()

    def subclass(self, clazz: Type[TypedDf]) -> __qualname__:
        """
        Make the type subclass some subclass of :py.class:`typeddfs.typed_dfs.TypedDf`.

        Returns:
            This builder for chaining

        Raises:
            ValueError: If ``clazz`` is not a subclass of ``TypedDf``.
        """
        if not issubclass(clazz, TypedDf):
            raise ValueError(f"{clazz.__name__} is not a subclass of {TypedDf.__name__}")
        self._clazz = clazz
        return self

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
            This builder for chaining

        Raises:
            typeddfs.df_errors.ClashError: If a name was already added or is forbidden
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
            This builder for chaining

        Raises:
            typeddfs.df_errors.ClashError: If a name was already added or is forbidden
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
        Adds columns (and index names) that should be automatically dropped.

        Args:
            names: Varargs list of names

        Returns:
            This builder for chaining
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
            This builder for chaining
        """
        self._strict_meta = index
        self._strict_cols = cols
        return self

    def _check_final(self) -> None:
        """
        Final method in the chain.
        Creates a new subclass of ``TypedDf``.

        Returns:
            The new class

        Raises:
            typeddfs.df_errors.ClashError: If there is a contradiction in the specification
        """
        all_names = [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]
        problem_names = [name for name in all_names if name in self._drop]
        if len(problem_names) > 0:
            raise ClashError(
                f"Required/reserved column/index names {problem_names} are auto-dropped"
            )

    def _check(self, names: Sequence[str]) -> None:
        if any([name in _AUTO_DROPPED_NAMES for name in names]):
            raise ClashError(f"Columns {','.join(_AUTO_DROPPED_NAMES)} are auto-dropped")
        if any([name in _FORBIDDEN_NAMES for name in names]):
            raise ClashError(f"{','.join(_FORBIDDEN_NAMES)} are forbidden names")
        for name in names:
            if name in [*self._req_cols, *self._req_meta, *self._res_cols, *self._res_meta]:
                raise ClashError(f"Column {name} for {self._name} already exists")


__all__ = ["TypedDfBuilder", "MatrixDfBuilder", "AffinityMatrixDfBuilder"]
