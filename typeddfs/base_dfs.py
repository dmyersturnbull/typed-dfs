"""
Defines the superclasses of the types ``TypedDf`` and ``UntypedDf``.
"""
from __future__ import annotations

import csv
import abc
import os
from pathlib import Path, PurePath
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union, FrozenSet

import pandas as pd
from pandas.io.common import get_handle

# noinspection PyProtectedMember
from tabulate import tabulate, TableFormat, DataRow
from natsort import natsorted, ns
from pandas.core.frame import DataFrame as _InternalDataFrame

# noinspection PyProtectedMember
from typeddfs._utils import _Utils, _SENTINAL

_FAKE_SEP = "\u2008"  # 6-em space; very unlikely to occur
PathLike = Union[str, PurePath]


class UnsupportedOperationError(Exception):
    """ """


class FilenameSuffixError(UnsupportedOperationError):
    """ """


class NonStrColumnError(UnsupportedOperationError):
    """ """


class NotSingleColumnError(UnsupportedOperationError):
    """ """


class InvalidDfError(ValueError):
    """ """


class ExtraConditionFailedError(InvalidDfError):
    """ """


class MissingColumnError(InvalidDfError):
    """ """


class AsymmetricDfError(InvalidDfError):
    """ """


class UnexpectedColumnError(InvalidDfError):
    """ """


class UnexpectedIndexNameError(InvalidDfError):
    """ """


class ValueNotUniqueError(ValueError):
    """ """


class NoValueError(ValueError):
    """ """


class PrettyDf(_InternalDataFrame, metaclass=abc.ABCMeta):
    """
    A DataFrame with an overridden ``_repr_html_`` and some simple additional methods.
    """

    def column_names(self) -> List[str]:
        """
        Returns the list of columns.

        Returns:
            A Python list
        """
        return list(self.columns)

    def index_names(self) -> List[str]:
        """
        Returns the list of index names.
        Unlike ``self.index.names``, returns ``[]`` instead of ``[None]`` if there is no index.

        Returns:
            A Python list
        """
        lst = list(self.index.names)
        return [] if lst == [None] else lst

    def is_multindex(self) -> bool:
        """
        Returns whether this is a ``pd.MultiIndex``.
        """
        return isinstance(self.index, pd.MultiIndex)

    def n_rows(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self)

    def n_columns(self) -> int:
        """
        Returns the number of columns.
        """
        return len(self.columns)

    def n_indices(self) -> int:
        """
        Returns the number of index names.
        """
        return len(self.index.names)

    @property
    def _constructor_expanddim(self):  # pragma: no cover
        # this raises a NotImplementedError in _InternalDataFrame, so let's override it here to prevent tools and IDEs from complaining
        raise UnsupportedOperationError()

    def _repr_html_(self) -> str:
        """
        Renders HTML for display() in Jupyter notebooks.
        Jupyter automatically uses this function.

        Returns:
            Just a string containing HTML, which will be wrapped in an HTML object
        """
        # noinspection PyProtectedMember
        return (
            f"<strong>{self.__class__.__name__}: {self._dims()}</strong>\n{super()._repr_html_()}"
        )

    def _dims(self) -> str:
        """
        Returns a string describing the dimensionality.

        Returns:
            A text description of the dimensions of this DataFrame
        """
        # we could handle multi-level columns, but they're quite rare, and the number of rows is probably obvious when looking at it
        if len(self.index.names) > 1:
            return f"{len(self)} rows × {len(self.columns)} columns, {len(self.index.names)} index columns"
        else:
            return f"{len(self)} rows × {len(self.columns)} columns"


class AbsDf(PrettyDf, metaclass=abc.ABCMeta):
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

    def write_file(self, path: Union[Path, str], *args, **kwargs):
        """
        Writes to a file (or possibly URL), guessing the format from the filename extension.
        Delegates to the ``to_*`` functions of this class (e.g. ``to_csv``).
        Only includes file formats that can be read back in with corresponding ``to`` methods,
        and excludes pickle.

        Supports:
            - .csv, .tsv, or .tab (optionally with .gz, .zip, .bz2, or .xz)
            - .json  (optionally with .gz, .zip, .bz2, or .xz)
            - .feather
            - .parquet or .snappy
            - .h5 or .hdf
            - .xlsx or .xls
            - .txt, .lines, or .list (optionally with .gz, .zip, .bz2, or .xz);
              see ``to_lines()``

        Args:
            path: Only path-like strings or pathlib objects are supported, not buffers
                  (because we need a filename).
            args: Positional args passed to the read_ function
            kwargs: Keyword args passed to the function

        Returns:
            Whatever the corresponding method on ``pd.to_*`` returns.
            This is usually either str or None

        Raises:
            ValueError: If the type of a column or index name is non-str
        """
        types = set(self.column_names()).union(self.index_names())
        if any((not isinstance(c, str) for c in types)):
            raise NonStrColumnError(f"Columns must be of str type to serialize, not {types}")
        cls = self.__class__
        return cls._guess_io(self, True, path, {}, *args, **kwargs)

    @classmethod
    def read_file(
        cls,
        path: Union[Path, str],
        *args,
        nl: Optional[str] = _SENTINAL,
        header: Optional[str] = _SENTINAL,
        skip_blank_lines: bool = _SENTINAL,
        comment: str = _SENTINAL,
        **kwargs,
    ) -> __qualname__:
        """
        Reads from a file (or possibly URL), guessing the format from the filename extension.
        Delegates to the ``read_*`` functions of this class.

        You can always write and then read back to get the same dataframe::

            # df is any DataFrame from typeddfs
            # path can use any suffix
            df.write_file(path))
            df.read_file(path)

        Supports:
            - .csv, .tsv, or .tab (optionally with .gz, .zip, .bz2, or .xz)
            - .json  (optionally with .gz, .zip, .bz2, or .xz)
            - .feather
            - .parquet or .snappy
            - .h5 or .hdf
            - .xlsx or .xls
            - .fxf (fixed-width; read_fwf)
            - .arrows (optional with .gz, etc.); uses ``--->`` as the delimiter, strips whitespace,
              and ignores blank lines and comments (#)
            - .txt, .lines, or .list (optionally .gz, .zip, .bz2, or .xz); see ``read_lines()``

        Args:
            path: Only path-like strings or pathlib objects are supported, not buffers
                  (because we need a filename).
            nl: Passes ``line_terminator=nl`` to ``.read_csv`` if the output is a CSV/TSV variant.
                This can usually be inferred and is more important with ``.write_file``.
            header: Same as ``header`` in ``to_csv`` but not passed for non-CSV/TSV.
                    Just allows passing header without worrying about whether it applies.
            skip_blank_lines: Same idea as for ``header``
            comment: Prefix indicating comments to ignore; only applies to ``to_lines``
            args: Positional args passed to the read_ function
            kwargs: Keyword args passed to the function

        Returns:
            An instance of this class
        """
        all_params = {}
        if nl != _SENTINAL:
            all_params["nl"] = dict(line_terminator="\n")
        if header != _SENTINAL:
            all_params["header"] = dict(header=header)
        if skip_blank_lines != _SENTINAL:
            all_params["skip_blank_lines"] = dict(skip_blank_lines=skip_blank_lines)
        if comment != _SENTINAL:
            all_params["comment"] = dict(comment=comment)
        return cls._guess_io(cls, False, path, all_params, *args, **kwargs)

    @classmethod
    def can_read(cls) -> FrozenSet[str]:
        """
        Returns all filename extensions that can be read using ``read_file``.
        Some, such as `.h5` and `.snappy`, are only included if their respective libraries
        were imported (when typeddfs was imported).
        The ``read_lines`` functions (``.txt``, ``.lines``, etc.) are only included if
        this DataFrame *can* support only 1 column+index.
        """
        return frozenset(_Utils.guess_io(False, cls._lines_files_apply(), {}).keys())

    @classmethod
    def can_write(cls) -> FrozenSet[str]:
        """
        Returns all filename extensions that can be written to using ``write_file``.
        Some, such as `.h5` and `.snappy`, are only included if their respective libraries
        were imported (when typeddfs was imported).
        The ``to_lines`` functions (``.txt``, ``.lines``, etc.) are only included if
        this DataFrame type *can* support only 1 column+index.
        """
        return frozenset(_Utils.guess_io(True, cls._lines_files_apply(), {}).keys())

    @classmethod
    def _guess_io(
        cls,
        clazz,
        writing: bool,
        path: Union[Path, str],
        all_params,
        *args,
        **kwargs,
    ) -> str:
        # we want nice error messages, so lie to _Utils.guess_io;
        # let it include .lines, etc., even if those won't work for this exact DF
        writing = False if writing is False else True
        inc_lines = cls._lines_files_apply()
        all_params = {p[0]: p[1] for p in all_params.items() if p is not _SENTINAL}
        path_is_a_path = isinstance(path, (str, PurePath))
        dct = _Utils.guess_io(writing, inc_lines, all_params)
        # `path` could be a URL, so don't use Path.suffix
        for suffix, (fn, params) in dct.items():
            if path_is_a_path and str(path).endswith(suffix):
                # Note the order! kwargs overwrites params
                # clazz.to_csv(path, sep="\t")
                my_kwargs = {**params, **kwargs}
                return getattr(clazz, fn)(path, *args, **my_kwargs)
        raise FilenameSuffixError(f"Suffix for {path} not recognized")

    def to_lines(
        self,
        path_or_buff,
        nl: Optional[str] = _SENTINAL,
    ) -> Optional[str]:
        r"""
        Writes a file that contains one row per line and 1 column per line.
        Associated with ``.lines`` or ``.txt``.

        .. caution::

            For technical reasons, values cannot contain a 6-em space (U+2008).
            Their presence will result in undefined behavior.

        Args:
            path_or_buff: Path or buffer
            nl: Forces using \n as the line separator

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        nl = {} if nl == _SENTINAL else dict(line_terminator="\n")
        df = self.vanilla_reset()
        if len(df.columns) != 1:
            raise NotSingleColumnError(f"Cannot write {len(df.columns)} columns ({df}) to lines")
        return df.to_csv(
            path_or_buff, index=False, sep=_FAKE_SEP, header=True, quoting=csv.QUOTE_NONE, **nl
        )

    @classmethod
    def read_lines(
        cls,
        path_or_buff,
        comment: Optional[str] = None,
        nl: Optional[str] = _SENTINAL,
    ) -> __qualname__:
        r"""
        Reads a file that contains 1 row and 1 column per line.
        Skips lines that are blank after trimming whitespace.
        Also skips comments if ``comment`` is set.

        .. caution::

            For technical reasons, values cannot contain a 6-em space (U+2008).
            Their presence will result in undefined behavior.

        Args:
            path_or_buff: Path or buffer
            comment: Any line starting with this substring (excluding spaces) is ignored;
                     no comment is used if empty
            nl: Forces using \n as the line separator (can almost always be inferred)
        """
        nl = {} if nl == _SENTINAL else dict(line_terminator="\n")
        try:
            df = pd.read_csv(
                path_or_buff,
                sep=_FAKE_SEP,
                header=0,
                index_col=False,
                quoting=csv.QUOTE_NONE,
                skip_blank_lines=True,
                comment=comment,
                encoding="utf8",
                engine="python",
                **nl,
            )
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if len(df.columns) > 1:
            raise NotSingleColumnError(f"Read multiple columns on {path_or_buff}")
        return cls._convert(df)

    @classmethod
    def read_flexwf(
        cls,
        path_or_buff,
        sep: str = "--->",
        comment: Optional[str] = None,
        nl: Optional[str] = _SENTINAL,
    ) -> __qualname__:
        r"""
        Reads a "flexible-width format".
        The delimiter (``sep``) is important.

        These are designed to read and write (``to_flexwf``) as though they
        were fixed-width. Specifically, all of the columns line up but are
        separated by a possibly multi-character delimiter.

        The files ignore blank lines, strip whitespace,
        always have a header, have no default index column
        (but can in ``required_columns()`` etc.), never quote values,

        Args:
            path_or_buff: Path or buffer
            sep: The delimiter, which can be a regex pattern
            comment: Prefix for comment lines
            nl: Forces using \n as the line separator (can almost always be inferred)
        """
        nl = {} if nl == _SENTINAL else dict(line_terminator="\n")
        try:
            df = pd.read_csv(
                path_or_buff,
                sep=sep,
                header=0,
                index_col=False,
                quoting=csv.QUOTE_NONE,
                skip_blank_lines=True,
                comment=comment,
                encoding="utf8",
                engine="python",
                **nl,
            )
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            try:
                df[c] = df[c].str.strip()
            except AttributeError:
                pass
        return cls._convert(df)

    def pretty_print(self, fmt: Union[str, TableFormat] = "plain", **kwargs) -> str:
        """
        Outputs a pretty table using the ``tabulate`` package.
        """
        return self._tabulate(fmt, **kwargs)

    def to_flexwf(self, path_or_buff, sep: str = "--->", mode: str = "w") -> Optional[str]:
        """
        Writes a fixed-width formatter, optionally with a delimiter, which can be multiple characters.

        See ``read_flexwf`` for more info.

        Args:
            path_or_buff: Path or buffer
            sep: The delimiter, 0 or more characters
            mode: write or append (w/a)

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        fmt = TableFormat(
            lineabove=None,
            linebelowheader=None,
            linebetweenrows=None,
            linebelow=None,
            headerrow=DataRow("", f" {sep} ", ""),
            datarow=DataRow("", f" {sep} ", ""),
            padding=0,
            with_header_hide=None,
        )
        content = self._tabulate(fmt)
        if path_or_buff is None:
            return content
        with get_handle(path_or_buff, mode, encoding="utf8", compression="infer") as f:
            f.handle.write(content)

    def _tabulate(self, fmt: Union[str, TableFormat], **kwargs) -> str:
        df = self.vanilla_reset()
        return tabulate(df.values.tolist(), list(df.columns), tablefmt=fmt, **kwargs)

    @classmethod
    def read_json(cls, *args, **kwargs) -> __qualname__:  # pragma: no cover
        try:
            df = pd.read_json(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert(df)

    def to_json(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        df = self.vanilla_reset()
        return df.to_json(path_or_buf, *args, **kwargs)

    @classmethod
    def read_feather(cls, *args, **kwargs) -> __qualname__:  # pragma: no cover
        # feather does not support MultiIndex, so reset index and use convert()
        try:
            df = pd.read_feather(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert(df)

    # noinspection PyMethodOverriding,PyBroadException,DuplicatedCode
    def to_feather(self, path_or_buf, *args, **kwargs) -> Optional[str]:  # pragma: no cover
        # feather does not support MultiIndex, so reset index and use convert()
        # if an error occurs you end up with a 0-byte file
        # this is fixed with exactly the same logic as for to_hdf -- see that method
        try:
            old_size = os.path.getsize(path_or_buf)
        except BaseException:
            old_size = None
        df = self.vanilla_reset()
        df.columns = df.columns.astype(str)
        try:
            return df.to_feather(path_or_buf, *args, **kwargs)
        except BaseException:
            try:
                size = os.path.getsize(path_or_buf)
            except BaseException:
                size = None
            if size is not None and size == 0 and (old_size is None or old_size > 0):
                try:
                    Path(path_or_buf).unlink()
                except BaseException:
                    pass
            raise

    @classmethod
    def read_parquet(cls, *args, **kwargs) -> __qualname__:  # pragma: no cover
        # parquet does not support MultiIndex, so reset index and use convert()
        try:
            df = pd.read_parquet(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert(df)

    # noinspection PyMethodOverriding,PyBroadException,DuplicatedCode
    def to_parquet(self, path_or_buf, *args, **kwargs) -> Optional[str]:  # pragma: no cover
        # parquet does not support MultiIndex, so reset index and use convert()
        # if an error occurs you end up with a 0-byte file
        # this is fixed with exactly the same logic as for to_hdf -- see that method
        try:
            old_size = os.path.getsize(path_or_buf)
        except BaseException:
            old_size = None
        reset = self.vanilla_reset()
        try:
            return reset.to_parquet(path_or_buf, *args, **kwargs)
        except BaseException:
            try:
                size = os.path.getsize(path_or_buf)
            except BaseException:
                size = None
            if size is not None and size == 0 and (old_size is None or old_size > 0):
                try:
                    Path(path_or_buf).unlink()
                except BaseException:
                    pass
            raise

    @classmethod
    def read_csv(cls, *args, **kwargs) -> __qualname__:
        """
        Reads from CSV, converting to this type.
        Using to_csv() and read_csv() from BaseFrame, this property holds::

            df.to_csv(path)
            df.__class__.read_csv(path) == df

        Passing ``index`` on ``to_csv`` or ``index_col`` on ``read_csv``
        explicitly will break this invariant.

        Args:
            args: Passed to ``pd.read_csv``; should start with a path or buffer
            kwargs: Passed to ``pd.read_csv``.
        """
        kwargs = dict(kwargs)
        # we want to set index=False, but we also want to let the user override
        # checking for index in the positional args
        # this is a really good case against positional arguments in languages
        # 'index_col' is in the 6th positional slot
        # that's ONLY IF we don't list the path as the first arg though!!!
        # if we added path_or_buf before `*args`, this would need to be < 5
        if len(args) < 6:
            kwargs.setdefault("index_col", False)
        try:
            df = pd.read_csv(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert(df)

    def to_csv(self, *args, bom: bool = False, **kwargs) -> Optional[str]:
        """
        Writes to CSV.
        Using to_csv() and read_csv() from BaseFrame, this property holds::

            df.to_csv(path)
            df.__class__.read_csv(path) == df

        Passing ``index`` on ``to_csv`` or ``index_col`` on ``read_csv``
        explicitly will break this invariant.

        Args:
            args: Passed to ``pd.read_csv``; should start with a path or buffer
            bom: Special flag to set the encoding to utf-8-sig on Windows but utf-8 otherwise.
                 This is useful because Windows often assumes ANSI (CP1252),
                 so many applications (esp. Excel) can't open it correctly without a BOM.
                 Passing `encoding` explicitly will override this.

            kwargs: Passed to ``pd.read_csv``.
        """
        kwargs = dict(kwargs)
        # same logic as for read_csv -- see that
        if len(args) < 7:
            kwargs.setdefault("index", False)
        if bom and len(args) < 10 and os.name == "nt":
            kwargs.setdefault("encoding", "utf-8-sig")
        elif bom and len(args) < 10:
            kwargs.setdefault("encoding", "utf-8")
        df = self.vanilla_reset()
        return df.to_csv(*args, **kwargs)

    @classmethod
    def read_hdf(cls, *args, key: str = "df", **kwargs) -> __qualname__:  # pragma: no cover
        """
        Reads from HDF with ``key`` as the default, converting to this type.

        Args:
            args: Passed; especially use ``path_or_buf``
            key: The HDF store key
            **kwargs: Passed to ``pd.DataFrame.to_hdf``

        Returns:
            A new instance of this class

        Raises:
            ImportError: If the ``tables`` package (pytables) is not available
            OSError: Likely for some HDF5 configurations
        """
        try:
            df = pd.read_hdf(*args, key=key, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert(df)

    # noinspection PyBroadException,PyFinal,DuplicatedCode
    def to_hdf(self, path: PathLike, key: str = "df", **kwargs) -> None:  # pragma: no cover
        """
        Writes to HDF with ``key`` as the default. Calling pd.to_hdf on this would error.

        Note:
            This handles an edge case in vanilla ``pd.DataFrame.to_hdf``
            that results in 0-byte files being written on error.
            Those empty files are deleted if they're created and didn't already exist.

        Args:
            path: A ``pathlib.Path`` or str value
            key: The HDF store key
            **kwargs: Passed to ``pd.DataFrame.to_hdf``

        Raises:
            ImportError: If the ``tables`` package (pytables) is not available
            OSError: Likely for some HDF5 configurations
        """
        path = Path(path)
        # if an error occurs you end up with a 0-byte file
        # delete it if and only if we CREATED an empty file --
        # subtle, but: we shouldn't delete the 0-byte file if it
        # already existed and was 0 bytes
        #
        # just wrap in try-except -- it might not be a file and might not exist
        # technically there's an edge case: what if it was just not readable?
        # if it isn't readable now but becomes readable (and writable) by the time
        # we try to write, then we delete it anyway
        # that's a super unlikely bug and shouldn't matter anyway
        try:
            old_size = os.path.getsize(path)
        except BaseException:
            old_size = None
        df = self.vanilla()
        try:
            df.to_hdf(str(path), key, **kwargs)
        except BaseException:
            # noinspection PyBroadException
            try:
                size = os.path.getsize(path)
            except BaseException:
                size = None
            if size is not None and size == 0 and (old_size is None or old_size > 0):
                try:
                    Path(path).unlink()
                except BaseException:
                    pass
            raise

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

    @classmethod
    def _lines_files_apply(cls) -> bool:
        if (
            hasattr(cls, "required_columns")
            and hasattr(cls, "reserved_columns")
            and hasattr(cls, "required_index_names")
            and hasattr(cls, "reserved_index_names")
        ):
            try:
                return (
                    len(
                        {
                            *cls.required_columns(),
                            *cls.reserved_columns(),
                            *cls.required_index_names(),
                            *cls.reserved_index_names(),
                        }
                    )
                    == 1
                )
            except (ValueError, TypeError):
                pass
        return True


class BaseDf(AbsDf, metaclass=abc.ABCMeta):
    """
    An extended DataFrame with ``convert()`` and ``vanilla()`` methods.
    """

    def __getitem__(self, item) -> __qualname__:
        if isinstance(item, str) and item in self.index.names:
            return self.index.get_level_values(item)
        else:
            return super().__getitem__(item)

    @classmethod
    def convert(cls, df: pd.DataFrame) -> __qualname__:
        """
        Converts a vanilla Pandas DataFrame to cls.
        Sets the index.

        Args:
            df: The Pandas DataFrame or member of cls; will have its __class_ change but will otherwise not be affected

        Returns:
            A copy
        """
        df = df.copy()
        df.__class__ = cls
        return df


__all__ = [
    "PrettyDf",
    "BaseDf",
    "InvalidDfError",
    "MissingColumnError",
    "UnexpectedColumnError",
    "UnexpectedIndexNameError",
    "AsymmetricDfError",
    "ExtraConditionFailedError",
    "NoValueError",
    "ValueNotUniqueError",
    "UnsupportedOperationError",
    "NonStrColumnError",
    "NotSingleColumnError",
]
