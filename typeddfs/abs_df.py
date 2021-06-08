"""
Defines a low-level DataFrame subclass that overrides
a lot of methods to auto-change the type back to ``cls``.
"""
from __future__ import annotations

import csv
import abc
import os
from pathlib import Path, PurePath
from typing import Optional, Union, FrozenSet

import pandas as pd
from pandas.io.common import get_handle

# noinspection PyProtectedMember
from tabulate import tabulate, TableFormat, DataRow

from typeddfs._core_dfs import CoreDf
from typeddfs._utils import _SENTINAL, _Utils, _FAKE_SEP, PathLike
from typeddfs.df_errors import NonStrColumnError, FilenameSuffixError, NotSingleColumnError


class AbsDf(CoreDf, metaclass=abc.ABCMeta):
    """
    A subclass of ``CoreDf`` that has new and overridden IO methods.
    This includes:
      - fixes to ``read_feather``, ``read_parquet``, and ``read_hdf``
      - support for auto-conversion of columns/index frames, for ``TypedDf``
        (which is a subclass of this class)
      - new methods ``read_file``, ``write_file``, and ``pretty_print``.
    """

    def write_file(self, path: Union[Path, str], *args, nl: Optional[str] = _SENTINAL, **kwargs):
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
            nl: Passes ``line_terminator=nl`` to ``.read_csv`` if the output is
                a CSV/TSV variant or .lines or .flexwf
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
        all_params = {}
        if nl != _SENTINAL:
            all_params["line_terminator"] = "\n"
        return cls._guess_io(self, True, path, all_params, *args, **kwargs)

    @classmethod
    def read_file(
        cls,
        path: Union[Path, str],
        *args,
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
            skip_blank_lines: If text-based, skip blank lines
            comment: Prefix indicating comments to ignore, if text-based
            args: Positional args passed to the read_ function
            kwargs: Keyword args passed to the function

        Returns:
            An instance of this class
        """
        all_params = {}
        if skip_blank_lines != _SENTINAL:
            all_params["skip_blank_lines"] = skip_blank_lines
        if comment != _SENTINAL:
            all_params["comment"] = comment
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

    def pretty_print(self, fmt: Union[str, TableFormat] = "plain", **kwargs) -> str:
        """
        Outputs a pretty table using the ``tabulate`` package.
        """
        return self._tabulate(fmt, **kwargs)

    @classmethod
    def read_flexwf(
        cls,
        path_or_buff,
        sep: str = r"\|\|\|",
        comment: Optional[str] = None,
        nl: Optional[str] = _SENTINAL,
    ) -> __qualname__:
        r"""
        Reads a "flexible-width format".
        The delimiter (``sep``) is important.
        **Note that ``sep`` is a regex pattern if it contains more than 1 char.**

        These are designed to read and write (``to_flexwf``) as though they
        were fixed-width. Specifically, all of the columns line up but are
        separated by a possibly multi-character delimiter.

        The files ignore blank lines, strip whitespace,
        always have a header, never quote values, and have no default index column
        unless given by ``required_columns()``, etc.

        Args:
            path_or_buff: Path or buffer
            sep: The delimiter, a regex pattern
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

    def to_flexwf(self, path_or_buff, sep: str = "|||", mode: str = "w") -> Optional[str]:
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

    @classmethod
    def read_json(cls, *args, **kwargs) -> __qualname__:  # pragma: no cover
        try:
            df = pd.read_json(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert(df)

    # noinspection PyFinal
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

    # noinspection PyFinal
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

    def _tabulate(self, fmt: Union[str, TableFormat], **kwargs) -> str:
        df = self.vanilla_reset()
        return tabulate(df.values.tolist(), list(df.columns), tablefmt=fmt, **kwargs)

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

    @classmethod
    def _lines_files_apply(cls) -> bool:
        if (
            hasattr(cls, "required_columns")
            and hasattr(cls, "reserved_columns")
            and hasattr(cls, "required_index_names")
            and hasattr(cls, "reserved_index_names")
            and hasattr(cls, "known_names")
        ):
            try:
                return len(cls.known_names()) == 1
            except (ValueError, TypeError):  # pragma: no cover
                pass
        return True


__all__ = ["AbsDf"]
