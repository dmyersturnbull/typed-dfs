"""
Defines a low-level DataFrame subclass.
It overrides a lot of methods to auto-change the type back to ``cls``.
"""
from __future__ import annotations

import abc
import csv
import os
from pathlib import Path, PurePath
from typing import Any, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from tabulate import TableFormat, tabulate

from typeddfs._core_dfs import CoreDf
from typeddfs._utils import _FAKE_SEP, PathLike
from typeddfs.df_errors import (
    NonStrColumnError,
    NotSingleColumnError,
    ValueNotUniqueError,
    NoValueError,
    FormatInsecureError,
)
from typeddfs.df_typing import DfTyping
from typeddfs.file_formats import FileFormat
from typeddfs.utils import Utils

_SheetNamesOrIndices = Union[Sequence[Union[int, str]], int, str]


class AbsDf(CoreDf, metaclass=abc.ABCMeta):
    """
    An abstract DataFrame type with typing rules and IO methods.
    The method :py.meth:`get_typing` contains a plethora of typing rules
    that the type can choose how to (and whether to) enforce.
    """

    @classmethod
    def get_typing(cls) -> DfTyping:
        """
        Returns the info about how this DataFrame should be typed.
        Note that not all info is necessarily applicable, or even enforced by this subclass.
        """
        raise NotImplementedError()

    @classmethod
    def _check(cls, df) -> None:
        """
        Should raise an :py.class:`typeddfs.df_errors.InvalidDfError` or subclass for issues.
        """

    def pretty_print(self, fmt: Union[str, TableFormat] = "plain", **kwargs) -> str:
        """
        Outputs a pretty table using the `tabulate <https://pypi.org/project/tabulate/>`_ package.
        """
        return self._tabulate(fmt, **kwargs)

    @classmethod
    def read_file(
        cls, path: Union[Path, str], check_hash: Union[None, bool, str, PurePath] = False
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
            - .fxf (fixed-width)
            - .flexwf (fixed-but-unspecified-width with an optional delimiter)
            - .txt, .lines, or .list (optionally .gz, .zip, .bz2, or .xz); see ``read_lines()``

        Args:
            path: Only path-like strings or pathlib objects are supported, not buffers
                  (because we need a filename).
            check_hash: Require a hash file for the file or directory and make sure it matches.
                        Must be one of these:
                          - False / "no" -- do not check
                          - True / "yes" -- check either a file hash or a dir hash
                          - None -- Treat as "yes" if dictated by ``self.get_typing().io``
                          - "file" -- check a file hash (ignoring the filename in the hash file)
                          - "dir" -- check a dir hash (looking up the filename in the hash file)
                          - str or Path (extant); a path to the hash file (does not check the filename)
                          - str matching [A-Ha-h0-9]+; a hex-encoded digest

        Returns:
            An instance of this class
        """
        path = Path(path)
        cls._check_hash(path, check_hash)
        df = cls._call_read(cls, path)
        return cls._convert_typed(df)

    def write_file(
        self,
        path: Union[Path, str],
        overwrite: bool = True,
        mkdirs: bool = False,
        file_hash: Optional[bool] = None,
        dir_hash: Optional[bool] = None,
    ) -> Optional[str]:
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
            overwrite: If False, complain if the file already exists
            mkdirs: Make the directory and parents if they do not exist
            file_hash: Write a hash for this file.
                       The filename will be path+"."+algorithm.
                       If None, chooses according to ``self.get_typing().io.hash_file``.
            dir_hash: Append a hash for this file into a list.
                       The filename will be the directory name suffixed by the algorithm;
                       (i.e. path.parent/(path.parent.name+"."+algorithm) ).
                       If None, chooses according to ``self.get_typing().io.hash_dir``.

        Returns:
            Whatever the corresponding method on ``pd.to_*`` returns.
            This is usually either str or None

        Raises:
            InvalidDfError: If the DataFrame is not valid for this type
            ValueError: If the type of a column or index name is non-str
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists")
        self._check(self)
        types = set(self.column_names()).union(self.index_names())
        if any((not isinstance(c, str) for c in types)):
            raise NonStrColumnError(f"Columns must be of str type to serialize, not {types}")
        if mkdirs:
            path.parent.mkdir(exist_ok=True, parents=True)
        z = self._call_write(path)
        file_hash = file_hash is True or file_hash is None and self.get_typing().io.file_hash
        dir_hash = dir_hash is True or dir_hash is None and self.get_typing().io.dir_hash
        Utils.add_any_hashes(
            path,
            to_file=file_hash,
            to_dir=dir_hash,
            algorithm=self.get_typing().io.hash_algorithm,
            overwrite=overwrite,
        )
        return z

    @classmethod
    def can_read(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be read using ``read_file``.
        Some depend on the availability of optional packages.
        The lines format (``.txt``, ``.lines``, etc.) is only included if
        this DataFrame *can* support only 1 column+index.
        See :py.meth:`typeddfs.file_formats.FileFormat.can_read`.
        """
        return {
            f
            for f in FileFormat.all_readable()
            if f is not FileFormat.lines or cls._lines_files_apply()
        }

    @classmethod
    def can_write(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be written to using ``write_file``.
        Some depend on the availability of optional packages.
        The lines format (``.txt``, ``.lines``, etc.) is only included if
        this DataFrame type *can* support only 1 column+index.
        See :py.meth:`typeddfs.file_formats.FileFormat.can_write`.
        """
        return {
            f
            for f in FileFormat.all_writable()
            if f is not FileFormat.lines or cls._lines_files_apply()
        }

    def to_lines(
        self,
        path_or_buff,
        mode: str = "w",
        **kwargs,
    ) -> Optional[str]:
        r"""
        Writes a file that contains one row per line and 1 column per line.
        Associated with ``.lines`` or ``.txt``.

        .. caution::

            For technical reasons, values cannot contain a 6-em space (U+2008).
            Their presence will result in undefined behavior.

        Args:
            path_or_buff: Path or buffer
            mode: Write ('w') or append ('a')
            kwargs: Passed to ``to_csv``

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("header", True)
        df = self.vanilla_reset()
        if len(df.columns) != 1:
            raise NotSingleColumnError(f"Cannot write {len(df.columns)} columns ({df}) to lines")
        return df.to_csv(
            path_or_buff, mode=mode, index=False, sep=_FAKE_SEP, quoting=csv.QUOTE_NONE, **kwargs
        )

    @classmethod
    def read_lines(
        cls,
        path_or_buff,
        **kwargs,
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
            kwargs: Passed to ``read_csv``; May include 'comment', 'encoding', 'skip_blank_lines', and 'line_terminator'
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("skip_blank_lines", True)
        kwargs.setdefault("header", 0)
        kwargs.setdefault("engine", "python")
        try:
            df = pd.read_csv(
                path_or_buff,
                sep=_FAKE_SEP,
                index_col=False,
                quoting=csv.QUOTE_NONE,
                **kwargs,
            )
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if len(df.columns) > 1:
            raise NotSingleColumnError(f"Read multiple columns on {path_or_buff}")
        return cls._convert_typed(df)

    @classmethod
    def read_fwf(cls, *args, **kwargs) -> __qualname__:
        try:
            return cls._convert_typed(pd.read_fwf(*args, **kwargs))
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert_typed(df)

    def to_fwf(
        self,
        path_or_buff,
        mode: str = "w",
        colspecs: Optional[Sequence[Tuple[int, int]]] = None,
        widths: Optional[Sequence[int]] = None,
        na_rep: Optional[str] = None,
        float_format: Optional[str] = None,
        date_format: Optional[str] = None,
        decimal: str = ".",
        **kwargs,
    ) -> Optional[str]:
        """
        Writes a fixed-width text format.
        See ``read_fwf`` and ``to_flexwf`` for more info.

        .. warning:

            This method is a preview. Not all options are complete, and
            behavior is subject to change in a future (major) version.
            Notably, Pandas may eventually introduce a method with the same name.

        Args:
            path_or_buff: Path or buffer
            mode: write or append (w/a)
            colspecs: A list of tuples giving the extents of the fixed-width fields of each line
                      as half-open intervals (i.e., [from, to[ )
            widths: A list of field widths which can be used instead of ``colspecs``
                   if the intervals are contiguous
            na_rep: Missing data representation
            float_format: Format string for floating point numbers
            date_format: Format string for datetime objects
            decimal: Character recognized as decimal separator. E.g. use ‘,’ for European data.

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        if colspecs is not None and widths is not None:
            raise ValueError("Both widths and colspecs passed")
        if widths is not None:
            colspecs = []
            at = 0
            for w in widths:
                colspecs.append((at, at + w))
                at += w
        # if colspecs is None:
        if True:
            # TODO: use format, etc.
            content = self._tabulate(Utils.plain_table_format(" "), disable_numparse=True)
        else:
            df = self.vanilla_reset()
            if len(df.columns) != len(colspecs):
                raise ValueError(f"{colspecs} column intervals for {len(df.columns)} columns")
            for col, (start, end) in zip(df.columns, colspecs):
                width = end - start
                mx = df[col].map(str).map(len).max()
                if mx > width:
                    raise ValueError(f"Column {col} has max length {mx} > {end-start}")
            _number_format = {
                "na_rep": na_rep,
                "float_format": float_format,
                "date_format": date_format,
                "quoting": csv.QUOTE_NONE,
                "decimal": decimal,
            }
            res = df._mgr.to_native_types(**_number_format)
            data: Sequence[Sequence[Any]] = [res.iget_values(i) for i in range(len(res.items))]
            content = None  # TODO
        if path_or_buff is None:
            return content
        _encoding = dict(encoding=kwargs.get("encoding")) if "encoding" in kwargs else {}
        _compression = dict(encoding=kwargs.get("compression")) if "compression" in kwargs else {}
        Utils.write(path_or_buff, content, mode, **_encoding, **_compression)

    @classmethod
    def read_flexwf(
        cls,
        path_or_buff,
        sep: str = r"\|\|\|",
        **kwargs,
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
            kwargs: Passed to ``read_csv``; may include 'comment' and 'skip_blank_lines'
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("skip_blank_lines", True)
        try:
            df = pd.read_csv(
                path_or_buff,
                sep=sep,
                index_col=False,
                quoting=csv.QUOTE_NONE,
                engine="python",
                header=0,
                **kwargs,
            )
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            try:
                df[c] = df[c].str.strip()
            except AttributeError:
                pass
        return cls._convert_typed(df)

    def to_flexwf(self, path_or_buff, sep: str = "|||", mode: str = "w", **kwargs) -> Optional[str]:
        """
        Writes a fixed-width formatter, optionally with a delimiter, which can be multiple characters.

        See ``read_flexwf`` for more info.

        Args:
            path_or_buff: Path or buffer
            sep: The delimiter, 0 or more characters
            mode: write or append (w/a)
            kwargs: Passed to ``Utils.write``; may include 'encoding'

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        fmt = Utils.plain_table_format(" " + sep + " ")
        content = self._tabulate(fmt, disable_numparse=True)
        if path_or_buff is None:
            return content
        Utils.write(path_or_buff, content, mode, **kwargs)

    @classmethod
    def read_excel(cls, io, sheet_name: _SheetNamesOrIndices = 0, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_excel(io, sheet_name, *args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        # This only applies for .xlsb -- the others don't have this problem
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        return cls._convert_typed(df)

    # noinspection PyFinal,PyMethodOverriding
    def to_excel(self, excel_writer, *args, **kwargs) -> Optional[str]:
        kwargs = dict(kwargs)
        df = self.vanilla_reset()
        if isinstance(excel_writer, (str, PurePath)) and Path(excel_writer).suffix in [
            ".xls",
            ".ods",
            ".odt",
            ".odf",
        ]:
            # Pandas's defaults for XLS and ODS are so buggy that we should never use them
            # that is, unless the user actually passes engine=
            kwargs.setdefault("engine", "openpyxl")
        return df.to_excel(excel_writer, *args, **kwargs)

    @classmethod
    def read_xlsx(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads XLSX Excel files.
        Prefer this method over :py.meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_xlsx(self, excel_writer, *args, **kwargs) -> Optional[str]:
        """
        Writes XLSX Excel files.
        Prefer this method over :py.meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(excel_writer, *args, **kwargs)

    @classmethod
    def read_xls(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads legacy XLS Excel files.
        Prefer this method over :py.meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_xls(self, excel_writer, *args, **kwargs) -> Optional[str]:
        """
        Reads legacy XLS Excel files.
        Prefer this method over :py.meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(excel_writer, *args, **kwargs, engine="openpyxl")

    @classmethod
    def read_xlsb(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads XLSB Excel files.
        This is a relatively uncommon format.
        Prefer this method over :py.meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_xlsb(self, excel_writer, *args, **kwargs) -> Optional[str]:
        """
        Writes XLSB Excel files.
        This is a relatively uncommon format.
        Prefer this method over :py.meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(excel_writer, *args, **kwargs)

    @classmethod
    def read_ods(cls, io, sheet_name: _SheetNamesOrIndices = 0, **kwargs) -> __qualname__:
        """
        Reads OpenDocument ODS/ODT files.
        Prefer this method over :py.meth:`read_excel`.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return cls.read_excel(io, sheet_name, **kwargs, engine="openpyxl")

    def to_ods(self, ods_writer, *args, **kwargs) -> Optional[str]:
        """
        Writes OpenDocument ODS/ODT files.
        Prefer this method over :py.meth:`write_excel`.
        """
        # ignore the deprecated option, for symmetry with read_
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
        return self.to_excel(ods_writer, *args, **kwargs)

    @classmethod
    def read_pickle(cls, filepath_or_buffer, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_pickle(filepath_or_buffer, *args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert_typed(df)

    # noinspection PyFinal
    def to_pickle(self, path, *args, **kwargs) -> None:
        df = self.vanilla()
        return df.to_pickle(path, *args, **kwargs)

    @classmethod
    def read_json(cls, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_json(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert_typed(df)

    @classmethod
    def read_xml(cls, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_xml(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        # see to_xml for why these fixes are needed
        if "__xml_is_empty_" in df.reset_index().columns:
            df = pd.DataFrame()
        elif "__xml_index_" in df.columns:
            df = df.drop(columns={"__xml_index_"})
        return cls._convert_typed(df)

    # noinspection PyFinal,PyMethodOverriding
    def to_xml(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        # Pandas's to_xml and read_xml have two buggy properties:
        # 1. Unnamed indices are called "index"
        #    for to_xml, but not read_xml -- so they're not inverses.
        #    We'll fix that by changing the index to "__xml_index_"
        # 2. Writing an empty DataFrame results in a KeyError from deep inside
        #    We'll fix that by replacing the empty DataFrame with a DataFrame
        #   containing column "__xml_is_empty_" with a single row with the same value
        # in the insanely unlikely situation that these exist, complain
        if "__xml_is_empty_" in self.column_names() or "__xml_is_empty_" in self.index_names():
            raise ValueError("Do not include a column called '__xml_is_empty_'")
        if "__xml_index_" in self.column_names() or "__xml_index_" in self.index_names():
            raise ValueError("Do not include a column called '__xml_index_'")
        df = self.vanilla()
        if len(df) == 0 == len(self.index_names()) == len(self.column_names()) == 0:
            series = pd.Series({"__xml_is_empty_": "__xml_is_empty_"})
            df = pd.DataFrame([series])
        elif len(self.index_names()) == 0:
            df.index = df.index.rename("__xml_index_")
        return df.to_xml(path_or_buf, *args, **kwargs)

    # noinspection PyFinal,PyMethodOverriding
    def to_json(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        df = self.vanilla_reset()
        return df.to_json(path_or_buf, *args, **kwargs)

    @classmethod
    def read_feather(cls, *args, **kwargs) -> __qualname__:
        # feather does not support MultiIndex, so reset index and use convert()
        try:
            df = pd.read_feather(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if "__feather_ignore_" in df.columns:
            df = df.drop("__feather_ignore_", axis=1)
        return cls._convert_typed(df)

    # noinspection PyMethodOverriding,PyBroadException,DuplicatedCode
    def to_feather(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        # feather does not support MultiIndex, so reset index and use convert()
        # if an error occurs you end up with a 0-byte file
        # this is fixed with exactly the same logic as for to_hdf -- see that method
        try:
            old_size = os.path.getsize(path_or_buf)
        except BaseException:
            old_size = None
        df = self.vanilla_reset()
        if len(df) == len(df.columns) == 0:
            df = df.append(
                pd.Series(dict(__feather_ignore_="__feather_ignore_")), ignore_index=True
            )
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
    def read_parquet(cls, *args, **kwargs) -> __qualname__:
        # parquet does not support MultiIndex, so reset index and use convert()
        try:
            df = pd.read_parquet(*args, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert_typed(df)

    # noinspection PyMethodOverriding,PyBroadException,DuplicatedCode
    def to_parquet(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        # parquet does not support MultiIndex, so reset index and use convert()
        # if an error occurs you end up with a 0-byte file
        # this is fixed with exactly the same logic as for to_hdf -- see that method
        try:
            old_size = os.path.getsize(path_or_buf)
        except BaseException:
            old_size = None
        reset = self.vanilla_reset()
        for c in reset.columns:
            if reset[c].dtype in [
                np.byte,
                np.ubyte,
                np.short,
                np.ushort,
                np.single,
                np.int32,
                np.intc,
            ]:
                reset[c] = reset[c].astype(np.intc)
            elif reset[c].dtype in [np.intc, np.uintc]:
                reset[c] = reset[c].astype(np.long)
            elif reset[c].dtype in [np.half, np.float16, np.single, np.float32]:
                reset[c] = reset[c].astype(np.float32)
            elif reset[c].dtype in [np.double, np.float64]:
                reset[c] = reset[c].astype(np.float64)
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
    def read_tsv(cls, *args, **kwargs) -> __qualname__:
        """
        Reads tab-separated data.
        See  :py.meth:`read_csv` for more info.
        """
        kwargs = {k: v for k, v in kwargs.items() if k != "sep"}
        return cls.read_csv(*args, sep="\t", **kwargs)

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
        return cls._convert_typed(df)

    def to_tsv(self, *args, **kwargs) -> Optional[str]:
        """
        Writes tab-separated data.
        See :py.meth:`to_csv` for more info.
        """
        return self.to_csv(*args, sep="\t", **kwargs)

    # noinspection PyFinal
    def to_csv(self, *args, **kwargs) -> Optional[str]:
        kwargs = dict(kwargs)
        kwargs.setdefault("index", False)
        df = self.vanilla_reset()
        return df.to_csv(*args, **kwargs)

    @classmethod
    def read_hdf(
        cls, *args, key: Optional[str] = None, **kwargs
    ) -> __qualname__:  # pragma: no cover
        if key is None:
            key = cls.get_typing().io.hdf_key
        try:
            df = pd.read_hdf(*args, key=key, **kwargs)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return cls._convert_typed(df)

    def to_html(self, *args, **kwargs) -> Optional[str]:
        df = self.vanilla_reset()
        return df.to_html(*args, **kwargs)

    def to_rst(
        self, path_or_none: Optional[PathLike], style: str = "simple", mode: str = "w"
    ) -> Optional[str]:
        """
        Writes a reStructuredText table.
        Args:
            path_or_none: Either a file path or ``None`` to return the string
            style: The type of table; currently only "simple" is supported
            mode: Write mode
        """
        txt = self._tabulate(fmt="rst") + "\n"
        return Utils.write(path_or_none, txt, mode=mode)

    def to_markdown(self, *args, **kwargs) -> Optional[str]:
        return super().to_markdown(*args, **kwargs)

    @classmethod
    def read_html(cls, path: PathLike, *args, **kwargs) -> __qualname__:
        """
        Similar to ``pd.read_html``, but requires exactly 1 table and returns it.

        Raises:
            lxml.etree.XMLSyntaxError: If the HTML could not be parsed
            NoValueError: If no tables are found
            ValueNotUniqueError: If multiple tables are found
        """
        try:
            dfs = pd.read_html(path, *args, **kwargs)
        except ValueError as e:
            if str(e) == "No tables found":
                raise NoValueError(f"No tables in {path}") from None
            raise
        if len(dfs) > 1:
            raise ValueNotUniqueError(f"{len(dfs)} tables in {path}")
        df = dfs[0]
        if "Unnamed: 0" in df:
            df = df.drop("Unnamed: 0", axis=1)
        return cls._convert_typed(df)

    # noinspection PyBroadException,PyFinal,DuplicatedCode
    def to_hdf(
        self, path: PathLike, key: Optional[str] = None, **kwargs
    ) -> None:  # pragma: no cover
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
        if key is None:
            key = self.__class__.get_typing().io.hdf_key
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
    def _call_read(
        cls,
        clazz,
        path: Union[Path, str],
    ) -> pd.DataFrame:
        t = cls.get_typing().io
        mp = FileFormat.suffix_map()
        mp.update(t.remap_suffixes)
        fmt = FileFormat.from_path(path, format_map=mp)
        if t.secure and not fmt.is_secure:
            raise FormatInsecureError(f"Insecure format {fmt} forbidden by typing")
        fn_name = "read_" + fmt.name
        kwargs = cls._get_read_kwargs(fmt)
        fn = getattr(clazz, fn_name)
        return fn(path, **kwargs)

    def _call_write(
        self,
        path: Union[Path, str],
    ) -> Optional[str]:
        cls = self.__class__
        t = cls.get_typing().io
        mp = FileFormat.suffix_map()
        mp.update(t.remap_suffixes)
        fmt = FileFormat.from_path(path, format_map=mp)
        if t.secure and not fmt.is_secure:
            raise FormatInsecureError(f"Insecure format {fmt} forbidden by typing")
        fn_name = "to_" + fmt.name
        kwargs = cls._get_write_kwargs(fmt)
        fn = getattr(self, fn_name)
        return fn(path, **kwargs)

    @classmethod
    def _get_read_kwargs(cls, fmt: FileFormat) -> Mapping[str, Any]:
        t = cls.get_typing().io
        kwargs = t.read_kwargs.get(fmt, {})
        if fmt in [
            FileFormat.csv,
            FileFormat.csv,
            FileFormat.lines,
            FileFormat.flexwf,
            FileFormat.fwf,
        ]:
            encoding = kwargs.get("encoding", t.text_encoding)
            kwargs["encoding"] = Utils.get_encoding(encoding)
        return kwargs

    @classmethod
    def _check_hash(cls, path: Path, check_hash: Union[None, bool, str, PurePath]):
        t = cls.get_typing().io
        if check_hash is None and t.file_hash:
            check_hash = "file"
        elif check_hash is None and t.dir_hash:
            check_hash = "dir"
        elif check_hash is None:
            check_hash = "no"
        Utils.verify_any_hash(path, check_hash, algorithm=t.hash_algorithm)

    @classmethod
    def _get_write_kwargs(cls, fmt: FileFormat) -> Mapping[str, Any]:
        t = cls.get_typing().io
        kwargs = t.write_kwargs.get(fmt, {})
        if fmt is FileFormat.json:
            # TODO: not perfect
            kwargs["force_ascii"] = False  # "utf" not in t.text_encoding
        elif fmt.supports_encoding:
            encoding = kwargs.get("encoding", t.text_encoding)
            kwargs["encoding"] = Utils.get_encoding(encoding)
        return kwargs

    @classmethod
    def _lines_files_apply(cls) -> bool:
        return len(cls.get_typing().required_names) <= 1


__all__ = ["AbsDf"]
