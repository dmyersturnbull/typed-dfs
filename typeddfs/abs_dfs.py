"""
Defines a low-level DataFrame subclass.
It overrides a lot of methods to auto-change the type back to ``cls``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Union

from typeddfs._core_dfs import CoreDf
from typeddfs._mixins._full_io_mixin import _FullIoMixin
from typeddfs.df_errors import (
    HashEntryExistsError,
    HashFileExistsError,
    NonStrColumnError,
)
from typeddfs.df_typing import DfTyping
from typeddfs.file_formats import FileFormat
from typeddfs.utils import Utils
from typeddfs.utils.checksums import Checksums


class AbsDf(_FullIoMixin, CoreDf):
    @classmethod
    def read_url(cls, url: str) -> __qualname__:
        """
        Reads from a URL, guessing the format from the filename extension.
        Delegates to the ``read_*`` functions of this class.

        See Also:
            :meth:`read_file`

        Returns:
            An instance of this class
        """
        df = cls._call_read(cls, url)
        return cls._convert_typed(df)

    @classmethod
    def read_file(
        cls,
        path: Union[Path, str],
        *,
        file_hash: Optional[bool] = None,
        dir_hash: Optional[bool] = None,
        hex_hash: Optional[str] = None,
        attrs: Optional[bool] = None,
    ) -> __qualname__:
        """
        Reads from a file (or possibly URL), guessing the format from the filename extension.
        Delegates to the ``read_*`` functions of this class.

        You can always write and then read back to get the same dataframe.
        .. code-block::

            # df is any DataFrame from typeddfs
            # path can use any suffix
            df.write_file(path))
            df.read_file(path)

        Text files always allow encoding with .gz, .zip, .bz2, or .xz.

        Supports:
            - .csv, .tsv, or .tab
            - .json
            - .xml
            - .feather
            - .parquet or .snappy
            - .h5 or .hdf
            - .xlsx, .xls, .odf, etc.
            - .toml
            - .properties
            - .ini
            - .fxf (fixed-width)
            - .flexwf (fixed-but-unspecified-width with an optional delimiter)
            - .txt, .lines, or .list

        See Also:
            :meth:`read_url`
            :meth:`write_file`


        Args:
            path: Only path-like strings or pathlib objects are supported, not buffers
                  (because we need a filename).
            file_hash: Check against a hash file specific to this file (e.g. <path>.sha1)
            dir_hash: Check against a per-directory hash file
            hex_hash: Check against this hex-encoded hash
            attrs: Set dataset attributes/metadata (``pd.DataFrame.attrs``) from a JSON file.
                   If True, uses :attr:`typeddfs.df_typing.DfTyping.attrs_suffix`.
                   If a str or Path, uses that file.
                   If None or False, does not set.

        Returns:
            An instance of this class
        """
        if any((str(path).startswith(x + "://") for x in ["http", "https", "ftp"])):
            # just save some pain -- better than a weird error in .resolve()
            raise ValueError(f"Cannot read from URL {path}; use read_url instead")
        path = Path(path).resolve()
        t: DfTyping = cls.get_typing()
        if attrs is None:
            attrs = t.io.use_attrs
        cs = Checksums(alg=t.io.hash_algorithm)
        cs.verify_any(path, file_hash=file_hash, dir_hash=dir_hash, computed=hex_hash)
        df = cls._call_read(cls, path)
        if attrs:
            attrs_path = path.parent / (path.name + t.io.attrs_suffix)
            json_data = Utils.json_decoder().from_str(attrs_path.read_text(encoding="utf-8"))
            df.attrs.update(json_data)
        return cls._convert_typed(df)

    def write_file(
        self,
        path: Union[Path, str],
        *,
        overwrite: bool = True,
        mkdirs: bool = False,
        file_hash: Optional[bool] = None,
        dir_hash: Optional[bool] = None,
        attrs: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Writes to a file, guessing the format from the filename extension.
        Delegates to the ``to_*`` functions of this class (e.g. ``to_csv``).
        Only includes file formats that can be read back in with corresponding ``to`` methods.

        Supports, where text formats permit optional .gz, .zip, .bz2, or .xz:
            - .csv, .tsv, or .tab
            - .json
            - .feather
            - .fwf (fixed-width)
            - .flexwf (columns aligned but using a delimiter)
            - .parquet or .snappy
            - .h5, .hdf, or .hdf5
            - .xlsx, .xls, and other variants for Excel
            - .odt and .ods (OpenOffice)
            - .xml
            - .toml
            - .ini
            - .properties
            - .pkl and .pickle
            - .txt, .lines, or .list; see :meth:`to_lines` and :meth:`read_lines`

        See Also:
            :meth:`read_file`

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
            attrs: Write dataset attributes/metadata (``pd.DataFrame.attrs``) to a JSON file.
                   uses :attr:`typeddfs.df_typing.DfTyping.attrs_suffix`.
                   If None, chooses according to ``self.get_typing().io.use_attrs``.

        Returns:
            Whatever the corresponding method on ``pd.to_*`` returns.
            This is usually either str or None

        Raises:
            InvalidDfError: If the DataFrame is not valid for this type
            ValueError: If the type of a column or index name is non-str
        """
        if any((str(path).startswith(x + "://") for x in ["http", "https", "ftp"])):
            # just save some pain -- better than a weird error in .resolve()
            raise ValueError(f"Cannot write to URL {path}")
        path = Path(path).resolve()
        t = self.__class__.get_typing()
        file_hash = file_hash is True or file_hash is None and t.io.file_hash
        dir_hash = dir_hash is True or dir_hash is None and t.io.dir_hash
        attrs = attrs is True or attrs is None and t.io.use_attrs
        attrs_path = path.parent / (path.name + t.io.attrs_suffix)
        attrs_data = Utils.json_encoder().as_str(self.attrs)
        cs = Checksums(alg=t.io.hash_algorithm)
        file_hash_path = cs.get_filesum_of_file(path)
        dir_hash_path = cs.get_dirsum_of_file(path)
        # check for overwrite errors now to preserve atomicity
        if not overwrite:
            if path.exists():
                raise FileExistsError(f"File {path} already exists")
            if file_hash and file_hash_path.exists():
                raise HashFileExistsError(f"{file_hash_path} already exists")
            if dir_hash_path.exists():
                dir_sums = Checksums(alg=t.io.hash_algorithm).load_dirsum_exact(dir_hash_path)
                if path in dir_sums:
                    raise HashEntryExistsError(f"Path {path} listed in {dir_hash_path}")
            if file_hash and file_hash_path.exists():
                raise HashFileExistsError(f"{file_hash_path} already exists")
            if attrs and attrs_path.exists():
                raise FileExistsError(f"{attrs_path} already exists")
        self._check(self)
        types = set(self.column_names()).union(self.index_names())
        if any((not isinstance(c, str) for c in types)):
            raise NonStrColumnError(f"Columns must be of str type to serialize, not {types}")
        # now we're ready to write
        if mkdirs:
            path.parent.mkdir(exist_ok=True, parents=True)
        # to get a FileNotFoundError instead of a WritePermissionsError:
        if not mkdirs and not path.parent.exists():
            raise FileNotFoundError(f"Directory {path.parent} not found")
        # check for lack of write-ability to any of the files
        # we had to do this after creating the dirs unfortunately
        _all_files = [(attrs, attrs_path), (file_hash, file_hash_path), (dir_hash, dir_hash_path)]
        all_files = [f for a, f in _all_files if a]
        all_dirs = [f.parent for (a, f) in _all_files]
        # we need to check both the dirs and the files
        Utils.verify_can_write_dirs(*all_dirs, missing_ok=False)
        Utils.verify_can_write_files(*all_files, missing_ok=True)
        # we verified as much as we can -- finally we can write!!
        # this writes the main file:
        z = self._call_write(path)
        # write the hashes
        # this shouldn't fail
        cs = Checksums(alg=t.io.hash_algorithm)
        cs.write_any(
            path,
            to_file=file_hash,
            to_dir=dir_hash,
            overwrite=overwrite,
        )
        # write dataset attributes
        # this also shouldn't fail
        if attrs:
            attrs_path.write_text(attrs_data, encoding="utf8")
        return z

    @classmethod
    def can_read(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be read using ``read_file``.
        Some depend on the availability of optional packages.
        The lines format (``.txt``, ``.lines``, etc.) is only included if
        this DataFrame *can* support only 1 column+index.
        See :meth:`typeddfs.file_formats.FileFormat.can_read`.
        """
        return {
            f
            for f in FileFormat.all_readable()
            if (f is not FileFormat.lines or cls._lines_files_apply())
            and (f is not FileFormat.properties or cls._properties_files_apply())
        }

    @classmethod
    def can_write(cls) -> Set[FileFormat]:
        """
        Returns all formats that can be written to using ``write_file``.
        Some depend on the availability of optional packages.
        The lines format (``.txt``, ``.lines``, etc.) is only included if
        this DataFrame type *can* support only 1 column+index.
        See :meth:`typeddfs.file_formats.FileFormat.can_write`.
        """
        return {
            f
            for f in FileFormat.all_writable()
            if (f is not FileFormat.lines or cls._lines_files_apply())
            and (f not in [FileFormat.properties, FileFormat.ini] or cls._properties_files_apply())
        }

    @classmethod
    def from_records(
        cls,
        *args,
        **kwargs,
    ) -> __qualname__:
        return cls.convert(super().from_records(*args, **kwargs))

    @classmethod
    def _check(cls, df) -> None:
        """
        Should raise an :class:`typeddfs.df_errors.InvalidDfError` or subclass for issues.
        """


__all__ = ["AbsDf"]
