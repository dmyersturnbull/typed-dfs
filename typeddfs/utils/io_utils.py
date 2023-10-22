# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
"""
Tools for IO.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path, PurePath

from pandas.io.common import get_handle

from typeddfs.df_errors import (
    ReadPermissionsError,
    UnsupportedOperationError,
    WritePermissionsError,
)
from typeddfs.file_formats import CompressionFormat, FileFormat
from typeddfs.utils._utils import PathLike


class IoUtils:
    @classmethod
    def verify_can_read_files(
        cls,
        *paths: str | Path,
        missing_ok: bool = False,
        attempt: bool = False,
    ) -> None:
        """
        Checks that all files can be written to, to ensure atomicity before operations.

        Args:
            *paths: The files
            missing_ok: Don't raise an error if a path doesn't exist
            attempt: Actually try opening

        Returns:
            ReadPermissionsError: If a path is not a file (modulo existence) or doesn't have 'W' set
        """
        paths = [Path(p) for p in paths]
        for path in paths:
            if path.exists() and not path.is_file():
                raise ReadPermissionsError(f"Path {path} is not a file", key=str(path))
            if (not missing_ok or path.exists()) and not os.access(path, os.R_OK):
                raise ReadPermissionsError(f"Cannot read from {path}", key=str(path))
            if attempt:
                try:
                    with open(path):
                        pass
                except OSError:
                    raise WritePermissionsError(f"Failed to open {path} for read", key=str(path))

    @classmethod
    def verify_can_write_files(
        cls,
        *paths: str | Path,
        missing_ok: bool = False,
        attempt: bool = False,
    ) -> None:
        """
        Checks that all files can be written to, to ensure atomicity before operations.

        Args:
            *paths: The files
            missing_ok: Don't raise an error if a path doesn't exist
            attempt: Actually try opening

        Returns:
            WritePermissionsError: If a path is not a file (modulo existence) or doesn't have 'W' set
        """
        paths = [Path(p) for p in paths]
        for path in paths:
            if path.exists() and not path.is_file():
                raise WritePermissionsError(f"Path {path} is not a file", key=str(path))
            if (not missing_ok or path.exists()) and not os.access(path, os.W_OK):
                raise WritePermissionsError(f"Cannot write to {path}", key=str(path))
            if attempt:
                try:
                    with open(path, "a"):  # or w
                        pass
                except OSError:
                    raise WritePermissionsError(f"Failed to open {path} for write", key=str(path))

    @classmethod
    def verify_can_write_dirs(cls, *paths: str | Path, missing_ok: bool = False) -> None:
        """
        Checks that all directories can be written to, to ensure atomicity before operations.

        Args:
            *paths: The directories
            missing_ok: Don't raise an error if a path doesn't exist

        Returns:
            WritePermissionsError: If a path is not a directory (modulo existence) or doesn't have 'W' set
        """
        paths = [Path(p) for p in paths]
        for path in paths:
            if path.exists() and not path.is_dir():
                raise WritePermissionsError(f"Path {path} is not a dir", key=str(path))
            if missing_ok and not path.exists():
                continue
            if not os.access(path, os.W_OK):
                raise WritePermissionsError(f"{path} lacks write permission", key=str(path))
            if not os.access(path, os.X_OK):
                raise WritePermissionsError(f"{path} lacks access permission", key=str(path))

    @classmethod
    def write(
        cls, path_or_buff, content, *, mode: str = "w", atomic: bool = False, **kwargs
    ) -> str | None:
        """
        Writes using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        if path_or_buff is None:
            return content
        compression = cls.path_or_buff_compression(path_or_buff, kwargs)
        kwargs = {**kwargs, "compression": compression.pandas_value}
        if atomic and isinstance(path_or_buff, PathLike):
            if "a" in mode:
                raise UnsupportedOperationError("Can't append in atomic write")
            path = Path(path_or_buff)
            tmp = cls.tmp_path(path)
            with get_handle(tmp, mode, **kwargs) as f:
                f.handle.write(content)
                os.replace(tmp, path)
        with get_handle(path_or_buff, mode, **kwargs) as f:
            f.handle.write(content)

    @classmethod
    def read(cls, path_or_buff, *, mode: str = "r", **kwargs) -> str:
        """
        Reads using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix.
        (e.g. ``.csv.gz``).
        """
        compression = cls.path_or_buff_compression(path_or_buff, kwargs)
        kwargs = {**kwargs, "compression": compression.pandas_value}
        with get_handle(path_or_buff, mode, **kwargs) as f:
            return f.handle.read()

    @classmethod
    def path_or_buff_compression(cls, path_or_buff, kwargs) -> CompressionFormat:
        if "compression" in kwargs:
            return CompressionFormat.of(kwargs["compression"])
        elif isinstance(path_or_buff, (PurePath, str)):
            return CompressionFormat.from_path(path_or_buff)
        return CompressionFormat.none

    @classmethod
    def is_binary(cls, path: PathLike) -> bool:
        path = Path(path)
        if CompressionFormat.from_path(path).is_compressed:
            return True
        return FileFormat.from_path(path).is_binary

    @classmethod
    def tmp_path(cls, path: PathLike, extra: str = "tmp") -> Path:
        now = datetime.now().isoformat(timespec="ns").replace(":", "").replace("-", "")
        path = Path(path)
        suffix = "".join(path.suffixes)
        return path.parent / (".__" + extra + "." + now + suffix)

    @classmethod
    def get_encoding(cls, encoding: str = "utf-8") -> str:
        """
        Returns a text encoding from a more flexible string.
        Ignores hyphens and lowercases the string.
        Permits these nonstandard shorthands:

          - ``"platform"``: use ``sys.getdefaultencoding()`` on the fly
          - ``"utf8(bom)"``: use ``"utf-8-sig"`` on Windows; ``"utf-8"`` otherwise
          - ``"utf16(bom)"``: use ``"utf-16-sig"`` on Windows; ``"utf-16"`` otherwise
          - ``"utf32(bom)"``: use ``"utf-32-sig"`` on Windows; ``"utf-32"`` otherwise
        """
        encoding = encoding.lower().replace("-", "")
        if encoding == "platform":
            encoding = sys.getdefaultencoding()
        if encoding == "utf8(bom)":
            encoding = "utf-8-sig" if os.name == "nt" else "utf-8"
        if encoding == "utf16(bom)":
            encoding = "utf-16-sig" if os.name == "nt" else "utf-16"
        if encoding == "utf32(bom)":
            encoding = "utf-32-sig" if os.name == "nt" else "utf-32"
        return encoding

    @classmethod
    def get_encoding_errors(cls, errors: str | None) -> str | None:
        """
        Returns the value passed as``errors=`` in ``open``.
        Raises:
            ValueError: If invalid
        """
        if errors is None:
            return "strict"
        if errors in (
            "strict",
            "ignore",
            "replace",
            "xmlcharrefreplace",
            "backslashreplace",
            "namereplace",
            "surrogateescape",
            "surrogatepass",
        ):
            return errors
        raise ValueError(f"Invalid value {errors} for errors")


__all__ = ["IoUtils"]
