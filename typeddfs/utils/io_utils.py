# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Tools for IO.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

from pandas.io.common import get_handle

from typeddfs.df_errors import (
    ReadPermissionsError,
    UnsupportedOperationError,
    WritePermissionsError,
)
from typeddfs.file_formats import CompressionFormat, FileFormat
from typeddfs.utils._utils import PathLike

if TYPE_CHECKING:
    from pandas._typing import BaseBuffer, FilePath


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
                msg = f"Path {path} is not a file"
                raise ReadPermissionsError(msg, key=str(path))
            if (not missing_ok or path.exists()) and not os.access(path, os.R_OK):
                msg = f"Cannot read from {path}"
                raise ReadPermissionsError(msg, key=str(path))
            if attempt:
                try:
                    with path.open():
                        pass
                except OSError as e:
                    msg = f"Failed to open {path} for read"
                    raise WritePermissionsError(msg, key=str(path)) from e

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
                msg = f"Path {path} is not a file"
                raise WritePermissionsError(msg, key=str(path))
            if (not missing_ok or path.exists()) and not os.access(path, os.W_OK):
                msg = f"Cannot write to {path}"
                raise WritePermissionsError(msg, key=str(path))
            if attempt:
                try:
                    with path.open("a"):  # or w
                        pass
                except OSError as e:
                    msg = f"Failed to open {path} for write"
                    raise WritePermissionsError(msg, key=str(path)) from e

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
                msg = f"Path {path} is not a dir"
                raise WritePermissionsError(msg, key=str(path))
            if missing_ok and not path.exists():
                continue
            if not os.access(path, os.W_OK):
                msg = f"{path} lacks write permission"
                raise WritePermissionsError(msg, key=str(path))
            if not os.access(path, os.X_OK):
                msg = f"{path} lacks access permission"
                raise WritePermissionsError(msg, key=str(path))

    @classmethod
    def write(
        cls,
        path_or_buff: FilePath | BaseBuffer,
        content,
        *,
        mode: str = "w",
        atomic: bool = False,
        compression_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | None:
        """
        Writes using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        if compression_kwargs is None:
            compression_kwargs = {}
        if atomic and "a" in mode:
            msg = "Can't append in atomic write"
            raise UnsupportedOperationError(msg)
        if path_or_buff is None:
            return content
        compression = cls.path_or_buff_compression(path_or_buff, kwargs)
        kwargs = {**kwargs, "compression": compression.pandas_value}
        if atomic and isinstance(path_or_buff, PathLike):
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
        elif isinstance(path_or_buff, PurePath | str):
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
        e = encoding.lower().replace("-", "")
        if e == "platform":
            return sys.getdefaultencoding()
        if e == "utf8(bom)":
            return "utf-8-sig" if os.name == "nt" else "utf-8"
        if e == "utf16(bom)":
            return "utf-16-sig" if os.name == "nt" else "utf-16"
        if e == "utf32(bom)":
            return "utf-32-sig" if os.name == "nt" else "utf-32"
        if e in {"utf8", "utf-8"}:
            return "utf-8"
        if e in {"utf16", "utf-16"}:
            return "utf-16"
        if e in {"utf32", "utf-32"}:
            return "utf-32"
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
        msg = f"Invalid value {errors} for errors"
        raise ValueError(msg)


__all__ = ["IoUtils"]
