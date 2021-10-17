"""
Tools for IO.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

from pandas.io.common import get_handle

from typeddfs.df_errors import WritePermissionsError


class IoUtils:
    @classmethod
    def verify_can_write_files(cls, *paths: Union[str, Path], missing_ok: bool = False) -> None:
        """
        Checks that all files can be written to, to ensure atomicity before operations.

        Args:
            *paths: The files
            missing_ok: Don't raise an error if a path doesn't exist

        Returns:
            WritePermissionsError: If a path is not a file (modulo existence) or doesn't have 'W' set
        """
        paths = [Path(p) for p in paths]
        for path in paths:
            if path.exists() and not path.is_file():
                raise WritePermissionsError(f"Path {path} is not a file", key=str(path))
            if (not missing_ok or path.exists()) and not os.access(path, os.W_OK):
                raise WritePermissionsError(f"Cannot write to {path}", key=str(path))

    @classmethod
    def verify_can_write_dirs(cls, *paths: Union[str, Path], missing_ok: bool = False) -> None:
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
    def write(cls, path_or_buff, content, *, mode: str = "w", **kwargs) -> Optional[str]:
        """
        Writes using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        if path_or_buff is None:
            return content
        with get_handle(path_or_buff, mode, **kwargs) as f:
            f.handle.write(content)

    @classmethod
    def read(cls, path_or_buff, *, mode: str = "r", **kwargs) -> str:
        """
        Reads using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        with get_handle(path_or_buff, mode, **kwargs) as f:
            return f.handle.read()

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
    def get_encoding_errors(cls, errors: Optional[str]) -> Optional[str]:
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
