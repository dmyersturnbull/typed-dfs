"""
Internal utilities for typeddfs.
"""
from pathlib import PurePath
from typing import Union

PathLike = Union[str, PurePath]


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


_FAKE_SEP = "\u2008"  # 6-em space; very unlikely to occur


__all__ = ["_FAKE_SEP", "PathLike", "_SENTINEL"]
