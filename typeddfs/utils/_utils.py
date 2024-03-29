# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Internal utilities for typeddfs.
"""
from __future__ import annotations

from pathlib import PurePath

PathLike = str | PurePath


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


_DEFAULT_HASH_ALG = "sha256"


_DEFAULT_ATTRS_SUFFIX = ".attrs.json"


_PICKLE_VR = 5


_FLEXWF_SEP = r"|||"


_HDF_KEY = "df"


_TOML_AOT: str = "row"


_FAKE_SEP = "\u2008"  # 6-em space; very unlikely to occur


_AUTO_DROPPED_NAMES = {"index", "level_0", "Unnamed: 0"}
_FORBIDDEN_NAMES = {"__xml_is_empty_", "__xml_index_", "__feather_ignore_"}


__all__ = [
    "_FAKE_SEP",
    "_DEFAULT_HASH_ALG",
    "PathLike",
    "_SENTINEL",
    "_AUTO_DROPPED_NAMES",
    "_FORBIDDEN_NAMES",
    "_DEFAULT_ATTRS_SUFFIX",
]
