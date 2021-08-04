"""
Metadata and top-level declarations for typeddfs.
"""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import metadata as __load
from pathlib import Path

from typeddfs._entries import FinalDf, TypedDfs
from typeddfs.base_dfs import BaseDf
from typeddfs.file_formats import FileFormat
from typeddfs.matrix_dfs import MatrixDf, AffinityMatrixDf
from typeddfs.typed_dfs import TypedDf
from typeddfs.untyped_dfs import UntypedDf

logger = logging.getLogger(Path(__file__).parent.name)
pkg = Path(__file__).absolute().parent.name
metadata = None
try:
    metadata = __load(Path(__file__).parent.name)
    __status__ = "Development"
    __copyright__ = "Copyright 2016–2021"
    __date__ = "2020-08-29"
    __uri__ = metadata["home-page"]
    __title__ = metadata["name"]
    __summary__ = metadata["summary"]
    __license__ = metadata["license"]
    __version__ = metadata["version"]
    __author__ = metadata["author"]
    __maintainer__ = metadata["maintainer"]
    __contact__ = metadata["maintainer"]
except PackageNotFoundError:  # pragma: no cover
    logger.error(f"Could not load package metadata for {pkg}. Is it installed?")


__all__ = [
    "BaseDf",
    "UntypedDf",
    "TypedDf",
    "FinalDf",
    "MatrixDf",
    "AffinityMatrixDf",
    "TypedDfs",
    "FileFormat",
]
