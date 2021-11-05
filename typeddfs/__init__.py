"""
Metadata and top-level declarations for typeddfs.
"""
from __future__ import annotations

import logging as __logging
from importlib.metadata import PackageNotFoundError as __PackageNotFoundError
from importlib.metadata import metadata as __load
from pathlib import Path

from typeddfs._entries import (
    AffinityMatrixDf,
    BaseDf,
    ChecksumFile,
    ChecksumMapping,
    Checksums,
    CompressionFormat,
    ExampleDfs,
    FileFormat,
    FinalDf,
    FrozeDict,
    FrozeList,
    FrozeSet,
    LazyDf,
    MatrixDf,
    TypedDf,
    TypedDfs,
    UntypedDf,
    Utils,
    affinity_matrix,
    example,
    matrix,
    typed,
    untyped,
    wrap,
)

logger = __logging.getLogger(Path(__file__).parent.name)
__pkg = Path(__file__).absolute().parent.name
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
except __PackageNotFoundError:  # pragma: no cover
    logger.error(f"Could not load package metadata for {__pkg}. Is it installed?")
