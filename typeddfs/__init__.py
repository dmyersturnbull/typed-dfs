# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Convenient code for import.
The only thing you need to import from ``typeddfs``.

Contains static factory methods to build new DataFrame subclasses.
In particular, see::

  - :meth:`typed`
  - :meth:`untyped`
  - :meth:`matrix`
  - :meth:`affinity_matrix`z
"""
import logging
from pathlib import Path

import pandas as pd

from typeddfs._meta import Metadata
from typeddfs.base_dfs import BaseDf
from typeddfs.builders import AffinityMatrixDfBuilder, MatrixDfBuilder, TypedDfBuilder
from typeddfs.datasets import ExampleDfs, LazyDf
from typeddfs.df_errors import (
    ClashError,
    FilenameSuffixError,
    InvalidDfError,
    MissingColumnError,
    NonStrColumnError,
    NotSingleColumnError,
    NoValueError,
    UnexpectedColumnError,
    UnexpectedIndexNameError,
    UnsupportedOperationError,
    ValueNotUniqueError,
    VerificationFailedError,
)
from typeddfs.file_formats import CompressionFormat, FileFormat
from typeddfs.frozen_types import FrozeDict, FrozeList, FrozeSet
from typeddfs.matrix_dfs import AffinityMatrixDf, MatrixDf
from typeddfs.typed_dfs import TypedDf
from typeddfs.untyped_dfs import UntypedDf
from typeddfs.utils import Utils
from typeddfs.utils.checksum_models import ChecksumFile, ChecksumMapping
from typeddfs.utils.checksums import Checksums

__version__ = Metadata.version
logger = logging.getLogger(Path(__file__).parent.name)


class FinalDf(UntypedDf):
    """An untyped DataFrame meant for general use."""


def example() -> type[TypedDf]:
    """
    Creates a new example TypedDf subclass.
    The class has:

        - required index "key"
        - required column "value"
        - reserved column "note"
        - no other columns

    Returns:
        The created class
    """
    # noinspection PyPep8Naming
    KeyValue = (
        typed("KeyValue")  # typed means enforced requirements
        .require("key", dtype=str, index=True)  # automagically make this an index
        .require("value", dtype=str)  # required
        .reserve("note")  # permitted but not required
        .strict()  # don't allow other columns
    ).build()
    return KeyValue


def wrap(df: pd.DataFrame) -> FinalDf:
    """
    Just wraps a DataFrame into a simple untyped DataFrame.
    Useful to quickly access a function only defined on typeddfs DataFrames.

    Example:
        ``TypedDfs.wrap(df).write_file("abc.feather")``
    """
    return FinalDf(df)


def typed(name: str, doc: str | None = None) -> TypedDfBuilder:
    """
    Creates a new type with flexible requirements.
    The class will enforce constraints and subclass :class:`typeddfs.typed_dfs.TypedDf`.

    Args:
        name: The name that will be used for the new class
        doc: The docstring for the new class

    Returns:
        A builder instance (builder pattern) to be used with chained calls

    Example:
        ``TypedDfs.typed("MyClass").require("name", index=True).build()``
    """
    return TypedDfBuilder(name, doc)


def matrix(name: str, doc: str | None = None) -> MatrixDfBuilder:
    """
    Creates a new subclass of an :class:`typeddfs.matrix_dfs.MatrixDf`.

    Args:
        name: The name that will be used for the new class
        doc: The docstring for the new class

    Returns:
        A builder instance (builder pattern) to be used with chained calls
    """
    return MatrixDfBuilder(name, doc)


def affinity_matrix(name: str, doc: str | None = None) -> AffinityMatrixDfBuilder:
    """
    Creates a new subclass of an :class:`typeddfs.matrix_dfs.AffinityMatrixDf`.

    Args:
        name: The name that will be used for the new class
        doc: The docstring for the new class

    Returns:
        A builder instance (builder pattern) to be used with chained calls
    """
    return AffinityMatrixDfBuilder(name, doc)


def untyped(name: str, doc: str | None = None) -> type[UntypedDf]:
    """
    Creates a new subclass of ``UntypedDf``.
    The returned class will not enforce constraints but will have some extra methods.
    In general :meth:`typed` should be preferred because it has more consistent behavior,
    especially for IO.

    Args:
        name: The name that will be used for the new class
        doc: The docstring for the new class

    Returns:
        A class instance

    Example:
        ``MyClass = TypedDfs.untyped("MyClass")``
    """

    class New(UntypedDf):
        pass

    New.__name__ = name
    New.__doc__ = doc
    return New


__all__ = [
    "AffinityMatrixDf",
    "BaseDf",
    "ChecksumFile",
    "ChecksumMapping",
    "Checksums",
    "CompressionFormat",
    "ExampleDfs",
    "FileFormat",
    "FinalDf",
    "FrozeDict",
    "FrozeList",
    "FrozeSet",
    "LazyDf",
    "MatrixDf",
    "TypedDf",
    "UntypedDf",
    "Utils",
    "affinity_matrix",
    "example",
    "matrix",
    "typed",
    "untyped",
    "wrap",
]
