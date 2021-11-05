"""
Convenient code for import.
"""
import logging
from pathlib import Path
from typing import Optional, Type

import pandas as pd

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

logger = logging.getLogger(Path(__file__).parent.name)


class FinalDf(UntypedDf):
    """An untyped DataFrame meant for general use."""


class TypedDfs:
    """
    The only thing you need to import from ``typeddfs``.

    Contains static factory methods to build new DataFrame subclasses.
    In particular, see::

      - :meth:`typed`
      - :meth:`untyped`
      - :meth:`matrix`
      - :meth:`affinity_matrix`
    """

    NoValueError = NoValueError
    ValueNotUniqueError = ValueNotUniqueError
    InvalidDfError = InvalidDfError
    MissingColumnError = MissingColumnError
    UnexpectedColumnError = UnexpectedColumnError
    UnexpectedIndexNameError = UnexpectedIndexNameError
    UnsupportedOperationError = UnsupportedOperationError
    FilenameSuffixError = FilenameSuffixError
    NonStrColumnError = NonStrColumnError
    NotSingleColumnError = NotSingleColumnError
    ClashError = ClashError
    VerificationFailedError = VerificationFailedError
    FileFormat = FileFormat
    CompressionFormat = CompressionFormat
    FinalDf = FinalDf
    Utils = Utils
    Checksums = Checksums
    FrozeList = FrozeList
    FrozeSet = FrozeSet
    FrozeDict = FrozeDict

    _logger = logger

    @classmethod
    def example(cls) -> Type[TypedDf]:
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
            TypedDfs.typed("KeyValue")  # typed means enforced requirements
            .require("key", dtype=str, index=True)  # automagically make this an index
            .require("value", dtype=str)  # required
            .reserve("note")  # permitted but not required
            .strict()  # don't allow other columns
        ).build()
        return KeyValue

    @classmethod
    def wrap(cls, df: pd.DataFrame) -> FinalDf:
        """
        Just wraps a DataFrame into a simple untyped DataFrame.
        Useful to quickly access a function only defined on typeddfs DataFrames.

        Example:
            ``TypedDfs.wrap(df).write_file("abc.feather")``
        """
        return FinalDf(df)

    @classmethod
    def typed(cls, name: str, doc: Optional[str] = None) -> TypedDfBuilder:
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

    @classmethod
    def matrix(cls, name: str, doc: Optional[str] = None) -> MatrixDfBuilder:
        """
        Creates a new subclass of an :class:`typeddfs.matrix_dfs.MatrixDf`.

        Args:
            name: The name that will be used for the new class
            doc: The docstring for the new class

        Returns:
            A builder instance (builder pattern) to be used with chained calls
        """
        return MatrixDfBuilder(name, doc)

    @classmethod
    def affinity_matrix(cls, name: str, doc: Optional[str] = None) -> AffinityMatrixDfBuilder:
        """
        Creates a new subclass of an :class:`typeddfs.matrix_dfs.AffinityMatrixDf`.

        Args:
            name: The name that will be used for the new class
            doc: The docstring for the new class

        Returns:
            A builder instance (builder pattern) to be used with chained calls
        """
        return AffinityMatrixDfBuilder(name, doc)

    @classmethod
    def untyped(cls, name: str, doc: Optional[str] = None) -> Type[UntypedDf]:
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


typed = TypedDfs.typed
untyped = TypedDfs.untyped
matrix = TypedDfs.matrix
affinity_matrix = TypedDfs.affinity_matrix
wrap = TypedDfs.wrap
example = TypedDfs.example


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
    "TypedDfs",
]
