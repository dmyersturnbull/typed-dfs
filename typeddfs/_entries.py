"""
Convenient code for import.
"""
import logging
from pathlib import Path
from typing import Optional, Type

import pandas as pd

from typeddfs.builders import AffinityMatrixDfBuilder, MatrixDfBuilder, TypedDfBuilder
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
from typeddfs.file_formats import FileFormat
from typeddfs.typed_dfs import TypedDf
from typeddfs.untyped_dfs import UntypedDf

logger = logging.getLogger(Path(__file__).parent.name)


class FinalDf(UntypedDf):
    """An untyped DataFrame meant for general use."""


class TypedDfs:
    """
    The only thing you need to import from ``typeddfs``.

    Contains static factory methods to build new DataFrame subclasses.
    In particular, see::

      - :py.meth:`typed`
      - :py.meth:`untyped`
      - :py.meth:`matrix`
      - :py.meth:`affinity_matrix`
    """

    NoValueError = NoValueError
    ValueNotUniqueError = ValueNotUniqueError
    InvalidDfError = InvalidDfError
    MissingColumnError = MissingColumnError
    UnexpectedColumnError = UnexpectedColumnError
    UnexpectedIndexNameError = UnexpectedIndexNameError
    ExtraConditionFailedError = VerificationFailedError
    UnsupportedOperationError = UnsupportedOperationError
    FilenameSuffixError = FilenameSuffixError
    NonStrColumnError = NonStrColumnError
    NotSingleColumnError = NotSingleColumnError
    ClashError = ClashError
    FileFormat = FileFormat
    FinalDf = FinalDf

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
            TypedDfs.wrap(df).write_file("abc.feather")
        """
        return FinalDf(df)

    @classmethod
    def typed(cls, name: str, doc: Optional[str] = None) -> TypedDfBuilder:
        """
        Creates a new type with flexible requirements.
        The class will enforce contstraints and subclass :py.class:`typeddfs.typed_dfs.TypedDf`.

        Args:
            name: The name that will be used for the new class
            doc: The docstring for the new class

        Returns:
            A builder instance (builder pattern) to be used with chained calls

        Example:
            TypedDfs.typed("MyClass").require("name", index=True).build()
        """
        return TypedDfBuilder(name, doc)

    @classmethod
    def matrix(cls, name: str, doc: Optional[str] = None) -> MatrixDfBuilder:
        """
        Creates a new subclass of an :py.class:`typeddfs.matrix_dfs.MatrixDf`.

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
        Creates a new subclass of an :py.class:`typeddfs.matrix_dfs.AffinityMatrixDf`.

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
        In general :py.meth:`typed` should be preferred because it has more consistent behavior,
        especially for IO.

        Args:
            name: The name that will be used for the new class
            doc: The docstring for the new class

        Returns:
            A class instance

        Example:
            MyClass = TypedDfs.untyped("MyClass")
        """

        class New(UntypedDf):
            pass

        New.__name__ = name
        New.__doc__ = doc
        return New


__all__ = ["FinalDf", "TypedDfs"]
