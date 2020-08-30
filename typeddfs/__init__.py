"""
Metadata for typeddfs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Type

# importlib.metadata is compat with Python 3.8 only
from importlib_metadata import PackageNotFoundError
from importlib_metadata import metadata as __load

from typeddfs.base_dfs import AsymmetricDfError as _AsymmetricDfError
from typeddfs.base_dfs import BaseDf
from typeddfs.base_dfs import ExtraConditionFailedError as _ExtraConditionFailedError
from typeddfs.base_dfs import InvalidDfError as _InvalidDfError
from typeddfs.base_dfs import MissingColumnError as _MissingColumnError
from typeddfs.base_dfs import NoValueError as _NoValueError
from typeddfs.base_dfs import UnexpectedColumnError as _UnexpectedColumnError
from typeddfs.base_dfs import UnexpectedIndexNameError as _UnexpectedIndexNameError
from typeddfs.base_dfs import ValueNotUniqueError as _ValueNotUniqueError
from typeddfs.builders import TypedDfBuilder
from typeddfs.typed_dfs import TypedDf
from typeddfs.untyped_dfs import UntypedDf

logger = logging.getLogger(Path(__file__).parent.name)

metadata = None
try:
    metadata = __load(Path(__file__).parent.name)
    __status__ = "Development"
    __copyright__ = "Copyright 2016–2020"
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
    logger.error(
        f"Could not load package metadata for {Path(__file__).absolute().parent.name}. Is it installed?"
    )


class TypedDfs:
    """
    The only thing you need to import from ``typeddfs``.

    Contains the ``typed()`` and ``untyped()`` static functions, which build new classes.
    Also contains specific exception types used by typeddfs, such as ``InvalidDfError`` and ``MissingColumnError``.
    """

    pkg_metadata = metadata
    NoValueError = _NoValueError
    ValueNotUniqueError = _ValueNotUniqueError
    InvalidDfError = _InvalidDfError
    MissingColumnError = _MissingColumnError
    UnexpectedColumnError = _UnexpectedColumnError
    UnexpectedIndexNameError = _UnexpectedIndexNameError
    AsymmetricDfError = _AsymmetricDfError
    ExtraConditionFailedError = _ExtraConditionFailedError

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
        KeyValue = (
            TypedDfs.typed("KeyValue")  # typed means enforced requirements
            .require("key", index=True)  # automagically make this an index
            .require("value")  # required
            .reserve("note")  # permitted but not required
            .strict()  # don't allow other columns
        ).build()
        return KeyValue

    @classmethod
    def typed(cls, name: str, doc: Optional[str] = None) -> TypedDfBuilder:
        """
        Creates a new builder (builder pattern) for a ``TypedDf``.
        The final class will enforce constraints.

        Args:
            name: The name that will be used for the new class
            doc: The docstring for the new class

        Returns:
            A builder pattern, to be used with chained calls

        Example:
            TypedDfs.typed("MyClass").require("name", index=True).build()
        """
        return TypedDfBuilder(name, doc)

    @classmethod
    def untyped(cls, name: str, doc: Optional[str] = None) -> Type[UntypedDf]:
        """
        Creates a new subclass of ``UntypedDf``.
        The returned class will NOT enforce any constraints,
        but it will have some convenient methods.

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


__all__ = ["BaseDf", "UntypedDf", "TypedDf", "TypedDfs"]
