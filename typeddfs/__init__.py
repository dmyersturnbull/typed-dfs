"""
Metadata for TypedDfs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, Sequence, Callable

# importlib.metadata is compat with Python 3.8 only
from importlib_metadata import metadata as __load

import pandas as pd

from typeddfs.base_dfs import (
    BaseDf,
    InvalidDfError as _InvalidDfError,
    MissingColumnError as _MissingColumnError,
    UnexpectedColumnError as _UnexpectedColumnError,
    AsymmetricDfError as _AsymmetricDfError,
    ExtraConditionError as _ExtraConditionError,
)
from typeddfs.typed_dfs import TypedDf
from typeddfs.untyped_dfs import UntypedDf

metadata = __load(Path(__file__).parent.name)
__status__ = "Development"
__copyright__ = "Copyright 2015–2020"
__date__ = "2020-05-19"
__uri__ = metadata["home-page"]
__title__ = metadata["name"]
__summary__ = metadata["summary"]
__license__ = metadata["license"]
__version__ = metadata["version"]
__author__ = metadata["author"]
__maintainer__ = metadata["maintainer"]
__contact__ = metadata["maintainer"]


class TypedDfBuilder:
    def __init__(self, name: str, doc: Optional[str] = None):
        self._name = name
        self._doc = doc
        self._req_meta = []
        self._res_meta = []
        self._req_cols = []
        self._res_cols = []
        self._drop = []
        self._strict_meta = False
        self._strict_cols = False
        self._symmetric = False
        self._extra_reqs = []

    def require(self, *names: str, index: bool = False) -> TypedDfBuilder:
        if index:
            self._req_meta.extend(names)
        else:
            self._req_cols.extend(names)
        return self

    def reserve(self, *names: str, index: bool = False) -> TypedDfBuilder:
        if index:
            self._res_meta.extend(names)
        else:
            self._res_cols.extend(names)
        return self

    def drop(self, *names: str):
        self._drop.extend(names)

    def strict(self, index: bool = True, cols: bool = True) -> TypedDfBuilder:
        self._strict_meta = index
        self._strict_cols = cols
        return self

    def symmetric(self) -> TypedDfBuilder:
        self._symmetric = True
        return self

    def condition(self, *conditions: Callable[[pd.DataFrame], Optional[str]]) -> TypedDfBuilder:
        self._extra_reqs.extend(conditions)
        return self

    def build(self) -> Type[TypedDf]:
        class New(TypedDf):
            @classmethod
            def more_indices_allowed(cls) -> bool:
                return not self._strict_meta

            @classmethod
            def more_columns_allowed(cls) -> bool:
                return not self._strict_cols

            @classmethod
            def required_columns(cls) -> Sequence[str]:
                return self._req_cols

            @classmethod
            def reserved_columns(cls) -> Sequence[str]:
                return self._res_cols

            @classmethod
            def reserved_index_names(cls) -> Sequence[str]:
                return self._res_meta

            @classmethod
            def required_index_names(cls) -> Sequence[str]:
                return self._req_meta

            @classmethod
            def must_be_symmetric(cls) -> bool:
                return self._symmetric

            @classmethod
            def columns_to_drop(cls) -> Sequence[str]:
                return self._drop

            @classmethod
            def extra_conditions(cls) -> Sequence[Callable[[pd.DataFrame], Optional[str]]]:
                return self._extra_reqs

        New.__name__ = self._name
        New.__doc__ = self._doc
        return New


class TypedDfs:

    InvalidDfError = (_InvalidDfError,)
    MissingColumnError = (_MissingColumnError,)
    UnexpectedColumnError = (_UnexpectedColumnError,)
    AsymmetricDfError = (_AsymmetricDfError,)
    ExtraConditionError = _ExtraConditionError

    @classmethod
    def example(cls):
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
        return TypedDfBuilder(name, doc)

    @classmethod
    def untyped(cls, name: str, doc: Optional[str] = None) -> Type[UntypedDf]:
        class New(UntypedDf):
            pass

        New.__name__ = name
        New.__doc__ = doc
        return New


__all__ = ["BaseDf", "UntypedDf", "TypedDf", "TypedDfs"]
