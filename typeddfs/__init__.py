"""
Metadata for TypedDfs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, Sequence

# importlib.metadata is compat with Python 3.8 only
from importlib_metadata import metadata as __load

import pandas as pd

from typeddfs.typed_dfs import (
    SimpleFrame,
    FinalFrame,
    TypedFrame,
    OrganizingFrame,
    MissingColumnError,
    UnexpectedColumnError,
)

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

    def build(self) -> Type[OrganizingFrame]:
        class New(OrganizingFrame):
            @classmethod
            def more_index_names_allowed(cls) -> bool:
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
            def columns_to_drop(cls) -> Sequence[str]:
                return self._drop

        New.__name__ = self._name
        New.__doc__ = self._doc
        return New


class TypedDfs:
    @classmethod
    def fancy(cls, name: str, doc: Optional[str] = None) -> TypedDfBuilder:
        return TypedDfBuilder(name, doc)

    @classmethod
    def simple(cls, name: str, doc: Optional[str] = None) -> Type[SimpleFrame]:
        class New(SimpleFrame):
            pass

        New.__name__ = name
        New.__doc__ = doc
        return New

    @classmethod
    def wrap(cls, df: pd.DataFrame, class_name: Optional[str] = None) -> TypedFrame:
        """
        Wrap `df` in a typed DataFrame (ConvertibleFrame).
        The returned Pandas DataFrame will have additional methods and better display in Jupyter.
        - If `df` is already a `ConvertibleFrame`, will just return it.
        - Otherwise:
            * Creates a new class with name `class_name` if `class_name` is non-null.
            * Otherwise wraps in a `FinalFrame`.
        :param df: Any Pandas DataFrame.
        :param class_name: Only applies if `df` isn't already a `ConvertableFrame`
        :return: A copy of `df` of the new class
        """
        if isinstance(df, TypedFrame):
            return df
        elif isinstance(df, pd.DataFrame):
            if class_name is None:
                return FinalFrame(df)
            else:

                class X(SimpleFrame):
                    pass

                X.__name__ = class_name
                return X(df)
        else:
            raise TypeError("Invalid DataFrame type {}".format(df))


__all__ = [
    "TypedFrame",
    "SimpleFrame",
    "OrganizingFrame",
    "TypedDfs",
    "MissingColumnError",
    "UnexpectedColumnError",
]
