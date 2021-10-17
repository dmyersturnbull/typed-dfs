"""
Mixin for JSON and XML.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class _JsonXmlMixin:
    @classmethod
    def read_json(cls, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_json(*args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        return cls._convert_typed(df)

    @classmethod
    def read_xml(cls, *args, **kwargs) -> __qualname__:
        try:
            df = pd.read_xml(*args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        # see to_xml for why these fixes are needed
        if "__xml_is_empty_" in df.reset_index().columns:
            # TODO: This ok?
            # df = pd.DataFrame()
            return cls.new_df()
        elif "__xml_index_" in df.columns:
            df = df.drop(columns={"__xml_index_"})
        return cls._convert_typed(df)

    # noinspection PyFinal,PyMethodOverriding
    def to_xml(self, path_or_buf=None, *args, **kwargs) -> Optional[str]:
        # Pandas's to_xml and read_xml have two buggy properties:
        # 1. Unnamed indices are called "index"
        #    for to_xml, but not read_xml -- so they're not inverses.
        #    We'll fix that by changing the index to "__xml_index_"
        # 2. Writing an empty DataFrame results in a KeyError from deep inside
        #    We'll fix that by replacing the empty DataFrame with a DataFrame
        #   containing column "__xml_is_empty_" with a single row with the same value
        # in the insanely unlikely situation that these exist, complain
        if "__xml_is_empty_" in self.column_names() or "__xml_is_empty_" in self.index_names():
            raise ValueError("Do not include a column called '__xml_is_empty_'")
        if "__xml_index_" in self.column_names() or "__xml_index_" in self.index_names():
            raise ValueError("Do not include a column called '__xml_index_'")
        df = self.vanilla()
        if len(df) == 0 == len(self.index_names()) == len(self.column_names()) == 0:
            series = pd.Series({"__xml_is_empty_": "__xml_is_empty_"})
            df = pd.DataFrame([series])
        elif len(self.index_names()) == 0:
            df.index = df.index.rename("__xml_index_")
        return df.to_xml(path_or_buf, *args, **kwargs)

    # noinspection PyFinal,PyMethodOverriding
    def to_json(self, path_or_buf=None, *args, **kwargs) -> Optional[str]:
        df = self.vanilla_reset()
        return df.to_json(path_or_buf, *args, **kwargs)


__all__ = ["_JsonXmlMixin"]
