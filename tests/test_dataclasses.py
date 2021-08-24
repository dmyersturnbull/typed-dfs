import contextlib
import io
from typing import Optional

import pytest

from typeddfs.builders import TypedDfBuilder
from typeddfs.abs_dfs import AbsDf, TypedDfDataclass


class _UnhashableUnorderable:
    def __init__(self, v):
        self.v = v


class TestDataclasses:
    def test_create_empty(self):
        t = TypedDfBuilder("T").build()
        dc = t.create_dataclass()
        assert dc is not None
        assert issubclass(dc, TypedDfDataclass)
        assert dc.get_fields() == []

    def test_create_simple(self):
        t = TypedDfBuilder("T").require("hi", dtype=str).reserve("val", dtype=int).build()
        dc = t.create_dataclass()
        assert dc is not None
        assert issubclass(dc, TypedDfDataclass)
        assert len(dc.get_fields()) == 2
        assert dc.get_fields()[0].name == "hi"
        assert dc.get_fields()[0].type == str
        assert dc.get_fields()[1].name == "val"
        assert dc.get_fields()[1].type == Optional[int]

    def test_create_ordered(self):
        t = TypedDfBuilder("T").require("hi", dtype=str).reserve("val", dtype=int).build()
        dc = t.create_dataclass()
        assert issubclass(dc, TypedDfDataclass)
        assert hasattr(dc, "__lt__") and dc.__lt__ is not None

    def test_create_unordered(self):
        t = TypedDfBuilder("T").require("hi", dtype=list).reserve("val", dtype=int).build()
        dc = t.create_dataclass()
        assert issubclass(dc, TypedDfDataclass)
        assert hasattr(dc, "__lt__") and dc.__lt__ is not None


if __name__ == "__main__":
    pytest.main()
