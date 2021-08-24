import contextlib
import io
from typing import Optional

import pandas as pd
import pytest

from typeddfs.builders import TypedDfBuilder
from typeddfs.abs_dfs import AbsDf, TypedDfDataclass
from typeddfs.utils import FrozeList


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
        # assert dc.get_df_type() is t  # TODO
        # noinspection PyArgumentList
        x = dc(hi="cat", val=1)
        assert isinstance(x, TypedDfDataclass)
        # noinspection PyUnresolvedReferences
        assert x.hi == "cat"
        # noinspection PyUnresolvedReferences
        assert x.val == 1
        assert hasattr(dc, "__lt__") and dc.__lt__ is not None

    def test_to_instances(self):
        t = TypedDfBuilder("T").require("hi", dtype=str).reserve("val", dtype=int).build()
        df: t = t.of([pd.Series(dict(hi="dog", val=111))])
        dc = t.create_dataclass()
        inst = df.to_dataclass_instances()
        assert len(inst) == 1
        assert inst[0].hi == "dog"
        assert inst[0].val == 111
        assert inst[0] == dc(hi="dog", val=111)

    def test_to_instances_frozen(self):
        t = TypedDfBuilder("T").require("hi", dtype=str).require("vals").build()
        df: t = t.of([pd.Series(dict(hi="dog", vals=[1, 2]))])
        dc = t.create_dataclass()
        inst = df.to_dataclass_instances()
        assert len(inst) == 1
        assert inst[0].hi == "dog"
        assert inst[0].vals == FrozeList([1, 2])
        assert inst[0].vals == [1, 2]
        assert inst[0] == dc(hi="dog", vals=FrozeList([1, 2]))
        assert inst[0] == dc(hi="dog", vals=[1, 2])  # technically works


if __name__ == "__main__":
    pytest.main()
