from dataclasses import dataclass
from typing import Optional, Any

import pandas as pd
import pytest

from typeddfs.builders import TypedDfBuilder
from typeddfs.typed_dfs import TypedDfDataclass


class TestDataclasses:
    def test_create_empty(self):
        t = TypedDfBuilder("T").build()
        dc = t.create_dataclass()
        assert dc is not None
        assert issubclass(dc, TypedDfDataclass)
        assert dc.get_fields() == []

    def test_create_simple(self):
        t = TypedDfBuilder("T").require("greeting", dtype=str).reserve("bonus", dtype=int).build()
        dc = t.create_dataclass()
        assert dc is not None
        assert issubclass(dc, TypedDfDataclass)
        assert len(dc.get_fields()) == 2
        assert dc.get_fields()[0].name == "greeting"
        assert dc.get_fields()[0].type == str
        assert dc.get_fields()[1].name == "bonus"
        assert dc.get_fields()[1].type == Optional[int]

    def test_create_any(self):
        t = TypedDfBuilder("T").require("greeting").reserve("bonus").build()
        dc = t.create_dataclass()
        assert dc is not None
        assert issubclass(dc, TypedDfDataclass)
        assert len(dc.get_fields()) == 2
        assert dc.get_fields()[0].name == "greeting"
        assert dc.get_fields()[0].type == Any
        assert dc.get_fields()[1].name == "bonus"
        assert dc.get_fields()[1].type == Optional[Any]

    def test_to_instances(self):
        t = TypedDfBuilder("T").require("animal", dtype=str).reserve("age", dtype=int).build()
        df: t = t.of(
            [
                pd.Series(dict(animal="goldfish", age=2)),
                pd.Series(dict(animal="goldfish", age=1)),
                pd.Series(dict(animal="gazelle", age=8)),
                pd.Series(dict(animal="pineapple", age=114)),
                pd.Series(dict(animal="anteater", age=11)),
            ]
        )
        dc = t.create_dataclass()
        instances = df.to_dataclass_instances()
        assert instances == [
            dc("goldfish", 2),
            dc("goldfish", 1),
            dc("gazelle", 8),
            dc("pineapple", 114),
            dc("anteater", 11),
        ]
        assert list(sorted(instances)) == [
            dc("anteater", 11),
            dc("gazelle", 8),
            dc("goldfish", 1),
            dc("goldfish", 2),
            dc("pineapple", 114),
        ]

    def test_to_instances_empty(self):
        t = TypedDfBuilder("T").reserve("animal", dtype=str).build()
        df: t = t.of([])
        instances = df.to_dataclass_instances()
        assert instances == []

    def test_read_instances(self):
        @dataclass(
            frozen=True,
        )
        class Dc:
            animal: str
            val: Optional[int]

        t = TypedDfBuilder("T").require("animal", dtype=str).reserve("age", dtype=int).build()
        df = t.from_dataclass_instances([Dc("cat", 1), Dc("kitten", 2)])
        assert len(df) == 2
        assert df.values.tolist() == [["cat", 1], ["kitten", 2]]

    def test_read_instances_empty(self):
        t = TypedDfBuilder("T").require("animal", dtype=str).build()
        df = t.from_dataclass_instances([])
        assert len(df) == 0

    def test_read_instances_empty_fields(self):
        @dataclass(frozen=True)
        class Dc:
            pass

        t = TypedDfBuilder("T").reserve("animal", dtype=str).build()
        df = t.from_dataclass_instances([Dc()])
        assert len(df) == 1
        assert "animal" not in df.columns


if __name__ == "__main__":
    pytest.main()
