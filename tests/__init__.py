import contextlib
import inspect
from pathlib import Path
from typing import Sequence

import pandas as pd

from typeddfs.typed_dfs import TypedDf


@contextlib.contextmanager
def tmpfile() -> Path:
    caller = inspect.stack()[1][3]
    path = Path(__file__).parent.parent.parent / "resources" / "tmp" / (str(caller) + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        path.unlink()


def sample_data():
    return [
        pd.Series({"abc": 1, "123": 2, "xyz": 3}),
        pd.Series({"abc": 4, "123": 5, "xyz": 6}),
    ]


def sample_symmetric_df():
    return pd.DataFrame(
        [
            pd.Series({"a": "x", "b": "y", "my_index": "a"}),
            pd.Series({"a": "x", "b": "y", "my_index": "b"}),
        ]
    ).set_index("my_index")


def sample_data_str():
    return [
        pd.Series({"abc": "zzz", "123": 2, "xyz": 3}),
        pd.Series({"abc": "bbb", "123": 5, "xyz": 6}),
    ]


def sample_data_2():
    return [
        pd.Series({"only": 1, "multi": 2, "none": None}),
        pd.Series({"only": 1, "multi": 5, "none": None}),
    ]


class TypedTrivial(TypedDf):
    pass


class TypedSingleIndex(TypedDf):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc"]


class TypedMultiIndex(TypedDf):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc", "xyz"]


class TypedSymmetric(TypedDf):
    @classmethod
    def must_be_symmetric(cls) -> bool:
        return True
