from pathlib import Path
import inspect

import pandas as pd

from typing import Sequence
from typeddfs.typed_dfs import TypedDf


def tmpfile() -> Path:
    caller = inspect.stack()[1][3]
    path = Path(__file__).parent.parent.parent / "resources" / "tmp" / (str(caller) + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sample_data():
    return [
        pd.Series({"abc": 1, "123": 2, "xyz": 3}),
        pd.Series({"abc": 4, "123": 5, "xyz": 6}),
    ]


def sample_data_str():
    return [
        pd.Series({"abc": "zzz", "123": 2, "xyz": 3}),
        pd.Series({"abc": "bbb", "123": 5, "xyz": 6}),
    ]


class SimpleOrg(TypedDf):
    pass


class SingleIndexOrg(TypedDf):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc"]


class MultiIndexOrg(TypedDf):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc", "xyz"]
