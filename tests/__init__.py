import contextlib
import logging
import random
from pathlib import Path
from typing import Sequence

import pandas as pd
from typeddfs.untyped_dfs import UntypedDf

from typeddfs.typed_dfs import TypedDf


# Separate logging in the main package vs. inside test functions
logger_name = Path(__file__).parent.parent.name.upper() + ".TEST"
logger = logging.getLogger(logger_name)


@contextlib.contextmanager
def tmpfile(ext: str) -> Path:
    # caller = inspect.stack()[1][3]
    caller = str(random.randint(1, 100000))  # nosec
    path = Path(__file__).parent / "resources" / "tmp" / (str(caller) + ext)
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


class Untyped(UntypedDf):
    pass


class UntypedEmpty(UntypedDf):
    pass


class Trivial(TypedDf):
    pass


class ActuallyEmpty(TypedDf):
    pass


class Col1(TypedDf):
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ["abc"]


class Ind1(TypedDf):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc"]


class Col2(TypedDf):
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ["abc", "xyz"]


class Ind2(TypedDf):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc", "xyz"]


class Ind1Col1(TypedDf):
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ["abc"]

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["qqq"]


class Ind1Col2(TypedDf):
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ["abc", "xyz"]

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["qqq"]


class Ind2Col1(TypedDf):
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ["abc"]

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["qqq", "rrr"]


class Ind2Col2(TypedDf):
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ["abc", "xyz"]

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["qqq", "rrr"]


class TypedSymmetric(TypedDf):
    @classmethod
    def must_be_symmetric(cls) -> bool:
        return True
