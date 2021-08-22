import contextlib
import logging
import random
import shutil
from pathlib import Path
from typing import Union

import pandas as pd

from typeddfs.df_typing import DfTyping
from typeddfs.typed_dfs import TypedDf
from typeddfs.untyped_dfs import UntypedDf

# Separate logging in the main package vs. inside test functions
logger_name = Path(__file__).parent.parent.name.upper() + ".TEST"
logger = logging.getLogger(logger_name)


def get_resource(*nodes: Union[str, Path]) -> Path:
    path = Path(Path(__file__).parent, "resources", *nodes)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


@contextlib.contextmanager
def tmpdir() -> Path:
    bit1 = str(random.randint(1, 100000))  # nosec
    bit2 = str(random.randint(1, 100000))  # nosec
    path = Path(__file__).parent / "resources" / "tmp" / bit1 / bit2
    yield path
    if path.exists():
        shutil.rmtree(str(path))


@contextlib.contextmanager
def tmpfile(ext: str) -> Path:
    # caller = inspect.stack()[1][3]
    caller = str(random.randint(1, 100000))  # nosec
    path = Path(__file__).parent / "resources" / "tmp" / (str(caller) + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        # noinspection PyBroadException
        try:
            path.unlink()
        except BaseException:
            logger.error(f"Could not close {path}")


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
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["abc"])


class Ind1(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_index_names=["abc"])


class Col2(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["abc", "xyz"])


class Ind2(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_index_names=["abc", "xyz"])


class Ind1Col1(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["abc"], _required_index_names=["qqq"])


class Ind1Col2(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["abc", "xyz"], _required_index_names=["qqq"])


class Ind2Col1(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["abc"], _required_index_names=["qqq", "rrr"])


class Ind2Col2(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(_required_columns=["abc", "xyz"], _required_index_names=["qqq", "rrr"])


def sample_data_ind2_col2():
    return [
        pd.Series({"abc": 1, "xyz": 3, "qqq": "hi", "rrr": "hello"}),
        pd.Series({"abc": 4, "xyz": 6, "qqq": "hi", "rrr": "hello"}),
    ]


class Ind2Col2Reserved1(TypedDf):
    @classmethod
    def get_typing(cls) -> DfTyping:
        return DfTyping(
            _required_columns=["abc", "xyz"],
            _reserved_columns=["res"],
            _required_index_names=["qqq", "rrr"],
        )
