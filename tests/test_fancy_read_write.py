from typing import Set
import pytest
import random

from typeddfs import TypedDf
from typeddfs._utils import _Utils

from . import (
    tmpfile,
    Untyped,
    Trivial,
    ActuallyEmpty,
    Col1,
    Ind1,
    Col2,
    Ind2,
    Ind1Col1,
    Ind1Col2,
    Ind2Col1,
    Ind2Col2,
)

gen = random.SystemRandom()

# h5, snappy, and parquet work too -- but can't run in CI yet
assert _Utils.has_tabulate
assert _Utils.has_feather
if _Utils.has_parquet:
    print("Parquet support is loaded.")
if _Utils.has_hdf5:
    print("HDF5 support is loaded.")
known_compressions = {"", ".gz", ".zip", ".bz2", ".xz"}


def get_req_ext(txt: bool) -> Set[str]:
    ne = {".feather"}
    xx = {".csv", ".tsv", ".tab", ".json"}
    if txt:
        xx.add(".txt")
        xx.add(".lines")
        xx.add(".list")
    for e in xx:
        for c in known_compressions:
            ne.add(e + c)
    return ne


def get_actual_ext(cls) -> Set[str]:
    known = cls.can_read().intersection(cls.can_write())
    return {
        e
        for e in known
        if (
            e not in {".hdf", ".h5", ".hdf5", ".snappy", ".parquet", ".xls", ".xlsx", ".fwf"}
            and ".flexwf" not in e
        )
    }


def rand_hexes():
    # include the 'a' so it's always a string
    return ["a" + "%030x".format(random.randrange(16 ** 7)) for _ in range(6)]  # nosec


def rand_df(t):
    if issubclass(t, TypedDf):
        cols = set(t.required_index_names()).union(set(t.required_columns()))
    else:
        cols = ["column"]
    if len(cols) == 0 and t != ActuallyEmpty:
        cols = ["made_up"]
    data = {c: rand_hexes() for i, c in enumerate(cols)}
    return t.convert(t(data))


class TestReadWrite:
    def test_extensions(self):
        assert get_actual_ext(Untyped) == get_req_ext(True)
        assert get_actual_ext(Col1) == get_req_ext(True)
        assert get_actual_ext(Ind1) == get_req_ext(True)
        assert get_actual_ext(Col2) == get_req_ext(False)
        assert get_actual_ext(Ind2) == get_req_ext(False)
        assert get_actual_ext(Ind1Col1) == get_req_ext(False)
        assert get_actual_ext(Ind1Col2) == get_req_ext(False)
        assert get_actual_ext(Ind2Col1) == get_req_ext(False)
        assert get_actual_ext(Ind2Col2) == get_req_ext(False)

    def test_great(self):
        for T in [
            Untyped,
            Trivial,
            ActuallyEmpty,
            Col1,
            Ind1,
            Col2,
            Ind2,
            Ind1Col1,
            Ind1Col2,
            Ind2Col1,
            Ind2Col2,
        ]:
            for ext in get_actual_ext(T):
                if ext == ".feather" and T == ActuallyEmpty:
                    continue
                try:
                    with tmpfile(ext) as path:
                        df = rand_df(T)
                        df.write_file(path)
                        df2 = T.read_file(path)
                        assert df2.index_names() == df.index_names()
                        assert df2.column_names() == df.column_names()
                except:
                    raise AssertionError(f"Failed on {T}, {ext}")

    # noinspection DuplicatedCode
    def test_read_write_txt(self):
        for c in get_req_ext(True):
            with tmpfile(c) as path:
                df = Col1(["a", "puppy", "and", "a", "parrot"], columns=["abc"])
                df = Col1.convert(df)
                df.write_file(path)
                df2 = Col1.read_file(path)
                assert df2.index_names() == []
                assert df2.column_names() == ["abc"]

    def test_tabulate(self):
        df = Col1(["a", "puppy", "and", "a", "parrot"], columns=["abc"])
        df = Col1.convert(df)
        assert df.pretty_print() == "abc\na\npuppy\nand\na\nparrot"
        assert len(df.pretty_print("pretty").splitlines()) == len(df) + 4


if __name__ == "__main__":
    pytest.main()
