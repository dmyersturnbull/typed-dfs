import io
from typing import Set, Type

import pandas as pd
import pytest
import random

from typeddfs import TypedDf, BaseDf, TypedDfBuilder
from typeddfs.file_formats import DfFormatSupport, FileFormat
from typeddfs.df_errors import NonStrColumnError, NotSingleColumnError, FilenameSuffixError

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
    logger,
)

gen = random.SystemRandom()

assert DfFormatSupport.has_tabulate
assert DfFormatSupport.has_feather
assert DfFormatSupport.has_parquet
if DfFormatSupport.has_hdf5:
    logger.info("HDF5 support is loaded. Cannot test... yet.")
known_compressions = {"", ".gz", ".zip", ".bz2", ".xz"}


def get_req_ext(txt: bool) -> Set[str]:
    ne = {".feather", ".snappy", ".parquet"}
    xx = {".csv", ".tsv", ".tab", ".json", ".flexwf"}
    if txt:
        xx.add(".txt")
        xx.add(".lines")
        xx.add(".list")
    for e in xx:
        for c in known_compressions:
            ne.add(e + c)
    return ne


def get_actual_ext(cls) -> Set[str]:
    known_fmts = cls.can_read().intersection(cls.can_write())
    known = set()
    for k in known_fmts:
        known.update(k.suffixes)
    return {e for e in known if (e not in {".hdf", ".h5", ".hdf5", ".xls", ".xlsx", ".fwf"})}


def rand_vals():
    # include the 'a' so it's always a string
    return ["a" + str(random.randint(1000, 9000)) for _ in range(6)]  # nosec


def rand_df(t):
    if issubclass(t, TypedDf):
        cols = set(t.required_index_names()).union(set(t.required_columns()))
    else:
        cols = ["column"]
    if len(cols) == 0 and t != ActuallyEmpty:
        cols = ["made_up"]
    data = {c: rand_vals() for i, c in enumerate(cols)}
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

    def test_untyped(self):
        self._test_great(Untyped)

    def test_trivial(self):
        self._test_great(Trivial)

    def test_actually_empty(self):
        self._test_great(ActuallyEmpty)

    def test_col1(self):
        self._test_great(Col1)

    def test_ind1(self):
        self._test_great(Ind1)

    def test_ind1_col1(self):
        self._test_great(Ind1Col1)

    def test_ind1_col2(self):
        self._test_great(Ind1Col2)

    def test_ind2_col1(self):
        self._test_great(Ind2Col1)

    def test_ind2_col2(self):
        self._test_great(Ind2Col2)

    def _test_great(self, t: Type[BaseDf]):
        for ext in get_actual_ext(t):
            if ext == ".feather" and t == ActuallyEmpty:
                continue
            try:
                with tmpfile(ext) as path:
                    df = rand_df(t)
                    df.write_file(path)
                    df2 = t.read_file(path)
                    assert df2.index_names() == df.index_names()
                    assert df2.column_names() == df.column_names()
            except Exception:
                logger.error(f"Failed on {t} / {ext}")
                raise

    def test_bad_suffix(self):
        df = Untyped({"abc": [1, 2], "xyz": [1, 2]})
        with tmpfile(".omg") as path:
            with pytest.raises(FilenameSuffixError):
                df.write_file(path)

    def test_non_str_cols(self):
        with tmpfile(".csv") as path:
            df = Untyped(["1", "2"])
            with pytest.raises(NonStrColumnError):
                df.write_file(path)

    def test_non_1_col_lines(self):
        with tmpfile(".lines") as path:
            df = Untyped({"abc": [1, 2], "xyz": [1, 2]})
            with pytest.raises(NotSingleColumnError):
                df.to_lines(path)
            df = Untyped({})
            with pytest.raises(NotSingleColumnError):
                df.to_lines(path)
            df = rand_df(Col2)
            with pytest.raises(NotSingleColumnError):
                df.to_lines(path)
            df = rand_df(Ind2)
            with pytest.raises(NotSingleColumnError):
                df.to_lines(path)
            df = rand_df(Ind1Col2)
            with pytest.raises(NotSingleColumnError):
                df.to_lines(path)

    # noinspection DuplicatedCode
    def test_read_write_txt(self):
        for c in get_req_ext(True):
            df = Col1(["a", "puppy", "and", "a", "parrot"], columns=["abc"])
            with tmpfile(c) as path:
                df.write_file(path)
                df2 = Col1.read_file(path)
                assert df2.index_names() == []
                assert df2.column_names() == ["abc"]

    def test_read_write_txt_fail(self):
        df = rand_df(Col2)
        with tmpfile(".lines") as path:
            with pytest.raises(NotSingleColumnError):
                df.to_lines(path)

    def test_read_write_flexwf_float(self):
        df = Col1([0.3, 0.4, 0.5], columns=["abc"])
        df = Col1.convert(df)
        data = df.to_flexwf(None)
        buf = io.StringIO(data)
        df2 = df.read_flexwf(buf)
        assert df.column_names() == df2.column_names()
        assert df.index_names() == df2.index_names()
        assert df.values.tolist() == df2.values.tolist()

    def test_read_write_flexwf_fancy_delimiter(self):
        df = Col1([0.3, 0.4, 0.5], columns=["abc"])
        df = Col1.convert(df)
        data = df.to_flexwf(None, sep="--->")
        buf = io.StringIO(data)
        df2 = df.read_flexwf(buf, sep="--->")
        assert df.column_names() == df2.column_names()
        assert df.index_names() == df2.index_names()
        assert df.values.tolist() == df2.values.tolist()

    def test_tabulate(self):
        df = Col1(["a", "puppy", "and", "a", "parrot"], columns=["abc"])
        df = Col1.convert(df)
        assert df.pretty_print() == "abc\na\npuppy\nand\na\nparrot"
        assert len(df.pretty_print("pretty").splitlines()) == len(df) + 4

    def test_lines_apply(self):
        assert Untyped._lines_files_apply()
        assert Col1._lines_files_apply()
        assert Ind1._lines_files_apply()
        assert not Col2._lines_files_apply()
        assert not Ind1Col1._lines_files_apply()

    def test_read_empty_csv(self):
        df = Untyped({})
        assert df.values.tolist() == []
        with tmpfile(".csv") as path:
            df.to_csv(path)
            df2 = Untyped.read_csv(path)
        assert df.values.tolist() == df2.values.tolist()

    def test_read_empty_txt(self):
        df = Untyped({})
        assert df.values.tolist() == []
        with tmpfile(".lines") as path:
            df.to_csv(path)
            df2 = Untyped.read_lines(path)
        assert df.values.tolist() == df2.values.tolist()

    def test_pass_io_options(self):
        t = TypedDfBuilder("a").reserve("x", "y").add_write_kwargs(FileFormat.csv, sep="&").build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="cat", y="dog"))]))
        with tmpfile(".csv") as path:
            df.write_file(path)
            lines = path.read_text(encoding="utf8").splitlines()
            assert lines == ["x&y", "cat&dog"]


if __name__ == "__main__":
    pytest.main()
