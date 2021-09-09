import io
import random
from typing import Set, Type

import pandas as pd
import pytest

from typeddfs.checksums import Checksums
from typeddfs.abs_dfs import AbsDf
from typeddfs import BaseDf, TypedDf
from typeddfs.builders import TypedDfBuilder
from typeddfs.df_errors import (
    FilenameSuffixError,
    NonStrColumnError,
    NotSingleColumnError,
    FormatInsecureError,
    UnsupportedOperationError,
)
from typeddfs.file_formats import DfFormatSupport, FileFormat

from . import (
    ActuallyEmpty,
    Col1,
    Col2,
    Ind1,
    Ind1Col1,
    Ind1Col2,
    Ind2,
    Ind2Col1,
    Ind2Col2,
    Trivial,
    Untyped,
    UntypedEmpty,
    logger,
    tmpfile,
    tmpdir,
)

gen = random.SystemRandom()

assert DfFormatSupport.has_feather
assert DfFormatSupport.has_parquet
assert DfFormatSupport.has_xlsx
assert DfFormatSupport.has_xls
assert DfFormatSupport.has_xlsb
assert DfFormatSupport.has_ods
known_compressions = {"", ".gz", ".zip", ".bz2", ".xz"}


def get_req_ext(*, lines: bool, properties: bool) -> Set[str]:
    ne = {
        ".feather",
        ".snappy",
        ".parquet",
        ".xlsx",
        ".xls",
        ".ods",
        ".odf",
        ".odt",
        ".xlsb",
        ".pickle",
        ".pkl",
    }
    ne.add(".fwf")
    xx = {".csv", ".tsv", ".tab", ".json", ".xml", ".flexwf"}
    if lines:
        xx.add(".txt")
        xx.add(".lines")
        xx.add(".list")
    if properties:
        xx.add(".properties")
    for e in xx:
        for c in known_compressions:
            ne.add(e + c)
    return ne


def get_actual_ext(cls: Type[AbsDf]) -> Set[str]:
    known_fmts = cls.can_read().intersection(cls.can_write())
    exclude_for_now = {".hdf", ".h5", ".hdf5"}
    known = set()
    for k in known_fmts:
        known.update(k.suffixes)
    return {e for e in known if (e not in exclude_for_now)}


def rand_vals():
    # include the 'a' so it's always a string
    return ["a" + str(random.randint(1000, 9000)) for _ in range(6)]  # nosec


def rand_df(t):
    if issubclass(t, TypedDf):
        cols = set(t.get_typing().required_index_names).union(set(t.get_typing().required_columns))
    else:
        cols = ["column"]
    if len(cols) == 0 and t != ActuallyEmpty and t != UntypedEmpty:
        cols = ["made_up"]
    data = {c: rand_vals() for i, c in enumerate(cols)}
    return t.convert(t(data))


class TestReadWrite:
    def test_extensions(self):
        assert get_actual_ext(Untyped) == get_req_ext(lines=True, properties=False)
        assert get_actual_ext(Col1) == get_req_ext(lines=True, properties=False)
        assert get_actual_ext(Ind1) == get_req_ext(lines=True, properties=False)
        assert get_actual_ext(Col2) == get_req_ext(lines=False, properties=True)
        assert get_actual_ext(Ind2) == get_req_ext(lines=False, properties=True)
        assert get_actual_ext(Ind1Col1) == get_req_ext(lines=False, properties=True)
        assert get_actual_ext(Ind1Col2) == get_req_ext(lines=False, properties=False)
        assert get_actual_ext(Ind2Col1) == get_req_ext(lines=False, properties=False)
        assert get_actual_ext(Ind2Col2) == get_req_ext(lines=False, properties=False)

    def test_untyped(self):
        self._test_great(Untyped)

    def test_untyped_empty(self):
        self._test_great(UntypedEmpty)

    def test_trivial(self):
        self._test_great(Trivial)

    def test_actually_empty(self):
        self._test_great(ActuallyEmpty, lines_fail=True)

    def test_col1(self):
        self._test_great(Col1)

    def test_col2(self):
        self._test_great(Col2, allow_properties=True)

    def test_ind1(self):
        self._test_great(Ind1)

    def test_ind2(self):
        self._test_great(Ind2, allow_properties=True)

    def test_ind1_col1(self):
        self._test_great(Ind1Col1, allow_properties=True)

    def test_ind1_col2(self):
        self._test_great(Ind1Col2)

    def test_ind2_col1(self):
        self._test_great(Ind2Col1)

    def test_ind2_col2(self):
        self._test_great(Ind2Col2)

    def _test_great(
        self, t: Type[BaseDf], *, lines_fail: bool = False, allow_properties: bool = False
    ):
        for ext in get_actual_ext(t):
            try:
                with tmpfile(ext) as path:
                    df = rand_df(t)
                    if lines_fail and (".lines" in ext or ".txt" in ext or ".list" in ext):
                        with pytest.raises(NotSingleColumnError):
                            df.write_file(path)
                        continue
                    if not allow_properties and ".properties" in ext:
                        with pytest.raises(UnsupportedOperationError):
                            df.write_file(path)
                        continue
                    df.write_file(path)
                    if path.suffix in [
                        ".xml",
                        ".json",
                        ".csv",
                        ".tsv",
                        ".properties",
                        ".lines",
                        ".txt",
                        ".flexwf",
                        ".fwf",
                    ]:
                        raw_data = path.read_text(encoding="utf8")
                    else:
                        raw_data = None
                    df2 = t.read_file(path)

                    assert (
                        df2.index_names() == df.index_names()
                    ), f"Wrong index [ path={path}, data = {raw_data} ]"
                    assert (
                        df2.column_names() == df.column_names()
                    ), f"Wrong columns [ path={path}, data = {raw_data} ]"
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
        for c in get_req_ext(lines=True, properties=False):
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
        data = df.to_flexwf(None)
        buf = io.StringIO(data)
        df2 = df.read_flexwf(buf)
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

    def test_read_empty_xml(self):
        df = Untyped({})
        assert df.values.tolist() == []
        with tmpfile(".xml") as path:
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

    def test_no_overwrite(self):
        t = TypedDfBuilder("a").reserve("x", "y").build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="cat", y="dog"))]))
        with tmpfile(".csv") as path:
            df.write_file(path, overwrite=False)
            with pytest.raises(FileExistsError):
                df.write_file(path, overwrite=False)

    def test_mkdir(self):
        t = TypedDfBuilder("a").reserve("x", "y").build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="cat", y="dog"))]))
        with tmpdir() as path:
            df.write_file(path / "a.csv", mkdirs=True)
        with tmpdir() as path:
            with pytest.raises(FileNotFoundError):
                df.write_file(path / "b.csv")

    def test_read_write_insecure(self):
        t = TypedDfBuilder("a").secure().build()
        df = t.new_df()
        insecure = [f for f in FileFormat.list() if not f.is_secure]
        for fmt in insecure:
            for suffix in fmt.suffixes:
                try:
                    with tmpfile(suffix) as path:
                        with pytest.raises(FormatInsecureError):
                            t.read_file(path)
                        with pytest.raises(FormatInsecureError):
                            df.write_file(path)
                except Exception:
                    logger.error(f"Failed on suffix {suffix}")
                    raise

    def test_file_hash(self):
        t = TypedDfBuilder("a").reserve("x", "y").build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="cat", y="dog"))]))
        # unfortunately, the file that gets output is os-dependent
        # \n vs \r\n is an issue, so we can't check the exact hash
        with tmpfile(".csv") as path:
            df.write_file(path, file_hash=True)
            hash_file = Checksums.get_hash_file(path)
            assert hash_file.exists()
            got = Checksums.parse_hash_file_resolved(hash_file)
            assert list(got.keys()) == [path.resolve()]
            hit = got[path.resolve()]
            assert len(hit) == 64
            t.read_file(path, file_hash=True)
            t.read_file(path, hex_hash=hit)

    def test_dir_hash(self):
        t = TypedDfBuilder("a").reserve("x", "y").build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="cat", y="kitten"))]))
        with tmpfile(".csv") as path:
            hash_dir = Checksums.get_hash_dir(path)
            hash_dir.unlink(missing_ok=True)
            df.write_file(path, dir_hash=True)
            assert hash_dir.exists()
            got = Checksums.parse_hash_file_resolved(hash_dir)
            assert list(got.keys()) == [path.resolve()]
            hit = got[path.resolve()]
            assert len(hit) == 64
            t.read_file(path, dir_hash=True)
            t.read_file(path, hex_hash=hit)


if __name__ == "__main__":
    pytest.main()
