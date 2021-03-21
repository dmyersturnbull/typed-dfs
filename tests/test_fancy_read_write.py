import pytest

from . import TypedMultiIndex, sample_data, tmpfile, TypedOneColumn, TypedSingleIndex

# h5, snappy, and parquet work too -- but can't run in CI yet
known_compressions = {"", ".gz", ".zip", ".bz2", ".xz"}


def _get_known_extensions():
    ne = {".feather"}
    for e in {".csv", ".tsv", ".tab"}:
        for c in known_compressions:
            ne.add(e + c)
    return ne


known_extensions = _get_known_extensions()


class TestReadWrite:
    def test_read_write_file_multi_index(self):
        for ext in known_extensions:
            with tmpfile(ext) as path:
                df = TypedMultiIndex.convert(TypedMultiIndex(sample_data()))
                df.write_file(path)
                df2 = TypedMultiIndex.read_file(path)
                assert df2.index_names() == ["abc", "xyz"]
                assert df2.column_names() == ["123"]

    def test_read_write_one_single_index(self):
        for ext in known_extensions:
            with tmpfile(ext) as path:
                df = TypedSingleIndex.convert(TypedSingleIndex(sample_data()))
                df.write_file(path)
                df2 = TypedSingleIndex.read_file(path)
                assert df2.index_names() == ["abc"]
                assert df2.column_names() == ["123", "xyz"]

    # noinspection DuplicatedCode
    def test_read_write_one_col(self):
        for ext in known_extensions:
            with tmpfile(ext) as path:
                df = TypedOneColumn(["a", "puppy", "and", "a", "parrot"], columns=["abc"])
                df = TypedOneColumn.convert(df)
                df.write_file(path)
                df2 = TypedOneColumn.read_file(path)
                assert df2.index_names() == []
                assert df2.column_names() == ["abc"]

    # noinspection DuplicatedCode
    def test_read_write_txt(self):
        for c in known_compressions:
            with tmpfile(".txt" + c) as path:
                df = TypedOneColumn(["a", "puppy", "and", "a", "parrot"], columns=["abc"])
                df = TypedOneColumn.convert(df)
                df.write_file(path)
                df2 = TypedOneColumn.read_file(path)
                assert df2.index_names() == []
                assert df2.column_names() == ["abc"]


if __name__ == "__main__":
    pytest.main()
