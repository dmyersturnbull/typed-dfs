import pytest

from typeddfs.untyped_dfs import UntypedDf

from . import TypedMultiIndex, TypedSingleIndex, TypedTrivial, sample_data, tmpfile


class TestReadWrite:
    def test_untyped_read_write_csv(self):
        with tmpfile() as path:
            for indices in [None, "abc", ["abc", "xyz"]]:
                df = UntypedDf(sample_data())
                if indices is not None:
                    df = df.set_index(indices)
                df.to_csv(path)
                df2 = UntypedDf.read_csv(path)
                assert list(df2.index.names) == [None]
                assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_write_passing_index(self):
        with tmpfile() as path:
            df = TypedTrivial(sample_data())
            df.to_csv(path, index=["abc"])  # fine
            df = UntypedDf(sample_data())
            df.to_csv(path, index=["abc"])  # calls super immediately

    def test_typed_read_write_csv_noindex(self):
        with tmpfile() as path:
            df = TypedTrivial(sample_data())
            df.to_csv(path)
            df2 = TypedTrivial.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_typed_read_write_csv_singleindex(self):
        with tmpfile() as path:
            df = TypedSingleIndex.convert(TypedSingleIndex(sample_data()))
            df.to_csv(path)
            assert df.index_names() == ["abc"]
            assert df.column_names() == ["123", "xyz"]
            df2 = TypedSingleIndex.read_csv(path)
            assert df2.index_names() == ["abc"]
            assert df2.column_names() == ["123", "xyz"]

    def test_typed_read_write_csv_multiindex(self):
        with tmpfile() as path:
            df = TypedMultiIndex.convert(TypedMultiIndex(sample_data()))
            df.to_csv(path)
            assert df.index_names() == ["abc", "xyz"]
            assert df.column_names() == ["123"]
            df2 = TypedMultiIndex.read_csv(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_hdf(self):
        with tmpfile() as path:
            df = TypedMultiIndex.convert(TypedMultiIndex(sample_data()))
            df.to_hdf(path)
            df2 = TypedMultiIndex.read_hdf(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]


if __name__ == "__main__":
    pytest.main()
