import pytest

from typeddfs.untyped_dfs import UntypedDf

from . import Ind2, Ind1, Trivial, sample_data, tmpfile


class TestReadWrite:
    def test_feather_lz4(self):
        with tmpfile(".feather") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_feather(path, compression="lz4")
            df2 = Ind2.read_feather(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_feather_zstd(self):
        with tmpfile(".feather") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_feather(path, compression="zstd")
            df2 = Ind2.read_feather(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_csv_gz(self):
        with tmpfile(".csv.gz") as path:
            df = UntypedDf(sample_data())
            df.to_csv(path)
            df2 = UntypedDf.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_untyped_read_write_csv(self):
        with tmpfile(".csv") as path:
            for indices in [None, "abc", ["abc", "xyz"]]:
                df = UntypedDf(sample_data())
                if indices is not None:
                    df = df.set_index(indices)
                df.to_csv(path)
                df2 = UntypedDf.read_csv(path)
                assert list(df2.index.names) == [None]
                assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_write_passing_index(self):
        with tmpfile(".csv") as path:
            df = Trivial(sample_data())
            df.to_csv(path, index=["abc"])  # fine
            df = UntypedDf(sample_data())
            df.to_csv(path, index=["abc"])  # calls super immediately

    def test_typed_read_write_csv_noindex(self):
        with tmpfile(".csv") as path:
            df = Trivial(sample_data())
            df.to_csv(path)
            df2 = Trivial.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_typed_read_write_csv_singleindex(self):
        with tmpfile(".csv") as path:
            df = Ind1.convert(Ind1(sample_data()))
            df.to_csv(path)
            assert df.index_names() == ["abc"]
            assert df.column_names() == ["123", "xyz"]
            df2 = Ind1.read_csv(path)
            assert df2.index_names() == ["abc"]
            assert df2.column_names() == ["123", "xyz"]

    def test_typed_read_write_csv_multiindex(self):
        with tmpfile(".csv") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_csv(path)
            assert df.index_names() == ["abc", "xyz"]
            assert df.column_names() == ["123"]
            df2 = Ind2.read_csv(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    """
    # TODO: re-enable when llvmlite wheels are available for Python 3.9
    def test_parquet(self):
        with tmpfile(".parquet") as path:
            df = UntypedDf(sample_data())
            df.to_parquet(path)
            df2 = UntypedDf.read_parquet(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}
    """

    """
    # TODO re-enable when we get a pytables 3.9 wheels on Windows
    def test_hdf(self):
        with tmpfile() as path:
            df = TypedMultiIndex.convert(TypedMultiIndex(sample_data()))
            df.to_hdf(path)
            df2 = TypedMultiIndex.read_hdf(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]
    """


if __name__ == "__main__":
    pytest.main()
