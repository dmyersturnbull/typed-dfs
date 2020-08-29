import pytest

from typeddfs.untyped_dfs import UntypedDf

from . import MultiIndexOrg, SimpleOrg, SingleIndexOrg, sample_data, tmpfile

raises = pytest.raises


class TestReadWrite:
    def test_simpleframe_read_write_csv(self):
        path = tmpfile()
        for indices in [None, "abc", ["abc", "xyz"]]:
            df = UntypedDf(sample_data())
            if indices is not None:
                df = df.set_index(indices)
            df.to_csv(path)
            df2 = UntypedDf.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_noindex(self):
        path = tmpfile()
        df = SimpleOrg(sample_data())
        df.to_csv(path)
        df2 = SimpleOrg.read_csv(path)
        assert list(df2.index.names) == [None]
        assert set(df2.columns) == {"abc", "123", "xyz"}
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_singleindex(self):
        path = tmpfile()
        df = SingleIndexOrg.convert(SingleIndexOrg(sample_data()))
        df.to_csv(path)
        assert df.index_names() == ["abc"]
        assert df.column_names() == ["123", "xyz"]
        df2 = SingleIndexOrg.read_csv(path)
        assert df2.index_names() == ["abc"]
        assert df2.column_names() == ["123", "xyz"]
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_multiindex(self):
        path = tmpfile()
        df = MultiIndexOrg.convert(MultiIndexOrg(sample_data()))
        df.to_csv(path)
        assert df.index_names() == ["abc", "xyz"]
        assert df.column_names() == ["123"]
        df2 = MultiIndexOrg.read_csv(path)
        assert df2.index_names() == ["abc", "xyz"]
        assert df2.column_names() == ["123"]
        if path.exists():
            path.unlink()

    def test_hdf(self):
        path = tmpfile()
        df = MultiIndexOrg.convert(MultiIndexOrg(sample_data()))
        df.to_hdf(path)
        df2 = MultiIndexOrg.read_hdf(path)
        assert df2.index_names() == ["abc", "xyz"]
        assert df2.column_names() == ["123"]


if __name__ == "__main__":
    pytest.main()
