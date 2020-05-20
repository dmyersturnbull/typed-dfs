import pytest
from typeddfs.untyped_dfs import UntypedDf
from . import tmpfile, sample_data, SimpleOrg, SingleIndexOrg, MultiIndexOrg

raises = pytest.raises


class TestCsv:
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
        assert list(df.index.names) == ["abc"]
        assert set(df.columns) == {"123", "xyz"}
        df2 = SingleIndexOrg.read_csv(path)
        assert list(df2.index.names) == ["abc"]
        assert set(df2.columns) == {"123", "xyz"}
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_multiindex(self):
        path = tmpfile()
        df = MultiIndexOrg.convert(MultiIndexOrg(sample_data()))
        df.to_csv(path)
        assert list(df.index.names) == ["abc", "xyz"]
        assert set(df.columns) == {"123"}
        df2 = MultiIndexOrg.read_csv(path)
        assert list(df2.index.names) == ["abc", "xyz"]
        assert set(df2.columns) == {"123"}
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    pytest.main()
