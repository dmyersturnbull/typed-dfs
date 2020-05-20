import pytest
import pandas as pd
from typing import Sequence
from typeddfs.typed_dfs import PrettyFrame, OrganizingFrame, SimpleFrame
from . import tmpfile, sample_data

raises = pytest.raises


class SimpleOrg(OrganizingFrame):
    pass


class SingleIndexOrg(OrganizingFrame):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc"]


class MultiIndexOrg(OrganizingFrame):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc", "xyz"]


class TestCore:
    def test_pretty(self):
        assert (
            PrettyFrame()
            ._repr_html_()
            .startswith("<strong>PrettyFrame: 0 rows × 0 columns</strong>")
        )

    def test_cfirst(self):
        df = SimpleOrg(sample_data())
        assert df.column_names() == ["abc", "123", "xyz"]
        df2 = df.cfirst(["xyz", "123", "abc"])
        assert df2.column_names() == ["xyz", "123", "abc"]
        assert df.column_names() == ["abc", "123", "xyz"]

    def test_drop_cols(self):
        df = SimpleOrg(sample_data())
        df2 = df.drop_cols(["abc", "123"])
        assert list(df.columns) == ["abc", "123", "xyz"]
        assert list(df2.columns) == ["xyz"]
        df3 = df.drop_cols("777")
        assert list(df3.columns) == ["abc", "123", "xyz"]

    def test_no_detype(self):
        df = SimpleOrg(sample_data())
        assert isinstance(df, SimpleOrg)
        assert isinstance(df.reset_index(), SimpleOrg)
        assert isinstance(df.sort_natural("abc"), SimpleOrg)
        assert isinstance(df.sort_values("abc"), SimpleOrg)
        assert isinstance(df.copy(), SimpleOrg)
        assert isinstance(df.drop(1), SimpleOrg)
        assert isinstance(df.drop("abc", axis=1), SimpleOrg)

    def test_not_inplace(self):
        df = pd.DataFrame(sample_data())
        df2 = MultiIndexOrg(df)
        df3 = MultiIndexOrg.convert(df)
        assert df.__class__.__name__ == "DataFrame"

    def test_index_names(self):
        df = MultiIndexOrg.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz"]
        df = SimpleOrg.convert(pd.DataFrame(sample_data()))
        assert isinstance(df.index_names(), list)
        assert df.index_names() == []

    def test_column_names(self):
        df = SimpleOrg(sample_data())
        # df.columns == [...] would fail because it would resolve to array==array, which is ambiguous
        assert isinstance(df.column_names(), list)
        assert df.column_names() == ["abc", "123", "xyz"]


class TestCsv:
    def test_simpleframe_read_write_csv(self):
        path = tmpfile()
        for indices in [None, "abc", ["abc", "xyz"]]:
            df = SimpleFrame(sample_data())
            if indices is not None:
                df = df.set_index(indices)
            df.to_csv(path)
            df2 = SimpleFrame.read_csv(path)
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
