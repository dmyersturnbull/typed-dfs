import pytest
import pandas as pd

# noinspection PyProtectedMember
from typeddfs.base_dfs import AbsDf
from typeddfs.untyped_dfs import UntypedDf
from . import sample_data, SimpleOrg, MultiIndexOrg

raises = pytest.raises


class TestBase:
    def test_pretty(self):
        assert (
            UntypedDf()._repr_html_().startswith("<strong>UntypedDf: 0 rows × 0 columns</strong>")
        )

    def test_vanilla(self):
        df = SimpleOrg.convert(pd.DataFrame(sample_data()))
        df2 = df.vanilla()
        assert isinstance(df, SimpleOrg)
        assert isinstance(df2, pd.DataFrame)
        assert not isinstance(df2, AbsDf)

    def test_dtype(self):
        df = SimpleOrg.convert(pd.DataFrame(sample_data()))
        df2 = df.untyped()
        assert isinstance(df, SimpleOrg)
        assert isinstance(df2, UntypedDf)

    def test_untyped_convert(self):
        df = SimpleOrg.convert(pd.DataFrame(sample_data()))
        assert df.__class__.__name__ == "SimpleOrg"
        df2 = SimpleOrg(pd.DataFrame(sample_data()))
        assert df2.__class__.__name__ == "SimpleOrg"
        df3 = SimpleOrg(sample_data())
        assert df3.__class__.__name__ == "SimpleOrg"

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
        assert df2.__class__.__name__ == "MultiIndexOrg"
        assert df3.__class__.__name__ == "MultiIndexOrg"

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


if __name__ == "__main__":
    pytest.main()
