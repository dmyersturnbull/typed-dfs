import pandas as pd
import pytest

# noinspection PyProtectedMember
from typeddfs.base_dfs import AbsDf
from typeddfs.untyped_dfs import UntypedDf

from . import MultiIndexOrg, SimpleOrg, SingleIndexOrg, sample_data, sample_data_2, sample_data_str

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

    def test_detype(self):
        df = SimpleOrg.convert(pd.DataFrame(sample_data()))
        df2 = df.untyped()
        assert isinstance(df, SimpleOrg)
        assert isinstance(df2, UntypedDf)

    def test_is_multindex(self):
        assert not SimpleOrg.convert(pd.DataFrame(sample_data())).is_multindex()
        assert not SingleIndexOrg.convert(pd.DataFrame(sample_data())).is_multindex()
        assert MultiIndexOrg.convert(pd.DataFrame(sample_data())).is_multindex()

    def test_only(self):
        df = SingleIndexOrg.convert(pd.DataFrame(sample_data_2()))
        assert df.only("multi") is None
        with raises(ValueError):
            df.only("multi", exclude_na=True)
        with raises(ValueError):
            df.only("none")
        with raises(ValueError):
            df.only("none", exclude_na=True)
        with raises(KeyError):
            df.only("abc")
        assert df.only("only") == 1
        assert df.only("only", exclude_na=True) == 1

    def test_lengths(self):
        df = SingleIndexOrg.convert(pd.DataFrame(sample_data()))
        assert df.n_columns() == 2
        assert df.n_indices() == 1
        assert df.n_rows() == 2

    def test_sort_col(self):
        df = SimpleOrg.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural("abc")
        assert df2.index_names() == []
        assert df2.index.tolist() == [1, 0]  # reversed

    def test_sort_no_index(self):
        df = SimpleOrg.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.index_names() == []

    def test_sort_single_index(self):
        df = SingleIndexOrg.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.column_names() == ["123", "xyz"]
        assert df2.index_names() == ["abc"]
        assert df2.index.tolist() == ["bbb", "zzz"]

    def test_sort_multiindex(self):
        df = MultiIndexOrg.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.column_names() == ["123"]
        assert df2.index_names() == ["abc", "xyz"]
        assert df2.index.tolist() == [("bbb", 6), ("zzz", 3)]

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
        assert isinstance(df.reindex(), SimpleOrg)
        assert isinstance(df.sort_natural("abc"), SimpleOrg)
        assert isinstance(df.sort_values("abc"), SimpleOrg)
        assert isinstance(df.copy(), SimpleOrg)
        assert isinstance(df.abs(), SimpleOrg)
        assert isinstance(df.drop_duplicates(), SimpleOrg)
        assert isinstance(df.bfill(0), SimpleOrg)
        assert isinstance(df.ffill(0), SimpleOrg)
        assert isinstance(df.replace(123, 1), SimpleOrg)
        assert isinstance(df.applymap(lambda s: s), SimpleOrg)
        assert isinstance(df.drop(1), SimpleOrg)
        assert isinstance(df.astype(str), SimpleOrg)
        assert isinstance(df.drop("abc", axis=1), SimpleOrg)
        assert isinstance(df.dropna(), SimpleOrg)
        assert isinstance(df.fillna(0), SimpleOrg)
        assert isinstance(df.append(df), SimpleOrg)

    def test_change(self):
        # should be in place
        df = pd.DataFrame(sample_data())
        assert df.__class__.__name__ == "DataFrame"
        SimpleOrg._change(df)
        assert df.__class__.__name__ == "SimpleOrg"

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

    def test_isvalid(self):
        df = pd.DataFrame(sample_data())
        assert MultiIndexOrg.is_valid(df)
        assert not MultiIndexOrg.is_valid(df.drop("abc", axis=1))


if __name__ == "__main__":
    pytest.main()
