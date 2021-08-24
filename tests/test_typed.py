import pandas as pd
import pytest

from typeddfs.abs_dfs import AbsDf
from typeddfs.untyped_dfs import UntypedDf

from . import (
    Ind1,
    Ind1Col1,
    Ind2,
    Ind2Col2,
    Ind2Col2Reserved1,
    Trivial,
    sample_data,
    sample_data_str,
)


class TestTyped:
    def test_pretty(self):
        assert (
            UntypedDf()._repr_html_().startswith("<strong>UntypedDf: 0 rows × 0 columns</strong>")
        )

    def test_vanilla(self):
        df = Trivial.convert(pd.DataFrame(sample_data()))
        df2 = df.vanilla()
        assert isinstance(df, Trivial)
        assert isinstance(df2, pd.DataFrame)
        assert not isinstance(df2, AbsDf)

    def test_convert_fail(self):
        with pytest.raises(TypeError):
            Trivial.convert(55)

    def test_detype(self):
        df = Trivial.convert(pd.DataFrame(sample_data()))
        df2 = df.untyped()
        assert isinstance(df, Trivial)
        assert isinstance(df2, UntypedDf)

    def test_is_multindex(self):
        assert not Trivial.convert(pd.DataFrame(sample_data())).is_multindex()
        assert not Ind1.convert(pd.DataFrame(sample_data())).is_multindex()
        assert Ind2.convert(pd.DataFrame(sample_data())).is_multindex()

    def test_lengths(self):
        df = Ind1.convert(pd.DataFrame(sample_data()))
        assert df.n_columns() == 2
        assert df.n_indices() == 1
        assert df.n_rows() == 2

    def test_sort_no_index(self):
        df = Trivial.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.index_names() == []

    def test_sort_single_index(self):
        df = Ind1.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.column_names() == ["123", "xyz"]
        assert df2.index_names() == ["abc"]
        assert df2.index.tolist() == ["bbb", "zzz"]

    def test_sort_multiindex(self):
        df = Ind2.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.column_names() == ["123"]
        assert df2.index_names() == ["abc", "xyz"]
        assert df2.index.tolist() == [("bbb", 6), ("zzz", 3)]

    def test_meta(self):
        df = Ind2.convert(pd.DataFrame(sample_data_str()))
        df = df.meta()
        assert df.index_names() == ["abc", "xyz"]
        assert df.column_names() == []

    def test_assign(self):
        df = Ind2.convert(pd.DataFrame(sample_data_str()))
        df2 = df.assign(**{"123": "omg"}).vanilla_reset()
        assert df2.values.tolist() == [["zzz", 3, "omg"], ["bbb", 6, "omg"]]

    def test_meta_empty(self):
        df = Trivial({})
        df2 = df.meta()
        assert df is df2
        assert df2.column_names() == []
        assert df2.index_names() == []

    def test_change(self):
        # should be in place
        df = pd.DataFrame(sample_data())
        assert df.__class__.__name__ == "DataFrame"
        Trivial._change(df)
        assert df.__class__.__name__ == "Trivial"

    def test_not_inplace(self):
        df = pd.DataFrame(sample_data())
        df2 = Ind2(df)
        df3 = Ind2.convert(df)
        assert df.__class__.__name__ == "DataFrame"
        assert df2.__class__.__name__ == "Ind2"
        assert df3.__class__.__name__ == "Ind2"

    def test_index_names(self):
        df = Ind2.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz"]
        df = Trivial.convert(pd.DataFrame(sample_data()))
        assert isinstance(df.index_names(), list)
        assert df.index_names() == []

    def test_known_names(self):
        assert Ind1Col1.get_typing().known_names == ["qqq", "abc"]
        assert Ind2Col2.get_typing().known_names == ["qqq", "rrr", "abc", "xyz"]

    def test_column_names(self):
        df = Trivial(sample_data())
        # df.columns == [...] would fail because it would resolve to array==array, which is ambiguous
        assert isinstance(df.column_names(), list)
        assert df.column_names() == ["abc", "123", "xyz"]

    def test_new(self):
        df = Ind2.new_df()
        assert isinstance(df, Ind2)
        assert len(df) == 0
        df = Ind2Col2Reserved1.new_df()
        assert isinstance(df, Ind2Col2Reserved1)
        assert df.index_names() == ["qqq", "rrr"]
        assert df.column_names() == ["abc", "xyz"]
        df = Ind2Col2Reserved1.new_df(reserved=True)
        assert isinstance(df, Ind2Col2Reserved1)
        assert df.index_names() == ["qqq", "rrr"]
        assert df.column_names() == ["abc", "xyz", "res"]

    def test_records(self):
        df = Ind2.convert(pd.DataFrame(sample_data()))
        records = df.to_records()
        df2 = Ind2.from_records(records)
        assert isinstance(df2, Ind2)
        assert df2.values.tolist() == [[2], [5]]
        assert df2.reset_index().values.tolist() == [[1, 3, 2], [4, 6, 5]]


if __name__ == "__main__":
    pytest.main()
