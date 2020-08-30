import pandas as pd
import pytest

# noinspection PyProtectedMember
from typeddfs.base_dfs import AbsDf, AsymmetricDfError
from typeddfs.untyped_dfs import UntypedDf

from . import (
    TypedMultiIndex,
    TypedSingleIndex,
    TypedSymmetric,
    TypedTrivial,
    sample_data,
    sample_data_str,
    sample_symmetric_df,
)


class TestTyped:
    def test_pretty(self):
        assert (
            UntypedDf()._repr_html_().startswith("<strong>UntypedDf: 0 rows × 0 columns</strong>")
        )

    def test_vanilla(self):
        df = TypedTrivial.convert(pd.DataFrame(sample_data()))
        df2 = df.vanilla()
        assert isinstance(df, TypedTrivial)
        assert isinstance(df2, pd.DataFrame)
        assert not isinstance(df2, AbsDf)

    def test_convert_fail(self):
        with pytest.raises(TypeError):
            TypedTrivial.convert(55)

    def test_detype(self):
        df = TypedTrivial.convert(pd.DataFrame(sample_data()))
        df2 = df.untyped()
        assert isinstance(df, TypedTrivial)
        assert isinstance(df2, UntypedDf)

    def test_is_multindex(self):
        assert not TypedTrivial.convert(pd.DataFrame(sample_data())).is_multindex()
        assert not TypedSingleIndex.convert(pd.DataFrame(sample_data())).is_multindex()
        assert TypedMultiIndex.convert(pd.DataFrame(sample_data())).is_multindex()

    def test_lengths(self):
        df = TypedSingleIndex.convert(pd.DataFrame(sample_data()))
        assert df.n_columns() == 2
        assert df.n_indices() == 1
        assert df.n_rows() == 2

    def test_sort_no_index(self):
        df = TypedTrivial.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.index_names() == []

    def test_sort_single_index(self):
        df = TypedSingleIndex.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.column_names() == ["123", "xyz"]
        assert df2.index_names() == ["abc"]
        assert df2.index.tolist() == ["bbb", "zzz"]

    def test_sort_multiindex(self):
        df = TypedMultiIndex.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural_index()
        assert df2.column_names() == ["123"]
        assert df2.index_names() == ["abc", "xyz"]
        assert df2.index.tolist() == [("bbb", 6), ("zzz", 3)]

    def test_change(self):
        # should be in place
        df = pd.DataFrame(sample_data())
        assert df.__class__.__name__ == "DataFrame"
        TypedTrivial._change(df)
        assert df.__class__.__name__ == "TypedTrivial"

    def test_not_inplace(self):
        df = pd.DataFrame(sample_data())
        df2 = TypedMultiIndex(df)
        df3 = TypedMultiIndex.convert(df)
        assert df.__class__.__name__ == "DataFrame"
        assert df2.__class__.__name__ == "TypedMultiIndex"
        assert df3.__class__.__name__ == "TypedMultiIndex"

    def test_index_names(self):
        df = TypedMultiIndex.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz"]
        df = TypedTrivial.convert(pd.DataFrame(sample_data()))
        assert isinstance(df.index_names(), list)
        assert df.index_names() == []

    def test_column_names(self):
        df = TypedTrivial(sample_data())
        # df.columns == [...] would fail because it would resolve to array==array, which is ambiguous
        assert isinstance(df.column_names(), list)
        assert df.column_names() == ["abc", "123", "xyz"]

    def test_isvalid(self):
        df = pd.DataFrame(sample_data())
        assert TypedMultiIndex.is_valid(df)
        assert not TypedMultiIndex.is_valid(df.drop("abc", axis=1))

    def test_symmetric(self):
        df = sample_symmetric_df()
        TypedSymmetric(df)
        TypedSymmetric.convert(df)
        assert TypedSymmetric.is_valid(df)

    def test_asymmetric_multiindex(self):
        df = pd.DataFrame(sample_data()).set_index(["abc", "xyz"])
        assert not TypedSymmetric.is_valid(df)
        with pytest.raises(AsymmetricDfError):
            TypedSymmetric.convert(df)

    def test_asymmetric_shape(self):
        df = pd.DataFrame(sample_data())
        assert not TypedSymmetric.is_valid(df)
        with pytest.raises(AsymmetricDfError):
            TypedSymmetric.convert(df)

    def test_asymmetric_names(self):
        df = pd.DataFrame(sample_symmetric_df().reset_index(drop=True))
        assert not TypedSymmetric.is_valid(df)
        with pytest.raises(AsymmetricDfError):
            TypedSymmetric.convert(df)


if __name__ == "__main__":
    pytest.main()
