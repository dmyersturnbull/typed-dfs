import pandas as pd
import pytest

# noinspection PyProtectedMember
from typeddfs.untyped_dfs import UntypedDf

from . import TypedTrivial, sample_data, sample_data_2, sample_data_str


class TestBase:
    def test_pretty(self):
        assert (
            UntypedDf()._repr_html_().startswith("<strong>UntypedDf: 0 rows × 0 columns</strong>")
        )
        df = UntypedDf(sample_data()).set_index(["abc", "123"])
        assert (
            UntypedDf(df)
            ._repr_html_()
            .startswith("<strong>UntypedDf: 2 rows × 1 columns, 2 index columns</strong>")
        )

    def test_st(self):
        df = UntypedDf().convert(pd.DataFrame(sample_data()))
        assert len(df[df["xyz"] == 6]) == 1
        assert len(df.st(df["xyz"] == 6)) == 1
        assert len(df.st(xyz=6)) == 1
        assert len(df.st(df["xyz"] == 6, df["xyz"] == 6)) == 1
        assert len(df.st(df["xyz"] == 6, df["xyz"] == 3)) == 0
        assert len(df.st(df["xyz"] == 6, xyz=6)) == 1
        assert len(df.st(df["xyz"] == 6, xyz=2)) == 0
        assert len(df.st(xyz=1, abc=1)) == 0
        assert len(df.st(df["xyz"] == 6, df["xyz"] == 1, xyz=6)) == 0

    def test_only(self):
        df = UntypedDf().convert(pd.DataFrame(sample_data_2()))
        with pytest.raises(ValueError):
            df.only("multi", exclude_na=True)
        with pytest.raises(KeyError):
            df.only("abc")
        assert df.only("only") == 1
        assert df.only("only", exclude_na=True) == 1
        assert pd.isna(df.only("none"))
        with pytest.raises(ValueError):
            df.only("none", exclude_na=True)

    def test_cfirst(self):
        df = TypedTrivial(sample_data())
        assert df.column_names() == ["abc", "123", "xyz"]
        df2 = df.cfirst(["xyz", "123", "abc"])
        assert df2.column_names() == ["xyz", "123", "abc"]
        assert df.column_names() == ["abc", "123", "xyz"]
        assert df.cfirst("xyz").column_names() == ["xyz", "abc", "123"]
        # empty DF
        assert TypedTrivial(df[df["abc"] == 9999]).cfirst("123").column_names() == [
            "123",
            "abc",
            "xyz",
        ]

    def test_sort_col(self):
        df = TypedTrivial.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural("abc")
        assert df2.index_names() == []
        assert df2.index.tolist() == [1, 0]  # reversed

    def test_drop_cols(self):
        df = TypedTrivial(sample_data())
        df2 = df.drop_cols(["abc", "123"])
        assert list(df.columns) == ["abc", "123", "xyz"]
        assert list(df2.columns) == ["xyz"]
        df3 = df.drop_cols("777")
        assert list(df3.columns) == ["abc", "123", "xyz"]

    def test_no_detype(self):
        df = TypedTrivial(sample_data())
        assert isinstance(df, TypedTrivial)
        assert isinstance(df.reset_index(), TypedTrivial)
        assert isinstance(df.reindex(), TypedTrivial)
        assert isinstance(df.sort_natural("abc"), TypedTrivial)
        assert isinstance(df.sort_values("abc"), TypedTrivial)
        assert isinstance(df.copy(), TypedTrivial)
        assert isinstance(df.abs(), TypedTrivial)
        assert isinstance(df.drop_duplicates(), TypedTrivial)
        assert isinstance(df.bfill(0), TypedTrivial)
        assert isinstance(df.ffill(0), TypedTrivial)
        assert isinstance(df.replace(123, 1), TypedTrivial)
        assert isinstance(df.applymap(lambda s: s), TypedTrivial)
        assert isinstance(df.drop(1), TypedTrivial)
        assert isinstance(df.astype(str), TypedTrivial)
        assert isinstance(df.drop("abc", axis=1), TypedTrivial)
        assert isinstance(df.dropna(), TypedTrivial)
        assert isinstance(df.fillna(0), TypedTrivial)
        assert isinstance(df.append(df), TypedTrivial)
        assert isinstance(df.rename(columns=dict(abc="twotwentytwo")), TypedTrivial)

    def test_set_index(self):
        df = UntypedDf().convert(pd.DataFrame(sample_data()).set_index("abc"))
        assert df.set_index([]).index_names() == []
        assert df.set_index([], append=True).index_names() == ["abc"]
        with pytest.warns(UserWarning):
            df.set_index([], inplace=True)


if __name__ == "__main__":
    pytest.main()
