# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
import pandas as pd
import pytest

# noinspection PyProtectedMember
from typeddfs.df_errors import UnsupportedOperationError
from typeddfs.untyped_dfs import UntypedDf

from . import Col1, Ind1, Trivial, sample_data, sample_data_2, sample_data_str


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

    def test_of(self):
        expected = [[1, 2, 3], [4, 5, 6]]
        df = UntypedDf.convert(pd.DataFrame(sample_data()))
        assert df.to_numpy().tolist() == expected
        df = UntypedDf.of(pd.DataFrame(sample_data()))
        assert df.to_numpy().tolist() == expected
        df = UntypedDf.of(sample_data())
        assert df.to_numpy().tolist() == expected

    def test_of_concat(self):
        df1 = UntypedDf.of(pd.DataFrame(sample_data()))
        df2 = UntypedDf.of(pd.DataFrame(sample_data()))
        df = UntypedDf.of([df1, df2])
        assert len(df) == len(df1) + len(df2) > 0

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
        df = Trivial(sample_data())
        assert df.column_names() == ["abc", "123", "xyz"]
        df2 = df.cfirst(["xyz", "123", "abc"])
        assert df2.column_names() == ["xyz", "123", "abc"]
        assert df.column_names() == ["abc", "123", "xyz"]
        assert df.cfirst("xyz").column_names() == ["xyz", "abc", "123"]
        # empty DF
        assert Trivial(df[df["abc"] == 9999]).cfirst("123").column_names() == [
            "123",
            "abc",
            "xyz",
        ]

    def test_sort_col(self):
        df = Trivial.convert(pd.DataFrame(sample_data_str()))
        df2 = df.sort_natural("abc")
        assert df2.index_names() == []
        assert df2.index.tolist() == [1, 0]  # reversed

    def test_drop_cols(self):
        df = Trivial(sample_data())
        df2 = df.drop_cols(["abc", "123"])
        assert list(df.columns) == ["abc", "123", "xyz"]
        assert list(df2.columns) == ["xyz"]
        df3 = df.drop_cols("777")
        assert list(df3.columns) == ["abc", "123", "xyz"]

    def test_drop_cols_2(self):
        df = Trivial(sample_data())
        df2 = df.drop_cols("abc", "123")
        assert list(df.columns) == ["abc", "123", "xyz"]
        assert list(df2.columns) == ["xyz"]

    def test_no_detype(self):
        df = Trivial(sample_data())
        assert isinstance(df, Trivial)
        assert isinstance(df.reset_index(), Trivial)
        assert isinstance(df.reindex(), Trivial)
        assert isinstance(df.sort_natural("abc"), Trivial)
        assert isinstance(df.sort_values("abc"), Trivial)
        assert isinstance(df.copy(), Trivial)
        assert isinstance(df.abs(), Trivial)
        assert isinstance(df.drop_duplicates(), Trivial)
        assert isinstance(df.bfill(axis=0), Trivial)
        assert isinstance(df.ffill(axis=0), Trivial)
        assert isinstance(df.replace(123, 1), Trivial)
        assert isinstance(df.applymap(lambda s: s), Trivial)
        assert isinstance(df.drop(1), Trivial)
        assert isinstance(df.astype(str), Trivial)
        assert isinstance(df.drop("abc", axis=1), Trivial)
        assert isinstance(df.dropna(), Trivial)
        assert isinstance(df.fillna(0), Trivial)
        assert isinstance(df.rename(columns=dict(abc="twotwentytwo")), Trivial)

    def test_set_index(self):
        df = UntypedDf.convert(pd.DataFrame(sample_data()).set_index("abc"))
        assert df.set_index([]).index_names() == []
        assert df.set_index([], append=True).index_names() == ["abc"]
        with pytest.raises(UnsupportedOperationError):
            df.set_index([], inplace=True)

    def test_iter_rc(self):
        df = UntypedDf.convert(pd.DataFrame(sample_data()))
        expected = [((0, 0), 1), ((0, 1), 2), ((0, 2), 3), ((1, 0), 4), ((1, 1), 5), ((1, 2), 6)]
        assert list(df.iter_row_col()) == expected

    def test_set_attrs(self):
        df = UntypedDf.convert(pd.DataFrame(sample_data()))
        df2 = df.set_attrs(animal="fishies")
        assert df2.attrs == dict(animal="fishies")
        assert df.attrs == {}

    def test_preserve_attrs(self):
        df = Col1([pd.Series(dict(abc="hippo"))])
        df2 = df.set_attrs(animal="fishies")
        df3 = Col1.convert(df2)
        df4 = Col1.convert(df)
        df5 = Col1.of(df2)
        assert df3.attrs == dict(animal="fishies")
        assert df4.attrs == {}
        assert df5.attrs == dict(animal="fishies")

    def test_concat_no_preserve_attrs(self):
        df0 = Col1([pd.Series(dict(abc="hippo"))])
        df1 = df0.set_attrs(animal="fishies")
        df2 = df0.set_attrs(animal="hippos")
        df = UntypedDf.of([df1, df2])
        assert df0.attrs == {}
        assert len(df) == len(df1) + len(df2) > 0
        assert df.attrs == {}

    def test_concat_preserve_attrs(self):
        df0 = Col1([pd.Series(dict(abc="x")), pd.Series(dict(abc="y"))])
        df1 = df0.set_attrs(animal="fishies")
        df2 = df0.set_attrs(animal="hippos")
        df = Col1.of([df1, df2], keys=["one", "two"])
        assert df0.attrs == {}
        assert len(df) == len(df1) + len(df2) > 0
        assert df.attrs == {"one": {"animal": "fishies"}, "two": {"animal": "hippos"}}

    def test_concat_ignore_index(self):
        df1 = Col1([pd.Series(dict(abc="hippo"))])
        df2 = Col1([pd.Series(dict(abc="hippo"))])
        df = Col1.of([df1, df2])
        assert len(df) == 2
        assert df.index.tolist() == [0, 1]

    def test_concat_ignore_index_index(self):
        df1 = Ind1([pd.Series(dict(abc="elephant"))])
        df2 = Ind1([pd.Series(dict(abc="lion"))])
        df = Ind1.of([df1, df2])
        assert len(df) == 2
        assert df.columns.tolist() == []
        assert df.index_names() == ["abc"]
        assert df.index.tolist() == ["elephant", "lion"]


if __name__ == "__main__":
    pytest.main()
