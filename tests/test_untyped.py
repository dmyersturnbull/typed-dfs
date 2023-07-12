# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
import pandas as pd
import pytest

# noinspection PyProtectedMember
from typeddfs.untyped_dfs import UntypedDf


class TestUntyped:
    def test_getitem(self):
        df = UntypedDf([pd.Series(dict(abc="xyz", ind="qqq"))]).set_index("ind")
        assert list(df["abc"]) == ["xyz"]
        assert list(df["ind"]) == ["qqq"]
        df = UntypedDf([pd.Series(dict(abc="xyz", ind="qqq"))]).set_index(["ind"])
        assert list(df["abc"]) == ["xyz"]
        assert list(df["ind"]) == ["qqq"]

    def test_new(self):
        df = UntypedDf.new_df()
        assert isinstance(df, UntypedDf)
        assert len(df) == len(df.columns) == 0
        df = UntypedDf.new_df(rows=2, cols=2)
        assert isinstance(df, UntypedDf)
        assert len(df) == len(df.columns) == 2
        assert df.column_names() == ["0", "1"]
        df = UntypedDf.new_df(rows=2, cols=["one", "two"])
        assert isinstance(df, UntypedDf)
        assert len(df) == len(df.columns) == 2
        assert df.column_names() == ["one", "two"]


if __name__ == "__main__":
    pytest.main()
