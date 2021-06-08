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


if __name__ == "__main__":
    pytest.main()
