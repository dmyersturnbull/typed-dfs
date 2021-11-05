import pytest

from typeddfs import ExampleDfs, LazyDf
from typeddfs.abs_dfs import AbsDf


class TestDatasets:
    def test_sns(self):
        dfx = ExampleDfs.tips
        assert isinstance(dfx, LazyDf)
        assert dfx.name == "tips"
        assert issubclass(dfx.clazz, AbsDf)
        df = dfx.df
        assert df.column_names() == ["total_bill", "tip", "sex", "smoker", "day", "time", "size"]
        assert len(df) > 10
        dfx2 = LazyDf.from_df(df)
        assert dfx2.name == df.__class__.__name__


if __name__ == "__main__":
    pytest.main()
