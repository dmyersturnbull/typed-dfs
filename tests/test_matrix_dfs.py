import numpy as np
import pandas as pd
import pytest

from typeddfs.builders import MatrixDfBuilder


class TestMatrixDfs:
    def test(self):
        matrix_type = (MatrixDfBuilder("T").dtype(np.float64)).build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        assert len(df) == 2
        assert df.cols == ["b", "a"]
        assert df.rows == ["b", "a"]
        assert not df.is_symmetric()
        ltr = df.triangle()
        assert [str(x) for x in ltr.flatten().tolist()] == [
            str(x) for x in [11.0, np.nan, 21.0, 22.0]
        ]
        srt = df.sort_alphabetical()
        assert srt.cols == ["a", "b"]
        assert srt.rows == ["a", "b"]
        long = df.long_form()
        assert long.columns.tolist() == ["row", "column", "value"]
        assert long["value"].map(str).tolist() == [str(x) for x in [11.0, 12.0, 21.0, 22.0]]


if __name__ == "__main__":
    pytest.main()
