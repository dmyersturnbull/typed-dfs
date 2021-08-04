import numpy as np
import pandas as pd
from pandas.errors import IntCastingNaNError
import pytest
from typeddfs.matrix_dfs import MatrixDf
from typeddfs.df_errors import VerificationFailedError, InvalidDfError
from typeddfs import AffinityMatrixDf
from typeddfs.builders import MatrixDfBuilder, AffinityMatrixDfBuilder


class TestMatrixDfs:
    def test_matrix(self):
        matrix_type = MatrixDfBuilder("T").build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        assert isinstance(df, MatrixDf)
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
        assert long["value"].map(str).tolist() == [str(x) for x in [11, 12, 21, 22]]

    def test_matrix_dtype(self):
        matrix_type = (MatrixDfBuilder("T").dtype(np.float16)).build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        assert isinstance(df, MatrixDf)
        assert df.dtypes.tolist() == [np.float16, np.float16]

    def test_matrix_bad_nan(self):
        matrix_type = MatrixDfBuilder("T").dtype(np.int32).build()
        df = pd.DataFrame([[11, float("NaN")], [21, 22]], columns=["b", "a"], index=["b", "a"])
        matrix_type(df)
        with pytest.raises(IntCastingNaNError):
            matrix_type.convert(df)

    def test_matrix_bad_inf(self):
        matrix_type = MatrixDfBuilder("T").dtype(np.int32).build()
        df = pd.DataFrame([[11, float("inf")], [21, 22]], columns=["b", "a"], index=["b", "a"])
        matrix_type(df)
        with pytest.raises(IntCastingNaNError):
            matrix_type.convert(df)

    def test_matrix_bad_cols(self):
        matrix_type = MatrixDfBuilder("T").dtype(np.int32).build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=[1, 2], index=[1, 2])
        with pytest.raises(InvalidDfError):
            matrix_type.convert(df)

    def test_condition_pass(self):
        matrix_type = MatrixDfBuilder("T").verify(lambda d: len(d) == 2).build()
        assert matrix_type.is_strict()
        df = pd.DataFrame([[11, float("NaN")], [21, 22]], columns=["b", "a"], index=["b", "a"])
        matrix_type.convert(df)

    def test_condition_fail(self):
        matrix_type = MatrixDfBuilder("T").verify(lambda d: len(d) == 4).build()
        assert matrix_type.is_strict()
        df = pd.DataFrame([[11, float("NaN")], [21, 22]], columns=["b", "a"], index=["b", "a"])
        with pytest.raises(VerificationFailedError):
            matrix_type.convert(df)

    def test_affinity_matrix(self):
        matrix_type = AffinityMatrixDfBuilder("T").build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        assert isinstance(df, AffinityMatrixDf)
        assert isinstance(df.transpose(), AffinityMatrixDf)
        assert df.symmetrize().flatten().tolist() == [11, (12 + 21) / 2, (12 + 21) / 2, 22]

    def test_new(self):
        mx = MatrixDf.new_df(0, 0)
        assert len(mx) == len(mx.columns) == 0
        mx = MatrixDf.new_df(2, 2)
        assert mx.flatten().tolist() == [0, 0, 0, 0]
        mx = MatrixDf.new_df(2, 2, fill=3)
        assert mx.flatten().tolist() == [3, 3, 3, 3]

    def test_new_affinity_matrix(self):
        mx = AffinityMatrixDf.new_df(0)
        assert len(mx) == len(mx.columns) == 0
        mx = AffinityMatrixDf.new_df(2)
        assert mx.flatten().tolist() == [0, 0, 0, 0]
        mx = AffinityMatrixDf.new_df(2, fill=3)
        assert mx.flatten().tolist() == [3, 3, 3, 3]


if __name__ == "__main__":
    pytest.main()
