# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
import numpy as np
import pandas as pd
import pytest
from pandas.errors import IntCastingNaNError

from typeddfs.builders import AffinityMatrixDfBuilder, MatrixDfBuilder
from typeddfs.df_errors import VerificationFailedError
from typeddfs.matrix_dfs import AffinityMatrixDf, MatrixDf

from . import get_resource, tmpfile


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

    def test_matrix_int_names(self):
        matrix_type = AffinityMatrixDfBuilder("T").dtype(np.int32).build()
        # noinspection PyTypeChecker
        df = pd.DataFrame([[11, 12], [21, 22]], columns=[1, 2], index=[1, 2])
        df: AffinityMatrixDf = matrix_type.convert(df)
        assert df.rows == df.cols

    def test_condition_pass(self):
        matrix_type = MatrixDfBuilder("T").verify(lambda d: len(d) == 2).build()
        df = pd.DataFrame([[11, float("NaN")], [21, 22]], columns=["b", "a"], index=["b", "a"])
        matrix_type.convert(df)

    def test_condition_fail(self):
        matrix_type = MatrixDfBuilder("T").verify(lambda d: len(d) == 4).build()
        df = pd.DataFrame([[11, float("NaN")], [21, 22]], columns=["b", "a"], index=["b", "a"])
        with pytest.raises(VerificationFailedError):
            matrix_type.convert(df)

    def test_read_plain(self):
        matrix = AffinityMatrixDf.read_csv(get_resource("matrix.csv"))
        assert matrix.rows == ["a", "b", "c"]
        s = matrix.to_csv()
        assert len(s.splitlines()) == 4

    def test_affinity_matrix(self):
        matrix_type = AffinityMatrixDfBuilder("T").build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        assert isinstance(df, AffinityMatrixDf)
        assert isinstance(df, matrix_type)
        assert isinstance(df.transpose(), AffinityMatrixDf)
        assert df.symmetrize().flatten().tolist() == [11, (12 + 21) / 2, (12 + 21) / 2, 22]

    def test_affinity_matrix_new_methods(self):
        matrix_type = (
            AffinityMatrixDfBuilder("T").add_methods(fix=lambda dx: dx.convert(dx + 0.5))
        ).build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        assert isinstance(df, AffinityMatrixDf)
        assert isinstance(df, matrix_type)
        assert df.fix().flatten().tolist() == [11.5, 12.5, 21.5, 22.5]

    def test_new(self):
        mx = MatrixDf.new_df()
        assert len(mx) == len(mx.columns) == 0
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

    def test_io(self):
        matrix_type = MatrixDfBuilder("T").build()
        df = pd.DataFrame([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        df = matrix_type.convert(df)
        for s in [".feather", ".snappy", ".csv.gz", ".tsv"]:
            with tmpfile(s) as path:
                df.write_file(path)
                df2 = matrix_type.read_file(path)
            assert df2.flatten().tolist() == [11, 12, 21, 22]

    def test_shuffle(self):
        matrix_type = MatrixDfBuilder("T").build()
        df: MatrixDf = matrix_type.of([[11, 12], [21, 22]], columns=["b", "a"], index=["b", "a"])
        for _ in range(10):
            df2 = df.shuffle(rand=0)
            assert df2.values.tolist() == [[21, 22], [12, 11]]
        df2 = df.shuffle(rand=1)
        assert df2.values.tolist() == [[22, 21], [11, 12]]
        df2 = df.shuffle(rand=None)
        assert len(df2.values.tolist()) == 2


if __name__ == "__main__":
    pytest.main()
