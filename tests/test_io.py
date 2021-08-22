import numpy as np
import pytest
from lxml.etree import XMLSyntaxError  # nosec
from typeddfs.utils import Utils

from typeddfs.df_errors import NoValueError

from typeddfs.untyped_dfs import UntypedDf

from . import Ind1, Ind2, Trivial, sample_data, tmpfile, logger, Ind2Col2, sample_data_ind2_col2


class TestReadWrite:
    def test_feather_lz4(self):
        with tmpfile(".feather") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_feather(path, compression="lz4")
            df2 = Ind2.read_feather(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_feather_zstd(self):
        with tmpfile(".feather") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_feather(path, compression="zstd")
            df2 = Ind2.read_feather(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_csv_gz(self):
        with tmpfile(".csv.gz") as path:
            df = UntypedDf(sample_data())
            df.to_csv(path)
            df2 = UntypedDf.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_untyped_read_write_csv(self):
        with tmpfile(".csv") as path:
            for indices in [None, "abc", ["abc", "xyz"]]:
                df = UntypedDf(sample_data())
                if indices is not None:
                    df = df.set_index(indices)
                df.to_csv(path)
                df2 = UntypedDf.read_csv(path)
                assert list(df2.index.names) == [None]
                assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_write_passing_index(self):
        with tmpfile(".csv") as path:
            df = Trivial(sample_data())
            df.to_csv(path, index=["abc"])  # fine
            df = UntypedDf(sample_data())
            df.to_csv(path, index=["abc"])  # calls super immediately

    def test_typed_read_write_csv_noindex(self):
        with tmpfile(".csv") as path:
            df = Trivial(sample_data())
            df.to_csv(path)
            df2 = Trivial.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_typed_read_write_csv_singleindex(self):
        with tmpfile(".csv") as path:
            df = Ind1.convert(Ind1(sample_data()))
            df.to_csv(path)
            assert df.index_names() == ["abc"]
            assert df.column_names() == ["123", "xyz"]
            df2 = Ind1.read_csv(path)
            assert df2.index_names() == ["abc"]
            assert df2.column_names() == ["123", "xyz"]

    def test_typed_read_write_csv_multiindex(self):
        with tmpfile(".csv") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_csv(path)
            assert df.index_names() == ["abc", "xyz"]
            assert df.column_names() == ["123"]
            df2 = Ind2.read_csv(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_parquet(self):
        with tmpfile(".parquet") as path:
            df = UntypedDf(sample_data())
            df.to_parquet(path)
            df2 = UntypedDf.read_parquet(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_parquet_feather_dtypes(self):
        dtypes = [
            str,
            bool,
            np.byte,
            np.ubyte,
            np.short,
            np.ushort,
            np.single,
            np.int32,
            np.intc,
            np.half,
            np.float16,
            np.double,
            np.float64,
        ]
        for suffix, fn in [(".snappy", "parquet"), (".feather", "feather")]:
            with tmpfile(suffix) as path:
                for dtype in dtypes:
                    logger.info(dtype)
                    df = Ind2Col2.convert(Ind2Col2(sample_data_ind2_col2())).astype(dtype)
                    assert list(df.index.names) == ["qqq", "rrr"]
                    assert list(df.columns) == ["abc", "xyz"]
                    getattr(df, "to_" + fn)(path)
                    df2 = getattr(Ind2Col2, "read_" + fn)(path)
                    assert list(df2.index.names) == ["qqq", "rrr"]
                    assert list(df2.columns) == ["abc", "xyz"]

    def test_xml(self):
        with tmpfile(".xml.gz") as path:
            df = UntypedDf(sample_data())
            df.to_csv(path)
            df2 = UntypedDf.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_html_untyped(self):
        with tmpfile(".html") as path:
            df = UntypedDf(sample_data())
            df.to_html(path)
            df2 = UntypedDf.read_html(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}

    def test_html_singleindex(self):
        with tmpfile(".html") as path:
            df = Ind1.convert(Ind1(sample_data()))
            df.to_html(path)
            assert df.index_names() == ["abc"]
            assert df.column_names() == ["123", "xyz"]
            df2 = Ind1.read_html(path)
            assert df2.index_names() == ["abc"]
            assert df2.column_names() == ["123", "xyz"]

    def test_html_multiindex(self):
        with tmpfile(".html") as path:
            df = Ind2.convert(Ind2(sample_data()))
            df.to_html(path)
            assert df.index_names() == ["abc", "xyz"]
            assert df.column_names() == ["123"]
            df2 = Ind2.read_html(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]

    def test_html_invalid(self):
        with tmpfile(".html") as path:
            path.write_text("", encoding="utf8")
            with pytest.raises(XMLSyntaxError):
                UntypedDf.read_html(path)

    def test_html_empty(self):
        with tmpfile(".html") as path:
            path.write_text("<html></html>", encoding="utf8")
            with pytest.raises(NoValueError):
                UntypedDf.read_html(path)

    """
    # TODO re-enable when we get a tables 3.9 wheels on Windows
    def test_hdf(self):
        with tmpfile(".h5") as path:
            df = TypedMultiIndex.convert(TypedMultiIndex(sample_data()))
            df.to_hdf(path)
            df2 = TypedMultiIndex.read_hdf(path)
            assert df2.index_names() == ["abc", "xyz"]
            assert df2.column_names() == ["123"]
    """


if __name__ == "__main__":
    pytest.main()
