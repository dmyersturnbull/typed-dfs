# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from lxml.etree import XMLSyntaxError  # nosec

from typeddfs.df_errors import NoValueError
from typeddfs.untyped_dfs import UntypedDf

from . import Ind1NonStrict as Ind1
from . import Ind2Col2NonStrict as Ind2Col2
from . import Ind2NonStrict as Ind2
from . import (
    Trivial,
    logger,
    sample_data,
    sample_data_ind2_col2,
    sample_data_ind2_col2_pd_na,
    tmpfile,
)


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

    def test_records(self):
        df = UntypedDf(sample_data())
        records = df.to_records()
        df2 = UntypedDf.from_records(records)
        assert isinstance(df2, UntypedDf)

    def test_numeric_dtypes(self):
        dtypes = [
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
            pd.StringDtype(),
            pd.Int64Dtype(),
            pd.UInt64Dtype(),
            pd.Int32Dtype(),
            pd.UInt32Dtype(),
            pd.Int16Dtype(),
            pd.UInt16Dtype(),
            pd.Int8Dtype(),
            pd.UInt8Dtype(),
        ]
        for suffix, fn in [
            (".parquet", "parquet"),
            (".feather", "feather"),
            (".xml", "xml"),
            (".csv", "csv"),
            (".tsv", "tsv"),
            (".json", "json"),
            (".xlsx", "xlsx"),
            (".xls", "xls"),
            (".xlsb", "xlsb"),
            (".ods", "ods"),
            (".pickle", "pickle"),
        ]:
            with tmpfile(suffix) as path:
                for dtype in dtypes:
                    try:
                        df = Ind2Col2.convert(Ind2Col2(sample_data_ind2_col2())).astype(dtype)
                        assert list(df.index.names) == ["qqq", "rrr"]
                        assert list(df.columns) == ["abc", "xyz"]
                        getattr(df, "to_" + fn)(path)
                        df2 = getattr(Ind2Col2, "read_" + fn)(path)
                        assert list(df2.index.names) == ["qqq", "rrr"]
                        assert list(df2.columns) == ["abc", "xyz"]
                    except Exception:
                        logger.error(f"Failed on path {path}, dtype {dtype}")
                        raise

    def test_numeric_nullable_dtypes(self):
        dtypes = [
            pd.StringDtype(),
            pd.BooleanDtype(),
            pd.Float64Dtype(),
            pd.Float32Dtype(),
            pd.Int64Dtype(),
            pd.UInt64Dtype(),
            pd.Int32Dtype(),
            pd.UInt32Dtype(),
            pd.Int16Dtype(),
            pd.UInt16Dtype(),
            pd.Int8Dtype(),
            pd.UInt8Dtype(),
            pd.StringDtype(),
        ]
        for suffix, fn in [
            (".parquet", "parquet"),
            (".feather", "feather"),
            (".csv", "csv"),
            (".tsv", "tsv"),
            (".json", "json"),
            (".xlsx", "xlsx"),
            (".xls", "xls"),
            (".xlsb", "xlsb"),
            (".ods", "ods"),
            (".pickle", "pickle"),
            (".xml", "xml"),
        ]:
            for dtype in dtypes:
                with tmpfile(suffix) as path:
                    try:
                        df = Ind2Col2.convert(Ind2Col2(sample_data_ind2_col2_pd_na())).astype(dtype)
                        assert list(df.index.names) == ["qqq", "rrr"]
                        assert list(df.columns) == ["abc", "xyz"]
                        getattr(df, "to_" + fn)(path)
                        df2 = getattr(Ind2Col2, "read_" + fn)(path)
                        assert list(df2.index.names) == ["qqq", "rrr"]
                        assert list(df2.columns) == ["abc", "xyz"]
                    except Exception:
                        logger.error(f"Failed on path {path}, dtype {dtype}")
                        raise

    """
    # TODO: waiting for upstream: https://github.com/dmyersturnbull/typed-dfs/issues/46
    def test_raw_to_xml(self):
        dtypes = [
            pd.StringDtype(),
            pd.BooleanDtype(),
            pd.Float64Dtype(),
            pd.Float32Dtype(),
            pd.Int64Dtype(),
            pd.UInt64Dtype(),
            pd.Int32Dtype(),
            pd.UInt32Dtype(),
            pd.Int16Dtype(),
            pd.UInt16Dtype(),
            pd.Int8Dtype(),
            pd.UInt8Dtype(),
            pd.StringDtype(),
        ]
        data = [
            pd.Series({"abc": 1, "xyz": pd.NA}),
            pd.Series({"abc": pd.NA, "xyz": 0}),
        ]
        failed = {}
        for dtype in dtypes:
            df = pd.DataFrame(data).astype(dtype)
            try:
                df.to_xml()
            except TypeError as e:
                logger.error(dtype, exc_info=True)
                failed[str(dtype)] = str(e)
        assert failed == [], f"Failed on dtypes: {failed}"
    """

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
            path.write_text("", encoding="utf-8")
            with pytest.raises(XMLSyntaxError):
                UntypedDf.read_html(path)

    def test_html_empty(self):
        with tmpfile(".html") as path:
            path.write_text("<html></html>", encoding="utf-8")
            with pytest.raises(NoValueError):
                UntypedDf.read_html(path)

    def test_read_toml(self):
        data = """
        [[row]]
        # a comment
        key = "value"
        """
        s = StringIO(data)
        df = UntypedDf.read_toml(s)
        assert df.column_names() == ["key"]
        assert df.values.tolist() == [["value"]]

    def test_read_toml_jagged(self):
        data = """
        [[row]]
        key = "value1"
        [[row]]
        key = "value2"
        kitten = "elephant"
        cuteness = 10.3
        """
        s = StringIO(data)
        df = UntypedDf.read_toml(s)
        assert df.column_names() == ["key", "kitten", "cuteness"]
        xx = df.fillna(0).values.tolist()
        assert xx == [["value1", 0, 0], ["value2", "elephant", 10.3]]

    def test_read_ini(self):
        data = """
        [section]
        ; a comment
        key = value
        """
        s = StringIO(data)
        df = UntypedDf.read_ini(s)
        assert df.column_names() == ["key", "value"]
        assert df.values.tolist() == [["section.key", "value"]]

    def test_read_properties(self):
        data = r"""
        [section]
        # a comment
        ! another comment
        k\:e\\y = v:a\\lue
        """
        s = StringIO(data)
        df = UntypedDf.read_properties(s)
        assert df.column_names() == ["key", "value"]
        assert df.values.tolist() == [[r"section.k:e\y", r"v:a\lue"]]
        data: str = df.to_properties()
        lines = [s.strip() for s in data.splitlines()]
        assert "[section]" in lines
        assert r"k\:e\\y = v:a\\lue" in lines
        s = StringIO(data)
        df2 = UntypedDf.read_properties(s)
        assert df2.values.tolist() == df.values.tolist()

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
