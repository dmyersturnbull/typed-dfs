from pathlib import Path

import pytest

from typeddfs.df_errors import FilenameSuffixError
from typeddfs.file_formats import FileFormat, CompressionFormat


class TestFileFormats:
    def test_from_path(self):
        assert FileFormat.from_path("abc.csv.gz") is FileFormat.csv
        assert FileFormat.from_path("abc.txt.xz") is FileFormat.lines
        assert FileFormat.from_path("abc.snappy") is FileFormat.parquet
        with pytest.raises(FilenameSuffixError):
            FileFormat.from_path("abc.what")

    def test_variants(self):
        expected = {".molly", ".molly.gz", ".molly.zip", ".molly.xz", ".molly.bz2"}
        assert FileFormat.json.compressed_variants(".molly") == expected
        expected = {".gz", ".gz.gz", ".gz.zip", ".gz.xz", ".gz.bz2"}
        assert FileFormat.json.compressed_variants(".gz") == expected
        expected = {".", "..gz", "..zip", "..xz", "..bz2"}
        assert FileFormat.json.compressed_variants(".") == expected
        expected = {"", ".gz", ".zip", ".xz", ".bz2"}
        assert FileFormat.json.compressed_variants("") == expected
        assert FileFormat.feather.compressed_variants(".feather") == {".feather"}
        assert FileFormat.feather.compressed_variants(".whatever") == {".whatever"}

    def test_compression_from_path(self):
        assert FileFormat.compression_from_path("hello.nothing.gz") == CompressionFormat.gz
        assert FileFormat.compression_from_path(".gz") == CompressionFormat.gz
        assert FileFormat.compression_from_path("hello.nothing") == CompressionFormat.none
        assert FileFormat.compression_from_path("hello.nothing.") == CompressionFormat.none

    def test_strip_compression(self):
        assert FileFormat.strip_compression("hello.nothing.gz") == Path("hello.nothing")
        assert FileFormat.strip_compression("gz.gz.gz") == Path("gz.gz")
        assert FileFormat.strip_compression(".xz") == Path("")


if __name__ == "__main__":
    pytest.main()
