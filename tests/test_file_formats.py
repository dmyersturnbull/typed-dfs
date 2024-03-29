# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from typeddfs.df_errors import FilenameSuffixError
from typeddfs.file_formats import CompressionFormat, FileFormat


class TestFileFormats:
    def test_format_of(self):
        assert FileFormat.of("json") is FileFormat.json
        assert FileFormat.of(FileFormat.json) is FileFormat.json

    def test_compression_of(self):
        assert CompressionFormat.of(CompressionFormat.none) is CompressionFormat.none
        assert CompressionFormat.of("none") is CompressionFormat.none
        assert CompressionFormat.of("gz") is CompressionFormat.gz
        assert CompressionFormat.of("gzip") is CompressionFormat.gz

    def test_is_compressed(self):
        assert CompressionFormat.xz.is_compressed
        assert not CompressionFormat.none.is_compressed

    def test_from_suffix(self):
        assert FileFormat.from_suffix(".csv") is FileFormat.csv
        assert FileFormat.from_suffix(".csv.gz") is FileFormat.csv
        with pytest.raises(FilenameSuffixError):
            FileFormat.from_suffix("abc.csv")
        with pytest.raises(FilenameSuffixError):
            FileFormat.from_suffix(".what")
        assert FileFormat.from_suffix_or_none(".csv.gz") is FileFormat.csv
        assert FileFormat.from_suffix_or_none(".what") is None

    def test_split(self):
        e = (Path("abc") / "xyz", FileFormat.csv, CompressionFormat.gz)
        assert FileFormat.split("abc/xyz.csv.gz") == e
        e = (Path("https://google.com/a"), FileFormat.csv, CompressionFormat.none)
        assert FileFormat.split("https://google.com/a.csv") == e
        e = (Path("https://google.com/a"), None, CompressionFormat.none)
        assert FileFormat.split_or_none("https://google.com/a") == e
        with pytest.raises(FilenameSuffixError):
            FileFormat.split("https://google.com/x")

    def test_from_path(self):
        assert FileFormat.from_path("abc.csv.gz") is FileFormat.csv
        assert FileFormat.from_path("abc.txt.xz") is FileFormat.lines
        assert FileFormat.from_path("abc.snappy") is FileFormat.parquet
        with pytest.raises(FilenameSuffixError):
            FileFormat.from_path("abc.what")
        assert FileFormat.from_path_or_none("abc.snappy") is FileFormat.parquet
        assert FileFormat.from_path_or_none("abc.what") is None

    def test_variants(self):
        expected = {".molly", ".molly.gz", ".molly.xz", ".molly.bz2", ".molly.zst"}
        assert FileFormat.json.compressed_variants(".molly") == expected
        expected = {".gz", ".gz.gz", ".gz.xz", ".gz.bz2", ".gz.zst"}
        assert FileFormat.json.compressed_variants(".gz") == expected
        expected = {".", "..gz", "..xz", "..bz2", "..zst"}
        assert FileFormat.json.compressed_variants(".") == expected
        expected = {"", ".gz", ".xz", ".bz2", ".zst"}
        assert FileFormat.json.compressed_variants("") == expected
        assert FileFormat.feather.compressed_variants(".feather") == {".feather"}
        assert FileFormat.feather.compressed_variants(".whatever") == {".whatever"}

    def test_compression_from_path(self):
        assert CompressionFormat.from_path("hello.nothing.gz") == CompressionFormat.gz
        assert CompressionFormat.from_path(".gz") == CompressionFormat.gz
        assert CompressionFormat.from_path("hello.nothing") == CompressionFormat.none
        assert CompressionFormat.from_path("hello.nothing.") == CompressionFormat.none

    def test_strip_compression(self):
        assert CompressionFormat.strip_suffix("hello.nothing.gz") == Path("hello.nothing")
        assert CompressionFormat.strip_suffix("gz.gz.gz") == Path("gz.gz")
        assert CompressionFormat.strip_suffix(".xz") == Path()
        assert CompressionFormat.strip_suffix("hello") == Path("hello")


if __name__ == "__main__":
    pytest.main()
