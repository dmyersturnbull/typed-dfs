import pytest

from typeddfs.df_errors import FilenameSuffixError
from typeddfs.file_formats import FileFormat


class TestFileFormats:
    def test(self):
        assert FileFormat.from_path("abc.csv.gz") is FileFormat.csv
        assert FileFormat.from_path("abc.txt.xz") is FileFormat.lines
        assert FileFormat.from_path("abc.snappy") is FileFormat.parquet
        with pytest.raises(FilenameSuffixError):
            FileFormat.from_path("abc.what")


if __name__ == "__main__":
    pytest.main()
