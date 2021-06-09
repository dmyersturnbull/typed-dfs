import pytest

from typeddfs.df_errors import FilenameSuffixError
from typeddfs.file_formats import DfFileFormat


class TestFileFormats:
    def test(self):
        assert DfFileFormat.from_path("abc.csv.gz") is DfFileFormat.csv
        assert DfFileFormat.from_path("abc.txt.xz") is DfFileFormat.lines
        assert DfFileFormat.from_path("abc.snappy") is DfFileFormat.parquet
        with pytest.raises(FilenameSuffixError):
            DfFileFormat.from_path("abc.what")


if __name__ == "__main__":
    pytest.main()
