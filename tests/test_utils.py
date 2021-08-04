import sys

import pytest

from typeddfs.utils import Utils


class TestUtils:
    def test_encoding(self):
        assert Utils.get_encoding("platform") == sys.getdefaultencoding()
        assert "bom" not in Utils.get_encoding("utf8(bom)")
        assert "bom" not in Utils.get_encoding("utf16(bom)")
        assert Utils.get_encoding("UTF-8") == "utf8"
        assert Utils.get_encoding("utf-16") == "utf16"

    def test_table_formats(self):
        formats = list(Utils.table_formats())
        assert len(formats) > 10
        assert "simple" in formats


if __name__ == "__main__":
    pytest.main()
