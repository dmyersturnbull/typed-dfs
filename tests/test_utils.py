import sys

import pytest

from typeddfs.utils import Utils, TableFormat


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
        x = Utils.table_format("simple")
        assert isinstance(x, TableFormat)

    def test_basic(self):
        assert "sha1" in Utils.insecure_hash_functions()
        assert "__xml_index_" in Utils.banned_names()

    def test_strip_control_chars(self):
        assert Utils.strip_control_chars("ab\ncd") == "abcd"
        assert Utils.strip_control_chars("ab\0\0cℶd") == "abcℶd"
        assert Utils.strip_control_chars("ℶℶ\u202Cℶℶ") == "ℶℶℶℶ"
        assert Utils.strip_control_chars("\u202C") == ""


if __name__ == "__main__":
    pytest.main()
