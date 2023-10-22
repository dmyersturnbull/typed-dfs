# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
import sys

import pytest

from typeddfs.utils import TableFormat, Utils


class TestUtils:
    def test_encoding(self):
        assert Utils.get_encoding("platform") == sys.getdefaultencoding()
        assert "bom" not in Utils.get_encoding("utf8(bom)")
        assert "bom" not in Utils.get_encoding("utf16(bom)")
        assert Utils.get_encoding("UTF-8") == "utf8"
        assert Utils.get_encoding("utf-16") == "utf16"

    def test_basic(self):
        assert "sha1" in Utils.insecure_hash_functions()
        assert "__xml_index_" in Utils.banned_names()

    def test_strip_control_chars(self):
        assert Utils.strip_control_chars("ab\ncd") == "abcd"
        assert Utils.strip_control_chars("ab\0\0cℶd") == "abcℶd"
        assert Utils.strip_control_chars("ℶℶ\u202Cℶℶ") == "ℶℶℶℶ"
        assert Utils.strip_control_chars("\u202C") == ""

    def test_dots_and_dicts(self):
        dct = dict(abc=dict(xyz="123"), zzz=["456", "789"])
        dots = {"abc.xyz": "123", "zzz": ["456", "789"]}
        act_dots = Utils.dict_to_dots(dct)
        assert act_dots == dots
        act_dct = Utils.dots_to_dict(act_dots)
        assert act_dct == dct


if __name__ == "__main__":
    pytest.main()
