import inspect
import sys

import numpy as np
import pytest
from natsort import ns

from typeddfs.utils import TableFormat, Utils


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

    def test_exact_natsort_alg_numeric(self):
        names, z = Utils.exact_natsort_alg({"FLOAT", "SIGNED"})
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = Utils.exact_natsort_alg(ns.REAL)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = Utils.exact_natsort_alg({"REAL"})
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = Utils.exact_natsort_alg("REAL")
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = Utils.exact_natsort_alg({"INT"})
        assert names == set()
        assert z == 0
        names, z = Utils.exact_natsort_alg(0)
        assert names == set()
        assert z == 0

    def test_exact_natsort_alg_str(self):
        names, z = Utils.exact_natsort_alg("ignorecase")
        assert names == {"IGNORECASE"}
        assert z == ns.IGNORECASE

    def test_guess_natsort_alg_str(self):
        names, z = Utils.guess_natsort_alg(str)
        assert names == {"COMPATIBILITYNORMALIZE", "GROUPLETTERS"}
        assert z == ns.COMPATIBILITYNORMALIZE | ns.GROUPLETTERS

    def test_guess_natsort_alg_int(self):
        names, z = Utils.guess_natsort_alg(int)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED
        names, z = Utils.guess_natsort_alg(np.int32)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED
        names, z = Utils.guess_natsort_alg(bool)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED
        names, z = Utils.guess_natsort_alg(np.bool_)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED

    def test_guess_natsort_alg_float(self):
        names, z = Utils.guess_natsort_alg(float)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.FLOAT | ns.SIGNED
        names, z = Utils.guess_natsort_alg(np.float16)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.FLOAT | ns.SIGNED
        names, z = Utils.guess_natsort_alg(np.float32)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.FLOAT | ns.SIGNED

    def test_orjson_preserve_inf(self):
        matrix = np.zeros((2, 2))
        assert (Utils.orjson_preserve_inf(matrix) == matrix.astype(str)).all()
        matrix = np.asarray([[2, float("inf")], [float("inf"), 2]])
        assert (Utils.orjson_preserve_inf(matrix) == matrix.astype(str)).all()
        # TODO: nested tests

    def test_orjson_new_default(self):
        class X:
            def __str__(self):
                return "from-str"

            def __repr__(self):
                return "from-repr"

        default = Utils.orjson_new_default()
        x = default(None)
        assert x is None
        assert default(X()) == "from-str"
        default = Utils.orjson_new_default(use_repr=True)
        assert default(X()) == "from-repr"

        def fixer(obj):
            if isinstance(obj, X):
                return "gotcha!"

        default = Utils.orjson_new_default(fixer)
        assert default(X()) == "gotcha!"

    def test_to_json(self):
        assert Utils.orjson_str("hi") == '"hi"\n'
        assert Utils.orjson_str(["hi", "bye"]) == '[\n  "hi",\n  "bye"\n]\n'
        data = {
            "list": [
                {
                    "numbers": {
                        1: np.asarray([float("inf"), 0]),
                        2: np.asarray([1, 1]),
                        3: np.half(float("inf")),
                        4: np.half(float("-inf")),
                        5: float("inf"),
                        6: float("-inf"),
                        7: 1,
                    }
                }
            ]
        }
        x = Utils.orjson_str(data)
        assert (
            x
            == inspect.cleandoc(
                """
            {
              "list": [
                {
                  "numbers": {
                    "1": [
                      "inf",
                      "0.0"
                    ],
                    "2": [
                      "1",
                      "1"
                    ],
                    "3": "inf",
                    "4": "-inf",
                    "5": "inf",
                    "6": "-inf",
                    "7": 1
                  }
                }
              ]
            }
            """
            )
            + "\n"
        )


if __name__ == "__main__":
    pytest.main()
