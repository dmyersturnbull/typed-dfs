# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from natsort import ns

from typeddfs.utils.sort_utils import SortUtils


class TestSortUtils:
    def test_exact_natsort_alg_numeric(self):
        names, z = SortUtils.exact_natsort_alg({"FLOAT", "SIGNED"})
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = SortUtils.exact_natsort_alg(ns.REAL)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = SortUtils.exact_natsort_alg({"REAL"})
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = SortUtils.exact_natsort_alg("REAL")
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.REAL
        names, z = SortUtils.exact_natsort_alg({"INT"})
        assert names == set()
        assert z == 0
        names, z = SortUtils.exact_natsort_alg(0)
        assert names == set()
        assert z == 0

    def test_exact_natsort_alg_str(self):
        names, z = SortUtils.exact_natsort_alg("ignorecase")
        assert names == {"IGNORECASE"}
        assert z == ns.IGNORECASE

    def test_guess_natsort_alg_str(self):
        names, z = SortUtils.guess_natsort_alg(str)
        assert names == {"COMPATIBILITYNORMALIZE", "GROUPLETTERS"}
        assert z == ns.COMPATIBILITYNORMALIZE | ns.GROUPLETTERS

    def test_guess_natsort_alg_int(self):
        names, z = SortUtils.guess_natsort_alg(int)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED
        names, z = SortUtils.guess_natsort_alg(np.int32)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED
        names, z = SortUtils.guess_natsort_alg(bool)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED
        names, z = SortUtils.guess_natsort_alg(np.bool_)
        assert names == {"INT", "SIGNED"}
        assert z == ns.INT | ns.SIGNED

    def test_guess_natsort_alg_float(self):
        names, z = SortUtils.guess_natsort_alg(float)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.FLOAT | ns.SIGNED
        names, z = SortUtils.guess_natsort_alg(np.float16)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.FLOAT | ns.SIGNED
        names, z = SortUtils.guess_natsort_alg(np.float32)
        assert names == {"FLOAT", "SIGNED"}
        assert z == ns.FLOAT | ns.SIGNED


if __name__ == "__main__":
    pytest.main()
