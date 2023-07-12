# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
import contextlib
import io

import pytest

from typeddfs.example import run


class TestExample:
    def test(self):
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            run()
        cap = cap.getvalue().strip()
        assert cap == "['key'] ['value', 'note']\n123"


if __name__ == "__main__":
    pytest.main()
