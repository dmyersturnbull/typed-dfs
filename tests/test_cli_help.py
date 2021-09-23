import numpy as np
import pytest
from pandas import Period

from typeddfs import TypedDfs, FileFormat
from typeddfs.cli_help import DfCliHelp


class TestExample:
    def test_typed(self):
        clazz = (
            TypedDfs.typed("pretty bird table")
            .doc("A table of species of pretty birds and their characteristics.")
            .require("species", dtype=str)
            .require("prettiness", dtype=float)
            .require("cuteness", dtype=np.int16)
            .reserve("lifespan", dtype=Period)
        ).build()
        z = DfCliHelp.help(clazz)
        #
        # test long text
        #
        txt = z.get_long_text(nl="\n")
        lines = txt.splitlines()
        expected_lines = 1 + 1  # header and docstring
        expected_lines += 1 + 3  # required columns
        expected_lines += 1 + 1 + 1  # optional columns, inc. "any others allowed" (e.g.)
        expected_lines += 1 + len(FileFormat)  # file formats
        assert len(lines) == expected_lines, f"{len(lines)} != {expected_lines}: " + txt
        assert "pretty bird table" in lines[0]
        assert "table of species" in lines[1]
        assert "columns" in lines[2].lower()
        assert "- species (string)" in lines[3]
        assert "- prettiness (floating-point)" in txt
        assert "- cuteness (integer)" in txt
        assert "- lifespan (time period)" in txt
        assert "[avoid]" in txt or "[discouraged]" in txt or "[not recommended]" in txt
        #
        # test short text
        #
        txt = z.get_short_text(recommended_only=True, nl="\n")
        lines = txt.splitlines()
        expected_lines = 4  # 1 for header, 1 for doc, 1 for formats, 1 for typing
        assert len(lines) == expected_lines, f"{len(lines)} != {expected_lines}: {txt}"
        assert "pretty bird table" in lines[0]
        assert "table of species" in lines[1]
        assert "species (str)" in lines[2]
        assert "prettiness (float)" in lines[2]
        assert "lifespan (period)" in lines[2]
        assert "csv" in lines[3]

    def test_matrix(self):
        clazz = (
            TypedDfs.matrix("pretty bird matrix")
            .doc("A table of species of pretty birds and whether they like each other.")
            .dtype(bool)
        ).build()
        z = DfCliHelp.help(clazz)
        #
        # test long text
        #
        txt = z.get_long_text()
        lines = txt.splitlines()
        expected_lines = 3 + 1 + len(FileFormat)
        assert len(lines) == expected_lines, f"{len(lines)} != {expected_lines}: {txt}"
        assert "whether they like each other" in lines[1]
        assert "cast to bool" in lines[2]
        #
        # test short text
        #
        txt = z.get_short_text(recommended_only=True)
        lines = txt.splitlines()
        expected_lines = 4
        assert len(lines) == expected_lines, f"{len(lines)} != {expected_lines}: {txt}"
        assert "Boolean (bool)" in lines[2]


if __name__ == "__main__":
    pytest.main()
