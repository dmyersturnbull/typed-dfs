import numpy as np
import pytest
from pandas import Period

from typeddfs import TypedDfs
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
        txt = z.get_full_text()
        assert "pretty bird table" in txt
        assert "table of species" in txt
        assert "- species (string)" in txt
        assert "- prettiness (floating-point)" in txt
        assert "- cuteness (integer)" in txt
        assert "- lifespan (time period)" in txt
        assert "- .csv[.zip/.gz/.bz2/.xz]: Comma-delimited" in txt
        assert "[discouraged]" in txt
        assert "\n\n" in txt

    def test_matrix(self):
        clazz = (
            TypedDfs.matrix("pretty bird matrix")
            .doc("A table of species of pretty birds and whether they like each other.")
            .dtype(bool)
        ).build()
        z = DfCliHelp.help(clazz)
        txt = z.get_full_text()
        assert "whether they like each other" in txt
        assert "cast to bool" in txt


if __name__ == "__main__":
    pytest.main()
