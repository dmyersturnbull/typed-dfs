import numpy as np
import pytest
from pandas import Period

from typeddfs import TypedDfs
from typeddfs.cli_help import DfCliHelp


class TestExample:
    def test(self):
        clazz = (
            TypedDfs.typed("pretty bird table")
            .doc("A table species of pretty birds and their characteristics.")
            .require("species", dtype=str)
            .require("prettiness", dtype=float)
            .require("cuteness", dtype=np.int16)
            .reserve("lifespan", dtype=Period)
        ).build()
        z = DfCliHelp.help(clazz)
        txt = z.get_full_text()
        print("\n" + txt)
        pass


if __name__ == "__main__":
    pytest.main()
