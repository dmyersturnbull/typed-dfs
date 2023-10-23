# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import typeddfs
from typeddfs import TypedDf, UntypedDf
from typeddfs.abs_dfs import AbsDf
from typeddfs.base_dfs import BaseDf

from . import sample_data


class TestCore:
    def test_wrap(self):
        df = pd.DataFrame({})
        df2 = typeddfs.wrap(df)
        assert not isinstance(df, AbsDf)
        assert isinstance(df2, BaseDf)

    def test_wrap_multilayer(self):
        # not fully supported yet, but let's check that it's reasonable
        rows = ["yes", "no", "maybe"]
        cols = [("animal", "cat"), ("animal", "armadillo", ("person", "matt"))]
        cols = pd.MultiIndex.from_tuples(cols)
        df = pd.DataFrame(np.zeros((3, 2)), rows, cols)
        df = typeddfs.wrap(df)
        assert df.column_names() == [
            ("animal", "cat", np.nan),
            ("animal", "armadillo", ("person", "matt")),
        ]

    def test_empty_simple(self):
        new = typeddfs.untyped("a class")
        df = new.convert(pd.DataFrame())
        assert list(df.columns) == []

    def test_no_name_simple(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            typeddfs.untyped(None)

    def test_no_name_fancy(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            typeddfs.typed(None).build()

    def test_simple(self):
        new = typeddfs.untyped("a class", doc="A doc")
        assert new.__name__ == "a class"
        assert new.__doc__ == "A doc"
        df = new.convert(pd.DataFrame(sample_data()))
        assert isinstance(df, UntypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy(self):
        new = typeddfs.typed("a class", doc="A doc").build()
        assert new.__name__ == "a class"
        assert new.__doc__ == "A doc"
        df = new.convert(pd.DataFrame(sample_data()))
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_index(self):
        new = typeddfs.typed("a class").require("abc", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc"]
        assert df.column_names() == ["123", "xyz"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_col(self):
        new = typeddfs.typed("a class").require("abc", index=False).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == []
        assert df.column_names() == ["abc", "123", "xyz"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_multiindex(self):
        new = typeddfs.typed("a class").require("abc", "xyz", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz"]
        assert df.column_names() == ["123"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_all_index(self):
        new = typeddfs.typed("a class").require("abc", "xyz", "123", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz", "123"]
        assert df.column_names() == []
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_no_index(self):
        new = typeddfs.typed("a class").require("abc", "123", "xyz", index=False).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == []
        assert df.column_names() == ["abc", "123", "xyz"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_extra_col(self):
        new = typeddfs.typed("a class").require("abc", index=True).strict().build()
        df = pd.DataFrame(sample_data())
        with pytest.raises(typeddfs.UnexpectedColumnError):
            new.convert(df)

    def test_extra_index(self):
        new = typeddfs.typed("a class").require("xyz", index=False).strict().build()
        df = pd.DataFrame(sample_data())
        with pytest.raises(typeddfs.UnexpectedColumnError):
            new.convert(df)

    def test_missing(self):
        new = typeddfs.typed("a class").require("qqq", index=False).strict().build()
        df = pd.DataFrame(sample_data())
        with pytest.raises(typeddfs.MissingColumnError):
            new.convert(df)


if __name__ == "__main__":
    pytest.main()
