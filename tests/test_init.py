import pandas as pd
import pytest

from typeddfs import TypedDf, UntypedDf
from typeddfs import TypedDfs

from typeddfs.base_dfs import BaseDf
from typeddfs.abs_df import AbsDf

from . import sample_data


class TestCore:
    def test_wrap(self):
        df = pd.DataFrame({})
        df2 = TypedDfs.wrap(df)
        assert not isinstance(df, AbsDf)
        assert isinstance(df2, BaseDf)

    def test_empty_simple(self):
        new = TypedDfs.untyped("a class")
        df = new.convert(pd.DataFrame())
        assert list(df.columns) == []

    def test_no_name_simple(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            TypedDfs.untyped(None)

    def test_no_name_fancy(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            TypedDfs.typed(None).build()

    def test_simple(self):
        new = TypedDfs.untyped("a class", doc="A doc")
        assert new.__name__ == "a class"
        assert new.__doc__ == "A doc"
        df = new.convert(pd.DataFrame(sample_data()))
        assert isinstance(df, UntypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy(self):
        new = TypedDfs.typed("a class", doc="A doc").build()
        assert new.__name__ == "a class"
        assert new.__doc__ == "A doc"
        df = new.convert(pd.DataFrame(sample_data()))
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_index(self):
        new = TypedDfs.typed("a class").require("abc", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc"]
        assert df.column_names() == ["123", "xyz"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_col(self):
        new = TypedDfs.typed("a class").require("abc", index=False).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == []
        assert df.column_names() == ["abc", "123", "xyz"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_multiindex(self):
        new = TypedDfs.typed("a class").require("abc", "xyz", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz"]
        assert df.column_names() == ["123"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_all_index(self):
        new = TypedDfs.typed("a class").require("abc", "xyz", "123", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz", "123"]
        assert df.column_names() == []
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_no_index(self):
        new = TypedDfs.typed("a class").require("abc", "123", "xyz", index=False).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == []
        assert df.column_names() == ["abc", "123", "xyz"]
        assert isinstance(df, TypedDf)
        assert df.__class__.__name__ == "a class"

    def test_extra_col(self):
        new = TypedDfs.typed("a class").require("abc", index=True).strict().build()
        df = pd.DataFrame(sample_data())
        with pytest.raises(TypedDfs.UnexpectedColumnError):
            new.convert(df)

    def test_extra_index(self):
        new = TypedDfs.typed("a class").require("xyz", index=False).strict().build()
        df = pd.DataFrame(sample_data())
        with pytest.raises(TypedDfs.UnexpectedColumnError):
            new.convert(df)

    def test_missing(self):
        new = TypedDfs.typed("a class").require("qqq", index=False).strict().build()
        df = pd.DataFrame(sample_data())
        with pytest.raises(TypedDfs.MissingColumnError):
            new.convert(df)


if __name__ == "__main__":
    pytest.main()
