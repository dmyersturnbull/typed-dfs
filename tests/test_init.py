import pytest
import pandas as pd
from typeddfs import (
    TypedDfs,
    SimpleFrame,
    OrganizingFrame,
    UnexpectedColumnError,
    MissingColumnError,
)
from . import sample_data

raises = pytest.raises


class TestCore:
    def test_empty_simple(self):
        new = TypedDfs.simple("a class")
        df = new.convert(pd.DataFrame())
        assert list(df.columns) == []

    def test_no_name_simple(self):
        with raises(TypeError):
            TypedDfs.simple(None)

    def test_no_name_fancy(self):
        with raises(TypeError):
            TypedDfs.fancy(None).build()

    def test_simple(self):
        new = TypedDfs.simple("a class", doc="A doc")
        assert new.__name__ == "a class"
        assert new.__doc__ == "A doc"
        df = new.convert(pd.DataFrame(sample_data()))
        assert isinstance(df, SimpleFrame)
        assert df.__class__.__name__ == "a class"

    def test_fancy(self):
        new = TypedDfs.fancy("a class", doc="A doc").build()
        assert new.__name__ == "a class"
        assert new.__doc__ == "A doc"
        df = new.convert(pd.DataFrame(sample_data()))
        assert isinstance(df, OrganizingFrame)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_index(self):
        new = TypedDfs.fancy("a class").require("abc", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc"]
        assert df.column_names() == ["123", "xyz"]
        assert isinstance(df, OrganizingFrame)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_col(self):
        new = TypedDfs.fancy("a class").require("abc", index=False).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == []
        assert df.column_names() == ["abc", "123", "xyz"]
        assert isinstance(df, OrganizingFrame)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_multiindex(self):
        new = TypedDfs.fancy("a class").require("abc", "xyz", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz"]
        assert df.column_names() == ["123"]
        assert isinstance(df, OrganizingFrame)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_all_index(self):
        new = TypedDfs.fancy("a class").require("abc", "xyz", "123", index=True).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == ["abc", "xyz", "123"]
        assert df.column_names() == []
        assert isinstance(df, OrganizingFrame)
        assert df.__class__.__name__ == "a class"

    def test_fancy_with_no_index(self):
        new = TypedDfs.fancy("a class").require("abc", "123", "xyz", index=False).build()
        df = new.convert(pd.DataFrame(sample_data()))
        assert df.index_names() == []
        assert df.column_names() == ["abc", "123", "xyz"]
        assert isinstance(df, OrganizingFrame)
        assert df.__class__.__name__ == "a class"

    def test_extra_col(self):
        new = TypedDfs.fancy("a class").require("abc", index=True).strict().build()
        df = pd.DataFrame(sample_data())
        with raises(UnexpectedColumnError):
            new.convert(df)

    def test_extra_index(self):
        new = TypedDfs.fancy("a class").require("xyz", index=False).strict().build()
        df = pd.DataFrame(sample_data())
        with raises(UnexpectedColumnError):
            new.convert(df)

    def test_missing(self):
        new = TypedDfs.fancy("a class").require("qqq", index=False).strict().build()
        df = pd.DataFrame(sample_data())
        with raises(MissingColumnError):
            new.convert(df)


if __name__ == "__main__":
    pytest.main()
