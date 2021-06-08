import numpy as np
import pandas as pd
import pytest

from typeddfs.base_dfs import PrettyDf
from typeddfs.builders import TypedDfBuilder

# noinspection PyProtectedMember
from typeddfs.typed_dfs import (
    ExtraConditionFailedError,
    TypedDf,
    UnexpectedColumnError,
    UnexpectedIndexNameError,
)


def always_ok(x):
    return None


def always_fail(x):
    return "OH NO"


class TestBuilders:
    def test_symmetric_and_condition(self):
        t = TypedDfBuilder("a").symmetric().condition(always_ok).build()
        assert t.required_columns() == []
        assert t.required_index_names() == []
        assert t.must_be_symmetric()
        assert t.extra_conditions() == [always_ok]
        TypedDf(pd.DataFrame())
        t = TypedDfBuilder("a").symmetric().condition(always_fail).build()
        with pytest.raises(ExtraConditionFailedError):
            t.convert(pd.DataFrame())

    def test_require_and_reserve_col(self):
        t = TypedDfBuilder("a").require("column").reserve("reserved").build()
        assert t.required_columns() == ["column"]
        assert t.reserved_columns() == ["reserved"]
        assert t.required_index_names() == []
        assert t.reserved_index_names() == []
        assert not t.must_be_symmetric()
        assert t.extra_conditions() == []

    def test_require_and_reserve_index(self):
        t = (
            TypedDfBuilder("a")
            .require("column", index=True)
            .reserve("reserved", index=True)
            .build()
        )
        assert t.required_columns() == []
        assert t.reserved_columns() == []
        assert t.required_index_names() == ["column"]
        assert t.reserved_index_names() == ["reserved"]
        assert not t.must_be_symmetric()
        assert t.extra_conditions() == []

    def test_drop(self):
        t = TypedDfBuilder("a").reserve("column").drop("trash").build()
        assert t.columns_to_drop() == ["trash"]
        df = t.convert(pd.DataFrame([pd.Series(dict(x="x", zz="y"))]))
        assert df.column_names() == ["x", "zz"]
        df = t.convert(pd.DataFrame([pd.Series(dict(x="x", trash="y"))]))
        assert df.column_names() == ["x"]

    def test_bad_type(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            TypedDfBuilder(None).build()
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            TypedDfBuilder(5).build()

    def test_bad_symmetry(self):
        with pytest.raises(ValueError):
            TypedDfBuilder("a").require("x", "y", index=True).symmetric().build()

    def test_bad_require(self):
        for index in [True, False]:
            with pytest.raises(ValueError):
                TypedDfBuilder("a").require("level_0", index=index)
            with pytest.raises(ValueError):
                TypedDfBuilder("a").require("abc", "level_0", index=index)
            with pytest.raises(ValueError):
                TypedDfBuilder("a").require("abc", "index", index=index)

    def test_bad_reserve(self):
        for index in [True, False]:
            with pytest.raises(ValueError):
                TypedDfBuilder("a").reserve("level_0", index=index)
            with pytest.raises(ValueError):
                TypedDfBuilder("a").reserve("abc", "level_0", index=index)
            with pytest.raises(ValueError):
                TypedDfBuilder("a").reserve("abc", "index", index=index)

    def test_already_added(self):
        for cola in [True, False]:
            for indexa in [True, False]:
                for colb in [True, False]:
                    for indexb in [True, False]:
                        builder = TypedDfBuilder("a")
                        builder = (
                            builder.require("a", index=indexa)
                            if cola
                            else builder.reserve("a", index=indexa)
                        )
                        with pytest.raises(ValueError):
                            builder.require("a", index=indexb) if colb else builder.reserve(
                                "a", index=indexb
                            )

    def test_strict(self):
        # strict columns but not index
        t = TypedDfBuilder("a").strict(False, True).build()
        assert not t.more_columns_allowed()
        assert t.more_indices_allowed()
        t.convert(pd.DataFrame([pd.Series(dict(x="x"))]).set_index("x"))
        with pytest.raises(UnexpectedColumnError):
            t.convert(pd.DataFrame([pd.Series(dict(x="x"))]))
        # strict index but not columns
        t = TypedDfBuilder("a").strict(True, False).build()
        assert t.more_columns_allowed()
        assert not t.more_indices_allowed()
        t.convert(pd.DataFrame([pd.Series(dict(x="x"))]))
        with pytest.raises(UnexpectedIndexNameError):
            df = PrettyDf(pd.DataFrame([pd.Series(dict(x="x"))]).set_index("x"))
            assert df.index_names() == ["x"]
            assert df.column_names() == []
            t.convert(df)
        # neither strict
        t = TypedDfBuilder("a").strict(False, False).build()
        t.convert(pd.DataFrame([pd.Series(dict(x="x"))]))

    """
    def test_reserve_dtype(self):
        t = TypedDfBuilder("a").reserve("x", dtype=np.float32).build()
        t = t.convert(pd.DataFrame([pd.Series(dict(x="0.5"))]))
        assert t.column_names() == ["x"]
        assert t.values.tolist() == [[0.5]]
    """


if __name__ == "__main__":
    pytest.main()
