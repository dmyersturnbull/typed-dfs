import numpy as np
import pandas as pd
import pytest

# noinspection PyProtectedMember
from typeddfs._pretty_dfs import PrettyDf
from typeddfs.base_dfs import BaseDf
from typeddfs.builders import MatrixDfBuilder, TypedDfBuilder
from typeddfs.df_errors import (
    ClashError,
    DfTypeConstructionError,
    UnexpectedColumnError,
    UnexpectedIndexNameError,
    VerificationFailedError,
)
from typeddfs.df_typing import DfTyping
from typeddfs.typed_dfs import TypedDf


def always_ok(x):
    return None


def always_fail(x):
    return "OH NO"


class TestBuilders:
    def test_typed_subclass(self):
        t1 = TypedDfBuilder("t1").build()
        t2 = TypedDfBuilder("t2").subclass(t1).build()
        assert issubclass(t2, t1)
        assert not issubclass(t1, t2)

    def test_matrix_subclass(self):
        t1 = MatrixDfBuilder("t1").build()
        t2 = MatrixDfBuilder("t2").subclass(t1).build()
        assert issubclass(t2, t1)
        assert not issubclass(t1, t2)

    def test_condition(self):
        t = TypedDfBuilder("a").verify(always_ok).build()
        typ: DfTyping = t.get_typing()
        assert typ.required_columns == []
        assert typ.required_index_names == []
        assert typ.verifications == [always_ok]
        TypedDf(pd.DataFrame())
        t = TypedDfBuilder("a").verify(always_fail).build()
        with pytest.raises(VerificationFailedError):
            t.convert(pd.DataFrame())

    def test_require_and_reserve_col(self):
        t = TypedDfBuilder("a").require("column").reserve("reserved").build()
        typ: DfTyping = t.get_typing()
        assert typ.required_columns == ["column"]
        assert typ.reserved_columns == ["reserved"]
        assert typ.required_index_names == []
        assert typ.reserved_index_names == []
        assert typ.verifications == []

    def test_require_and_reserve_index(self):
        t = (
            TypedDfBuilder("a").require("column", index=True).reserve("reserved", index=True)
        ).build()
        typ: DfTyping = t.get_typing()
        assert typ.required_columns == []
        assert typ.reserved_columns == []
        assert typ.required_index_names == ["column"]
        assert typ.reserved_index_names == ["reserved"]
        assert typ.known_index_names == ["column", "reserved"]
        assert typ.known_column_names == []
        assert typ.known_names == ["column", "reserved"]
        assert typ.verifications == []

    def test_drop(self):
        t = TypedDfBuilder("a").reserve("column").drop("trash").build()
        typ: DfTyping = t.get_typing()
        assert typ.columns_to_drop == {"trash"}
        df = t.convert(pd.DataFrame([pd.Series(dict(x="x", zz="y"))]))
        assert df.column_names() == ["x", "zz"]
        df = t.convert(pd.DataFrame([pd.Series(dict(x="x", trash="y"))]))
        assert df.column_names() == ["x"]

    def test_drop_clash(self):
        t = TypedDfBuilder("a").reserve("trash").drop("trash")
        with pytest.raises(ClashError):
            t.build()

    def test_secure(self):
        TypedDfBuilder("a").secure().hash(alg="sha256").build()
        TypedDfBuilder("a").hash(alg="sha1").build()
        with pytest.raises(DfTypeConstructionError):
            TypedDfBuilder("a").secure().hash(alg="sha1").build()

    def test_bad_type(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            TypedDfBuilder(None).build()
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            TypedDfBuilder(5).build()

    def test_bad_require(self):
        for index in [True, False]:
            with pytest.raises(ClashError):
                TypedDfBuilder("a").require("level_0", index=index)
            with pytest.raises(ClashError):
                TypedDfBuilder("a").require("abc", "level_0", index=index)
            with pytest.raises(ClashError):
                TypedDfBuilder("a").require("abc", "index", index=index)

    def test_bad_reserve(self):
        for index in [True, False]:
            with pytest.raises(ClashError):
                TypedDfBuilder("a").reserve("level_0", index=index)
            with pytest.raises(ClashError):
                TypedDfBuilder("a").reserve("abc", "level_0", index=index)
            with pytest.raises(ClashError):
                TypedDfBuilder("a").reserve("abc", "index", index=index)

    def test_already_added(self):
        for cola in [True, False]:
            for indexa in [True, False]:
                for colb in [True, False]:
                    for indexb in [True, False]:
                        builder = TypedDfBuilder("a")
                        if cola:
                            builder = builder.require("a", index=indexa)
                        else:
                            cola = builder.reserve("a", index=indexa)
                        with pytest.raises(ClashError):
                            if colb:
                                builder.require("a", index=indexb)
                            else:
                                builder.reserve("a", index=indexb)

    def test_strict(self):
        # strict columns but not index
        t = TypedDfBuilder("a").strict(index=False, cols=True).build()
        typ: DfTyping = t.get_typing()
        assert typ.more_indices_allowed
        assert not typ.more_columns_allowed
        t.convert(pd.DataFrame([pd.Series(dict(x="x"))]).set_index("x"))
        with pytest.raises(UnexpectedColumnError):
            t.convert(pd.DataFrame([pd.Series(dict(x="x"))]))
        # strict index but not columns
        t = TypedDfBuilder("a").strict(True, False).build()
        typ: DfTyping = t.get_typing()
        assert typ.more_columns_allowed
        assert not typ.more_indices_allowed
        t.convert(pd.DataFrame([pd.Series(dict(x="x"))]))
        with pytest.raises(UnexpectedIndexNameError):
            df = PrettyDf(pd.DataFrame([pd.Series(dict(x="x"))]).set_index("x"))
            assert df.index_names() == ["x"]
            assert df.column_names() == []
            t.convert(df)
        # neither strict
        t = TypedDfBuilder("a").strict(False, False).build()
        t.convert(pd.DataFrame([pd.Series(dict(x="x"))]))

    def test_reserve_dtype(self):
        t = TypedDfBuilder("a").reserve("x", dtype=np.float32).build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="0.5"))]))
        assert df.column_names() == ["x"]
        assert df.to_numpy().tolist() == [[0.5]]
        with pytest.raises(ValueError):
            t.convert(pd.DataFrame([pd.Series(dict(x="kitten"))]))

    def test_dtype_post_process(self):
        # make sure these happen in the right order:
        # 1. dtype conversions
        # 2. post-processing
        # 3. final conditions

        def post(dd: BaseDf) -> BaseDf:
            assert dd["x"].dtype == np.float32
            dd2 = dd.copy()
            dd2["x"] += 9
            return dd2

        def cond(dd: BaseDf):
            return None if dd["x"].dtype == np.float32 else "failed"

        t = (TypedDfBuilder("a").reserve("x", dtype=np.float32).post(post).verify(cond)).build()
        df = t.convert(pd.DataFrame([pd.Series(dict(x="0.5"))]))
        assert df.to_numpy().tolist() == [[9.5]]


if __name__ == "__main__":
    pytest.main()
