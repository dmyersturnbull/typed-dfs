import pytest
import pandas as pd
from pathlib import Path
from typing import Sequence
import inspect
from typeddfs.typed_dfs import *

raises = pytest.raises


def tmpfile() -> Path:
    caller = inspect.stack()[1][3]
    path = (
        Path(__file__).parent.parent.parent
        / "resources"
        / "tmp"
        / (str(caller) + ".csv")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SimpleOrg(OrganizingFrame):
    pass


class SingleIndexOrg(OrganizingFrame):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc"]


class MultiIndexOrg(OrganizingFrame):
    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ["abc", "xyz"]


class TestCore:
    def test_pretty(self):
        assert (
            PrettyFrame()
            ._repr_html_()
            .startswith("<strong>PrettyFrame: 0 rows × 0 columns</strong>")
        )

    def test_simpleframe_read_write_csv(self):
        path = tmpfile()
        for indices in [None, "abc", ["abc", "xyz"]]:
            df = SimpleFrame(
                [
                    pd.Series({"abc": 1, "123": 2, "xyz": 3}),
                    pd.Series({"abc": 4, "123": 5, "xyz": 6}),
                ]
            )
            if indices is not None:
                df = df.set_index(indices)
            df.to_csv(path)
            df2 = SimpleFrame.read_csv(path)
            assert list(df2.index.names) == [None]
            assert set(df2.columns) == {"abc", "123", "xyz"}
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_noindex(self):
        path = tmpfile()
        df = SimpleOrg(
            [
                pd.Series({"abc": 1, "123": 2, "xyz": 3}),
                pd.Series({"abc": 4, "123": 5, "xyz": 6}),
            ]
        )
        df.to_csv(path)
        df2 = SimpleOrg.read_csv(path)
        assert list(df2.index.names) == [None]
        assert set(df2.columns) == {"abc", "123", "xyz"}
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_singleindex(self):
        path = tmpfile()
        df = SingleIndexOrg.convert(
            SingleIndexOrg(
                [
                    pd.Series({"abc": 1, "123": 2, "xyz": 3}),
                    pd.Series({"abc": 4, "123": 5, "xyz": 6}),
                ]
            )
        )
        df.to_csv(path)
        assert list(df.index.names) == ["abc"]
        assert set(df.columns) == {"123", "xyz"}
        df2 = SingleIndexOrg.read_csv(path)
        assert list(df2.index.names) == ["abc"]
        assert set(df2.columns) == {"123", "xyz"}
        if path.exists():
            path.unlink()

    def test_organizingframe_read_write_csv_multiindex(self):
        path = tmpfile()
        df = MultiIndexOrg.convert(
            MultiIndexOrg(
                [
                    pd.Series({"abc": 1, "123": 2, "xyz": 3}),
                    pd.Series({"abc": 4, "123": 5, "xyz": 6}),
                ]
            )
        )
        df.to_csv(path)
        assert list(df.index.names) == ["abc", "xyz"]
        assert set(df.columns) == {"123"}
        df2 = MultiIndexOrg.read_csv(path)
        assert list(df2.index.names) == ["abc", "xyz"]
        assert set(df2.columns) == {"123"}
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    pytest.main()
