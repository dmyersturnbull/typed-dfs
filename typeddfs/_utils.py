"""
Internal utilities for typeddfs.
"""

import typing
from pathlib import PurePath
from typing import Any, Mapping, Union
from warnings import warn

import pandas as pd

try:
    import pyarrow
except ImportError:  # pragma: no cover
    pyarrow = None

try:
    import fastparquet
except ImportError:  # pragma: no cover
    fastparquet = None

try:
    import tables
except ImportError:  # pragma: no cover
    tables = None

try:
    import tabulate

    try:
        import wcwidth  # noqa: F401
    except ImportError:
        warn("wcwidth is not installed")
except ImportError:  # pragma: no cover
    tabulate = None


_FAKE_SEP = "\u2008"  # 6-em space; very unlikely to occur
PathLike = Union[str, PurePath]


class _Sentinal:
    pass


_SENTINAL = _Sentinal()


class _Utils:

    has_feather = pyarrow is not None
    has_parquet = fastparquet is not None or pyarrow is not None
    has_hdf5 = tables is not None
    has_tabulate = tabulate is not None

    @classmethod
    def guess_io(
        cls, writing: typing.Union[bool, pd.DataFrame], inc_lines: bool, params: Mapping[str, Any]
    ) -> Mapping[str, typing.Tuple[str, Mapping[str, Any]]]:
        dct = {
            ".feather": ("feather", {}),
            ".parquet": ("parquet", {}),
            ".snappy": ("parquet", {}),
            ".h5": ("hdf", {}),
            ".hdf": ("hdf", {}),
            ".xlsx": ("excel", {}),
            ".xls": ("excel", {}),
        }

        def get(*qq):
            return {q: params.get(q) for q in qq if q in params}

        if _Utils.has_parquet:
            dct.update({x: ("parquet", {}) for x in [".snappy", ".parquet"]})
        if _Utils.has_feather and (isinstance(writing, bool) or len(writing.column_names()) > 0):
            dct[".feather"] = ("feather", {})
        if _Utils.has_hdf5:
            dct.update({x: ("hdf", {}) for x in [".h5", ".hdf", ".hdf5"]})
        if writing is False:
            dct[".fwf"] = ("fwf", {})
        for compression in {".gz", ".zip", ".bz2", ".xz", ""}:
            dct[".json" + compression] = ("json", {})
            if inc_lines and (isinstance(writing, bool) or len(writing.columns) == 1):
                for x in [".txt", ".lines", ".list"]:
                    dct[x + compression] = ("lines", get("comment", "nl"))
            for x in [".tab", ".tsv"]:
                dct[x + compression] = (
                    "csv",
                    dict(sep="\t", **get("nl", "comment", "skip_blank_lines")),
                )
            dct[".csv" + compression] = ("csv", get("nl", "comment", "skip_blank_lines"))
            if _Utils.has_tabulate:
                dct[".flexwf" + compression] = ("flexwf", get("sep", "comment"))
                # dct[".fwf" + compression] = ("flexwf", get("sep"))
        prefix = "read_" if writing is False else "to_"
        return {k: (prefix + fn, p) for k, (fn, p) in dct.items()}


__all__ = ["_Utils", "_FAKE_SEP", "PathLike", "_SENTINAL"]
