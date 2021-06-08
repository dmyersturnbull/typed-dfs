import typing
from typing import Any, Mapping, Optional
from warnings import warn

import pandas as pd

try:
    import pyarrow
except ImportError:
    pyarrow = None

try:
    import fastparquet
except ImportError:
    fastparquet = None

try:
    import tables
except ImportError:
    tables = None

try:
    import tabulate

    try:
        import wcwidth
    except ImportError:
        warn("wcwidth is not installed")
except ImportError:
    tabulate = None


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
                    dict(sep="\t", **get("nl", "header", "comment", "skip_blank_lines")),
                )
            dct[".csv" + compression] = ("csv", get("nl", "header", "comment", "skip_blank_lines"))
            if _Utils.has_tabulate:
                dct[".flexwf" + compression] = ("flexwf", get("sep", "comment"))
                # dct[".fwf" + compression] = ("flexwf", get("sep"))
        prefix = "read_" if writing is False else "to_"
        return {k: (prefix + fn, p) for k, (fn, p) in dct.items()}


__all__ = ["_Utils"]
