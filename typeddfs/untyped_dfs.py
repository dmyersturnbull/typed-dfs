"""
Defines DataFrames with convenience methods but that do not enforce invariants.
"""
from __future__ import annotations

import os
from pathlib import PurePath
from typing import Optional, Union

import pandas as pd

from typeddfs.base_dfs import BaseDf

PathLike = Union[str, PurePath]


class UntypedDf(BaseDf):
    """
    A concrete DataFrame that does not require columns or enforce conditions.
    Overrides a number of DataFrame methods that preserve the subclass.
    For example, calling ``df.reset_index()`` will return a ``UntypedDf`` of the same type as ``df``.
    """


__all__ = ["UntypedDf"]
