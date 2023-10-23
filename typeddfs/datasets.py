# SPDX-FileCopyrightText: Copyright 2020-2023, Contributors to typed-dfs
# SPDX-PackageHomePage: https://github.com/dmyersturnbull/typed-dfs
# SPDX-License-Identifier: Apache-2.0
"""
Near-replica of example from the readme.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from typeddfs.abs_dfs import AbsDf
from typeddfs.builders import TypedDfBuilder
from typeddfs.file_formats import FileFormat
from typeddfs.typed_dfs import PlainTypedDf

T_co = TypeVar("T_co", covariant=True, bound=AbsDf)
S_co = TypeVar("S_co", covariant=True, bound=AbsDf)
X_co = TypeVar("X_co", covariant=True, bound=AbsDf)


class LazyDf(Generic[T_co]):
    """
    A :class:`typeddfs.abs_dfs.AbsDf` that is lazily loaded from a source.
    Create normally via :meth:`from_source`.
    Create with :meth:`from_df` to wrap an extant DataFrame into a LazyDataFrame.

    Example:
        .. code-block::

            lazy = LazyDataFrame.from_source("https://google.com/dataframe.csv")
    """

    def __init__(self, name: str, source: str, clazz: type[T_co], _df: T_co | None) -> None:
        self._name = name
        self._source = source
        self._clazz: type[T_co] = clazz
        self._df: T_co | None = _df

    @classmethod
    def from_source(
        cls,
        source: str,
        clazz: type[S_co] = PlainTypedDf,
        name: str | None = None,
    ) -> LazyDf[S_co]:
        p, _, _ = FileFormat.split(source)
        if name is None:
            name = Path(p).name
        return LazyDf(name, source, clazz, None)

    @classmethod
    def from_df(cls, df: X_co, name: str | None = None) -> LazyDf[X_co]:
        if name is None:
            name = df.__class__.__name__
        return LazyDf(name, "", df.__class__, df)

    @property
    def name(self) -> str:
        return self._name

    @property
    def clazz(self) -> type[T_co]:
        return self._clazz

    @property
    def df(self) -> T_co:
        if self._df is None and self._source.startswith("https://"):
            self._df = self._clazz.read_url(self._source)
        elif self._df is None:
            self._df = self._clazz.read_file(self._source)
        return self._df


def _get(name: str, t: type[AbsDf] = PlainTypedDf) -> LazyDf:
    url = f"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv"
    if t is None:
        p, _, _ = FileFormat.split(url)
        t = TypedDfBuilder(p.name).build()
    return LazyDf.from_source(url, t)


class ExampleDfs:  # pragma: no cover
    """
    DataFrames derived from Seaborn and other sources.
    """

    anagrams = _get("anagrams")
    anscombe = _get("anscombe")
    attention = _get("attention")
    brain_networks = _get("brain_networks")
    car_crashes = _get("car_crashes")
    diamonds = _get("diamonds")
    dots = _get("dots")
    exercise = _get("exercise")
    flights = _get("flights")
    fmri = _get("fmri")
    gammas = _get("gammas")
    geyser = _get("geyser")
    iris = _get("iris")
    mpg = _get("mpg")
    penguins = _get("penguins")
    planets = _get("planets")
    taxis = _get("taxis")
    tips = _get("tips")
    titanic = _get("titanic")


__all__ = ["LazyDf", "ExampleDfs"]


if __name__ == "__main__":  # pragma: no cover
    dfx = ExampleDfs.penguins
    print(dfx.name)
    print(dfx.df)
