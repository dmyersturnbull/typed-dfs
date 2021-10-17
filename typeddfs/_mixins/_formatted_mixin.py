"""
Mixin for formats like HTML and RST.
"""
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

# noinspection PyProtectedMember
from tabulate import TableFormat, tabulate

from typeddfs.df_errors import NoValueError, ValueNotUniqueError
from typeddfs.utils import Utils
from typeddfs.utils._utils import PathLike


class _FormattedMixin:
    def to_html(self, *args, **kwargs) -> Optional[str]:
        df = self.vanilla_reset()
        return df.to_html(*args, **kwargs)

    def to_rst(
        self, path_or_none: Optional[PathLike] = None, style: str = "simple", mode: str = "w"
    ) -> Optional[str]:
        """
        Writes a reStructuredText table.
        Args:
            path_or_none: Either a file path or ``None`` to return the string
            style: The type of table; currently only "simple" is supported
            mode: Write mode
        """
        txt = self._tabulate(fmt="rst") + "\n"
        return Utils.write(path_or_none, txt, mode=mode)

    def to_markdown(self, *args, **kwargs) -> Optional[str]:
        return super().to_markdown(*args, **kwargs)

    @classmethod
    def read_html(cls, path: PathLike, *args, **kwargs) -> __qualname__:
        """
        Similar to ``pd.read_html``, but requires exactly 1 table and returns it.

        Raises:
            lxml.etree.XMLSyntaxError: If the HTML could not be parsed
            NoValueError: If no tables are found
            ValueNotUniqueError: If multiple tables are found
        """
        try:
            dfs = pd.read_html(path, *args, **kwargs)
        except ValueError as e:
            if str(e) == "No tables found":
                raise NoValueError(f"No tables in {path}") from None
            raise
        if len(dfs) > 1:
            raise ValueNotUniqueError(f"{len(dfs)} tables in {path}")
        df = dfs[0]
        if "Unnamed: 0" in df:
            df = df.drop("Unnamed: 0", axis=1)
        return cls._convert_typed(df)

    def _tabulate(self, fmt: Union[str, TableFormat], **kwargs) -> str:
        df = self.vanilla_reset()
        return tabulate(df.to_numpy().tolist(), list(df.columns), tablefmt=fmt, **kwargs)


__all__ = ["_FormattedMixin"]
