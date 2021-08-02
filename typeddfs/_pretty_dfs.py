"""
Defines a DataFrame with simple extra functions like ``column_names``.
"""
import abc
from typing import List

import pandas as pd
from pandas.core.frame import DataFrame as _InternalDataFrame

from typeddfs.df_errors import UnsupportedOperationError


class PrettyDf(_InternalDataFrame, metaclass=abc.ABCMeta):
    """
    A DataFrame with an overridden ``_repr_html_`` and some simple additional methods.
    """

    def column_names(self) -> List[str]:
        """
        Returns the list of columns.

        Returns:
            A Python list
        """
        return list(self.columns)

    def index_names(self) -> List[str]:
        """
        Returns the list of index names.
        Unlike ``self.index.names``, returns ``[]`` instead of ``[None]`` if there is no index.

        Returns:
            A Python list
        """
        lst = list(self.index.names)
        return [] if lst == [None] else lst

    def is_multindex(self) -> bool:
        """
        Returns whether this is a ``pd.MultiIndex``.
        """
        return isinstance(self.index, pd.MultiIndex)

    def n_rows(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self)

    def n_columns(self) -> int:
        """
        Returns the number of columns.
        """
        return len(self.columns)

    def n_indices(self) -> int:
        """
        Returns the number of index names.
        """
        return len(self.index.names)

    @property
    def _constructor_expanddim(self):  # pragma: no cover
        # this raises a NotImplementedError in _InternalDataFrame,
        # so let's override it here to prevent tools and IDEs from complaining
        raise UnsupportedOperationError()

    def _repr_html_(self) -> str:
        """
        Renders HTML for display() in Jupyter notebooks.
        Jupyter automatically uses this function.

        Returns:
            Just a string containing HTML, which will be wrapped in an HTML object
        """
        # noinspection PyProtectedMember
        return (
            f"<strong>{self.__class__.__name__}: {self._dims()}</strong>\n{super()._repr_html_()}"
        )

    def _dims(self) -> str:
        """
        Returns a string describing the dimensionality.

        Returns:
            A text description of the dimensions of this DataFrame
        """
        # we could handle multi-level columns
        # but they're quite rare, and the number of rows is probably obvious when looking at it
        if len(self.index.names) > 1:
            return f"{len(self)} rows × {len(self.columns)} columns, {len(self.index.names)} index columns"
        else:
            return f"{len(self)} rows × {len(self.columns)} columns"


__all__ = ["PrettyDf"]
