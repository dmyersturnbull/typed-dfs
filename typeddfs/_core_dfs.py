import abc

import pandas as pd

from typeddfs._mixins._new_methods_mixin import _NewMethodsMixin
from typeddfs._mixins._retype_mixin import _RetypeMixin
from typeddfs._pretty_dfs import PrettyDf


class CoreDf(_RetypeMixin, _NewMethodsMixin, PrettyDf, metaclass=abc.ABCMeta):
    """
    An abstract Pandas DataFrame subclass with additional methods.
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        # noinspection PyTypeChecker
        self.__class__._change(self)

    @classmethod
    def new_df(cls, **kwargs) -> __qualname__:
        """
        Creates a new, somewhat arbitrary DataFrame of this type.
        Calling this with no arguments should always be supported.

        Arguments:
            **kwargs: These should be narrowed by the overriding method as needed.

        Raises:
            UnsupportedOperationError: Can be raised if a valid DataFrame is too difficult to create.
            InvalidDfError: May be raised if the type requires specific constraints
                            and did not overload this method to account for them.
                            While programmers using the type should be aware of this possibility,
                            consuming code, in general, should assume that ``new_df`` will always work.
        """
        raise NotImplementedError()

    def vanilla_reset(self) -> pd.DataFrame:
        """
        Same as :meth:`vanilla`, but resets the index -- but dropping the index if it has no name.
        This means that an effectively index-less dataframe will not end up with an extra column
        called "index".
        """
        if len(self.index_names()) > 0:
            return self.vanilla().reset_index()
        else:
            return self.vanilla().reset_index(drop=True)

    def vanilla(self) -> pd.DataFrame:
        """
        Makes a copy that's a normal Pandas DataFrame.

        Returns:
            A shallow copy with its ``__class__`` set to pd.DataFrame
        """
        df = self.copy()
        df.__class__ = pd.DataFrame
        return df


__all__ = ["CoreDf"]
