"""
Mixin that overrides Pandas functions to retype.
"""

import pandas as pd
from pandas.core.frame import DataFrame as _InternalDataFrame

from typeddfs.df_errors import UnsupportedOperationError


class _RetypeMixin:
    def __add__(self, other):
        x = super().__add__(other)
        return self._change_if_df(x)

    def __radd__(self, other):
        x = super().__radd__(other)
        return self._change_if_df(x)

    def __sub__(self, other):
        x = super().__sub__(other)
        return self._change_if_df(x)

    def __rsub__(self, other):
        x = super().__rsub__(other)
        return self._change_if_df(x)

    def __mul__(self, other):
        x = super().__mul__(other)
        return self._change_if_df(x)

    def __rmul__(self, other):
        x = super().__rmul__(other)
        return self._change_if_df(x)

    def __truediv__(self, other):
        x = super().__truediv__(other)
        return self._change_if_df(x)

    def __rtruediv__(self, other):
        x = super().__rtruediv__(other)
        return self._change_if_df(x)

    def __divmod__(self, other):
        x = super().__divmod__(other)
        return self._change_if_df(x)

    def __rdivmod__(self, other):
        x = super().__rdivmod__(other)
        return self._change_if_df(x)

    def __mod__(self, other):
        x = super().__mod__(other)
        return self._change_if_df(x)

    def __rmod__(self, other):
        x = super().__rmod__(other)
        return self._change_if_df(x)

    def __pow__(self, other):
        x = super().__pow__(other)
        return self._change_if_df(x)

    def __rpow__(self, other):
        x = super().__rpow__(other)
        return self._change_if_df(x)

    def drop_duplicates(self, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        return self.__class__._change(super().drop_duplicates(**kwargs))

    def reindex(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        return self.__class__._change(super().reindex(*args, **kwargs))

    def sort_values(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().sort_values(*args, **kwargs)
        return self.__class__._change(df)

    def reset_index(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().reset_index(*args, **kwargs)
        return self.__class__._change(df)

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ) -> __qualname__:
        self._no_inplace(dict(inplace=inplace))
        if len(keys) == 0 and append:
            return self
        elif len(keys) == 0:
            return self.__class__._change(super().reset_index(drop=drop))
        df = super().set_index(
            keys=keys,
            drop=drop,
            append=append,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )
        return self.__class__._change(df)

    # noinspection PyFinal
    def asfreq(self, *args, **kwargs) -> __qualname__:
        return super().asfreq(*args, **kwargs)

    # noinspection PyFinal
    def shift(self, *args, **kwargs) -> __qualname__:
        return super().shift(*args, **kwargs)

    # noinspection PyFinal
    def tz_localize(self, *args, **kwargs) -> __qualname__:
        return super().tz_localize(*args, **kwargs)

    # noinspection PyFinal
    def tz_convert(self, *args, **kwargs) -> __qualname__:
        return super().tz_convert(*args, **kwargs)

    # noinspection PyFinal
    def to_timestamp(self, *args, **kwargs) -> __qualname__:
        return super().to_timestamp(*args, **kwargs)

    # noinspection PyFinal
    def to_period(self, *args, **kwargs) -> __qualname__:
        return super().to_period(*args, **kwargs)

    # noinspection PyFinal
    def convert_dtypes(self, *args, **kwargs) -> __qualname__:
        df = super().convert_dtypes(*args, **kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def infer_objects(self, *args, **kwargs) -> __qualname__:
        df = super().infer_objects(*args, **kwargs)
        return self.__class__._change(df)

    def dropna(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().dropna(*args, **kwargs)
        return self.__class__._change(df)

    def fillna(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().fillna(*args, **kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def copy(self, deep: bool = False) -> __qualname__:
        df = super().copy(deep=deep)
        return self.__class__._change(df)

    def assign(self, **kwargs) -> __qualname__:
        df = self.vanilla_reset()
        df = df.assign(**kwargs)
        return self.__class__._change(df)

    def append(self, *args, **kwargs) -> __qualname__:
        df = super().append(*args, **kwargs)
        return self.__class__._change(df)

    def transpose(self, *args, **kwargs) -> __qualname__:
        df = super().transpose(*args, **kwargs)
        return self.__class__._change(df)

    def truncate(self, *args, **kwargs) -> __qualname__:
        df = super().truncate(*args, **kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def ffill(self, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().ffill(**kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def bfill(self, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().bfill(**kwargs)
        return self.__class__._change(df)

    # noinspection PyFinal
    def abs(self) -> __qualname__:
        return self.__class__._change(super().abs())

    def rename(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().rename(*args, **kwargs)
        return self.__class__._change(df)

    def replace(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().replace(*args, **kwargs)
        return self.__class__._change(df)

    def applymap(self, *args, **kwargs) -> __qualname__:
        df = super().applymap(*args, **kwargs)
        return self.__class__._change(df)

    def astype(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().astype(*args, **kwargs)
        return self.__class__._change(df)

    def drop(self, *args, **kwargs) -> __qualname__:
        self._no_inplace(kwargs)
        df = super().drop(*args, **kwargs)
        return self.__class__._change(df)

    @classmethod
    def _convert_typed(cls, df: pd.DataFrame):
        # not great, but works ok
        # if this is a BaseDf, use convert
        # otherwise, just use check_and_change
        if hasattr(cls, "convert"):
            return cls.convert(df)
        else:
            return cls._change(df)

    @classmethod
    def _change_if_df(cls, df):
        if isinstance(df, _InternalDataFrame):
            df.__class__ = cls
        return df

    @classmethod
    def _change(cls, df) -> __qualname__:
        df.__class__ = cls
        return df

    def _no_inplace(self, kwargs):
        if kwargs.get("inplace") is True:  # pragma: no cover
            raise UnsupportedOperationError("inplace not supported. Use vanilla() if needed.")


__all__ = ["_RetypeMixin"]
