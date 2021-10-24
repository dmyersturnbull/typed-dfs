"""
Mixin for Feather, Parquet, and HDF5.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from typeddfs.utils._utils import PathLike


class _FeatherParquetHdfMixin:
    @classmethod
    def read_feather(cls, *args, **kwargs) -> __qualname__:
        # feather does not support MultiIndex, so reset index and use convert()
        try:
            df = pd.read_feather(*args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        if "__feather_ignore_" in df.columns:
            df = df.drop("__feather_ignore_", axis=1)
        return cls._convert_typed(df)

    # noinspection PyMethodOverriding,PyBroadException,DuplicatedCode
    def to_feather(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        # feather does not support MultiIndex, so reset index and use convert()
        # if an error occurs you end up with a 0-byte file
        # this is fixed with exactly the same logic as for to_hdf -- see that method
        try:
            old_size = os.path.getsize(path_or_buf)
        except BaseException:
            old_size = None
        df = self.vanilla_reset()
        if len(df) == len(df.columns) == 0:
            df = df.append(
                pd.Series(dict(__feather_ignore_="__feather_ignore_")), ignore_index=True
            )
        df.columns = df.columns.astype(str)
        try:
            return df.to_feather(path_or_buf, *args, **kwargs)
        except BaseException:
            try:
                size = os.path.getsize(path_or_buf)
            except BaseException:
                size = None
            if size is not None and size == 0 and (old_size is None or old_size > 0):
                try:
                    Path(path_or_buf).unlink()
                except BaseException:
                    pass
            raise

    @classmethod
    def read_parquet(cls, *args, **kwargs) -> __qualname__:
        # parquet does not support MultiIndex, so reset index and use convert()
        try:
            df = pd.read_parquet(*args, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out what EmptyDataError means
            # df = pd.DataFrame()
            return cls.new_df()
        return cls._convert_typed(df)

    # noinspection PyMethodOverriding,PyBroadException,DuplicatedCode
    def to_parquet(self, path_or_buf, *args, **kwargs) -> Optional[str]:
        # parquet does not support MultiIndex, so reset index and use convert()
        # if an error occurs you end up with a 0-byte file
        # this is fixed with exactly the same logic as for to_hdf -- see that method
        try:
            old_size = os.path.getsize(path_or_buf)
        except BaseException:
            old_size = None
        reset = self.vanilla_reset()
        for c in reset.columns:
            if reset[c].dtype in [np.ubyte, np.ushort]:
                reset[c] = reset[c].astype(np.intc)
            elif reset[c].dtype == np.uintc:
                reset[c] = reset[c].astype(np.long)
            elif reset[c].dtype == np.half:
                reset[c] = reset[c].astype(np.float32)
        try:
            return reset.to_parquet(path_or_buf, *args, **kwargs)
        except BaseException:
            try:
                size = os.path.getsize(path_or_buf)
            except BaseException:
                size = None
            if size is not None and size == 0 and (old_size is None or old_size > 0):
                try:
                    Path(path_or_buf).unlink()
                except BaseException:
                    pass
            raise

    @classmethod
    def read_hdf(
        cls, *args, key: Optional[str] = None, **kwargs
    ) -> __qualname__:  # pragma: no cover
        if key is None:
            key = cls.get_typing().io.hdf_key
        try:
            df = pd.read_hdf(*args, key=key, **kwargs)
        except pd.errors.EmptyDataError:
            # TODO: Figure out exactly what EmptyDataError is
            return cls.new_df()
        # noinspection PyTypeChecker
        return cls._convert_typed(df)

    # noinspection PyBroadException,PyFinal,DuplicatedCode
    def to_hdf(
        self, path: PathLike, key: Optional[str] = None, **kwargs
    ) -> None:  # pragma: no cover
        path = Path(path)
        # if an error occurs you end up with a 0-byte file
        # delete it if and only if we CREATED an empty file --
        # subtle, but: we shouldn't delete the 0-byte file if it
        # already existed and was 0 bytes
        #
        # just wrap in try-except -- it might not be a file and might not exist
        # technically there's an edge case: what if it was just not readable?
        # if it isn't readable now but becomes readable (and writable) by the time
        # we try to write, then we delete it anyway
        # that's a super unlikely bug and shouldn't matter anyway
        if key is None:
            key = self.__class__.get_typing().io.hdf_key
        try:
            old_size = os.path.getsize(path)
        except BaseException:
            old_size = None
        df = self.vanilla()
        try:
            df.to_hdf(str(path), key, **kwargs)
        except BaseException:
            # noinspection PyBroadException
            try:
                size = os.path.getsize(path)
            except BaseException:
                size = None
            if size is not None and size == 0 and (old_size is None or old_size > 0):
                try:
                    Path(path).unlink()
                except BaseException:
                    pass
            raise


__all__ = ["_FeatherParquetHdfMixin"]
