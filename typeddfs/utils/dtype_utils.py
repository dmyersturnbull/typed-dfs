# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
"""
Data type tools for typed-dfs.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pandas import BooleanDtype, Interval, Period, StringDtype, DatetimeTZDtype

# noinspection PyProtectedMember
from pandas.api.types import (
    is_bool,
    is_bool_dtype,
    is_complex,
    is_complex_dtype,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_interval,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
)


class DtypeUtils:
    is_integer_dtype = is_integer_dtype
    is_float_dtype = is_float_dtype
    is_bool_dtype = is_bool_dtype
    is_string_dtype = is_string_dtype
    is_complex_dtype = is_complex_dtype
    is_integer = is_integer
    is_float = is_float
    is_bool = is_bool
    is_complex = is_complex
    is_interval = is_interval
    is_numeric_dtype = is_numeric_dtype
    is_object_dtype = is_object_dtype
    is_number = is_number
    is_scalar = is_scalar

    @classmethod
    def describe_dtype(cls, t: type[Any], *, short: bool = False) -> str | None:
        """
        Returns a string name for a Pandas-supported dtype.

        Args:
            t: Any Python type
            short: Use shorter strings (e.g. "int" instead of "integer")

        Returns:
            A string like "floating-point" or "zoned datetime".
            Returns ``None`` if no good name is found or if ``t`` is ``None``.
        """
        if cls.is_bool_dtype(t) or issubclass(t, BooleanDtype):
            return "bool" if short else "boolean"
        elif issubclass(t, datetime) or issubclass(t, DatetimeTZDtype):
            return "datetime" if short else "date and time"
        elif issubclass(t, Period):
            return "period" if short else "time period"
        elif issubclass(t, timedelta):
            return "duration"
        elif issubclass(t, Interval):
            return "interval"
        elif cls.is_integer_dtype(t):
            return "int" if short else "integer"
        elif cls.is_float_dtype(t):
            return "float" if short else "floating-point"
        elif cls.is_complex_dtype(t):
            return "complex" if short else "complex number"
        elif cls.is_numeric_dtype(t):
            return "numeric"
        elif cls.is_string_dtype(t) or t is StringDtype:
            return "str" if short else "string"
        return None


__all__ = ["DtypeUtils"]
