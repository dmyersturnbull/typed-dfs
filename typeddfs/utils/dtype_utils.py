"""
Data type tools for typed-dfs.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Type

from pandas import BooleanDtype, Interval, Period, StringDtype

# noinspection PyProtectedMember
from pandas.api.types import (
    is_bool,
    is_bool_dtype,
    is_categorical,
    is_categorical_dtype,
    is_complex,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_extension_type,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_interval,
    is_interval_dtype,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_period_dtype,
    is_scalar,
    is_string_dtype,
)


class DtypeUtils:

    is_integer_dtype = is_integer_dtype
    is_float_dtype = is_float_dtype
    is_bool_dtype = is_bool_dtype
    is_string_dtype = is_string_dtype
    is_categorical_dtype = is_categorical_dtype
    is_complex_dtype = is_complex_dtype
    is_integer = is_integer
    is_float = is_float
    is_bool = is_bool
    is_categorical = is_categorical
    is_complex = is_complex
    is_datetime64_any_dtype = is_datetime64_any_dtype
    is_datetime64tz_dtype = is_datetime64tz_dtype
    is_period_dtype = is_period_dtype
    is_interval = is_interval
    is_numeric_dtype = is_numeric_dtype
    is_object_dtype = is_object_dtype
    is_number = is_number
    is_interval_dtype = is_interval_dtype
    is_extension_type = is_extension_type
    is_scalar = is_scalar

    @classmethod
    def describe_dtype(cls, t: Type[Any], *, short: bool = False) -> Optional[str]:
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
        elif (
            cls.is_datetime64tz_dtype(t)
            or cls.is_datetime64_any_dtype(t)
            or issubclass(t, datetime)
        ):
            return "datetime" if short else "date and time"
        elif cls.is_period_dtype(t) or issubclass(t, Period):
            return "period" if short else "time period"
        elif issubclass(t, timedelta):
            return "duration"
        elif cls.is_interval_dtype(t) or issubclass(t, Interval):
            return "interval"
        elif cls.is_integer_dtype(t):
            return "int" if short else "integer"
        elif cls.is_float_dtype(t):
            return "float" if short else "floating-point"
        elif cls.is_complex_dtype(t):
            return "complex" if short else "complex number"
        elif cls.is_numeric_dtype(t):
            return "numeric"
        elif cls.is_categorical_dtype(t):
            return "categorical"
        elif cls.is_string_dtype(t) or t is StringDtype:
            return "str" if short else "string"
        return None


__all__ = ["DtypeUtils"]
