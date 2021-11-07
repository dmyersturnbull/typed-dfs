"""
Tools that could possibly be used outside of typed-dfs.
"""
from __future__ import annotations

import base64
import enum
import inspect
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as _time
from datetime import tzinfo
from decimal import Decimal
from typing import (
    AbstractSet,
    Any,
    ByteString,
    Callable,
    ItemsView,
    KeysView,
    Mapping,
    Optional,
    Sequence,
    ValuesView,
)
from uuid import UUID

import numpy as np
import orjson


class MiscTypesJsonDefault(Callable[[Any], Any]):
    def __call__(self, obj: Any) -> Any:
        """
        Tries to return a serializable result for ``obj``.
        Meant to be passed as ``default=`` in ``orjson.dumps``.
        Only encodes types that can always be represented exactly,
        without any loss of information. For that reason, it does not
        fall back to calling ``str`` or ``repr`` for unknown types.
        Handles, at least:

        - ``decimal.Decimal`` → str (scientific notation)
        - ``complex`` or ``np.complexfloating`` → str (e.g. "(3+1j)")
        - ``typing.Mapping`` → dict
        - ``typing.ItemsView`` → dict
        - ``typing.{AbstractSet,Sequence,...}`` → list
        - ``enum.Enum`` → str (name)
        - ``typing.ByteString`` →  str (base-64)
        - ``datetime.tzinfo`` →  str (timezone name)
        - ``typing.NamedTuple`` →  dict
        - type or module →  str (name)

        Raise:
            TypeError: If none of those options worked
        """
        if obj is None:
            return obj  # we should never get here, but this seems safer
        elif isinstance(obj, (str, int, float, datetime, date, _time, UUID)):
            return obj  # we should never get here, but let's be safe
        elif (
            isinstance(obj, Decimal)
            or isinstance(obj, complex)
            or isinstance(obj, np.complexfloating)
        ):
            return str(obj)
        elif isinstance(obj, enum.Enum):
            return obj.name
        elif isinstance(obj, bytes):
            return base64.b64decode(obj)
        elif isinstance(obj, ByteString):
            return base64.b64decode(bytes(obj))
        elif isinstance(obj, tzinfo):
            return obj.tzname(datetime.now(tz=obj))
        elif isinstance(obj, (AbstractSet, Sequence, KeysView, ValuesView)):
            return list(obj)
        elif isinstance(obj, (Mapping, ItemsView)):
            return dict(obj)
        elif isinstance(obj, tuple) and hasattr(obj, "_asdict"):
            # namedtuple
            return obj._asdict()
        elif inspect.isclass(obj) or inspect.ismodule(obj):
            return obj.__qualname__
        raise TypeError


_misc_types_default = MiscTypesJsonDefault()


@dataclass(frozen=True, repr=True)
class JsonEncoder:
    bytes_options: int
    str_options: int
    default: Callable[[Any], Any]
    prep: Optional[Callable[[Any], Any]]

    def as_bytes(self, data: Any) -> ByteString:
        if self.prep is not None:
            data = self.prep(data)
        return orjson.dumps(data, default=self.default, option=self.bytes_options)

    def as_str(self, data: Any) -> str:
        if self.prep is not None:
            data = self.prep(data)
        x = orjson.dumps(data, default=self.default, option=self.str_options)
        return x.decode(encoding="utf8") + "\n"


@dataclass(frozen=True, repr=True)
class JsonDecoder:
    def from_bytes(self, data: ByteString) -> Any:
        if not isinstance(data, ByteString):
            raise TypeError(str(type(data)))
        if not isinstance(data, bytes):
            data = bytes(data)
        return orjson.loads(data)

    def from_str(self, data: str) -> Any:
        return orjson.loads(data)


class JsonUtils:
    @classmethod
    def misc_types_default(cls) -> Callable[[Any], Any]:
        return _misc_types_default

    @classmethod
    def new_default(
        cls,
        *fallbacks: Optional[Callable[[Any], Any]],
        first: Optional[Callable[[Any], Any]] = _misc_types_default,
        last: Optional[Callable[[Any], Any]] = str,
    ) -> Callable[[Any], Any]:
        """
        Creates a new method to be passed as ``default=`` to ``orjson.dumps``.
        Tries, in order: :meth:`orjson_default`, ``fallbacks``, then ``str``.

        Args:
            first: Try this first
            fallbacks: Tries these, in order, after ``first``, skipping any None
            last: Use this as the last resort; consider ``str`` or ``repr``
        """
        then = [f for f in [first, *fallbacks] if f is not None]

        def _default(obj):
            for t in then:
                try:
                    return t(obj)
                except TypeError:
                    pass
                if last is None:
                    raise TypeError
            return last(obj)

        _default.__name__ = f"default({', '.join([str(t) for t in then])})"
        return _default

    @classmethod
    def decoder(cls) -> JsonDecoder:
        return JsonDecoder()

    @classmethod
    def encoder(
        cls,
        *fallbacks: Optional[Callable[[Any], Any]],
        indent: bool = True,
        sort: bool = False,
        preserve_inf: bool = True,
        last: Optional[Callable[[Any], Any]] = str,
    ) -> JsonEncoder:
        """
        Serializes to string with orjson, indenting and adding a trailing newline.
        Uses :meth:`orjson_default` to encode more types than orjson can.

        Args:
            indent: Indent by 2 spaces
            preserve_inf: Preserve infinite values with :meth:`orjson_preserve_inf`
            sort: Sort keys with ``orjson.OPT_SORT_KEYS``;
                  only for :meth:`typeddfs.json_utils.JsonEncoder.as_str`
            last: Last resort option to encode a value
        """
        bytes_option = orjson.OPT_UTC_Z | orjson.OPT_NON_STR_KEYS
        str_option = orjson.OPT_UTC_Z
        if sort:
            bytes_option |= orjson.OPT_SORT_KEYS
            str_option |= orjson.OPT_SORT_KEYS
        if indent:
            str_option |= orjson.OPT_INDENT_2
        default = cls.new_default(
            *fallbacks,
            first=_misc_types_default,
            last=last,
        )
        prep = cls.preserve_inf if preserve_inf else None
        return JsonEncoder(
            default=default,
            bytes_options=bytes_option,
            str_options=str_option,
            prep=prep,
        )

    @classmethod
    def preserve_inf(cls, data: Any) -> Any:
        """
        Recursively replaces infinite float and numpy values with strings.
        Orjson encodes NaN, inf, and +inf as JSON null.
        This function converts to string as needed to preserve infinite values.
        Any float scalar (``np.floating`` and ``float``) will be replaced with a string.
        Any ``np.ndarray``, whether it contains an infinite value or not, will be converted
        to an ndarray of strings.
        The returned result may still not be serializable with orjson or :meth:`orjson_bytes`.
        Trying those methods is the best way to test for serializablity.
        """
        if isinstance(data, Mapping):
            return {str(k): cls.preserve_inf(v) for k, v in data.items()}
        elif (
            isinstance(data, Sequence)
            and not isinstance(data, str)
            and not isinstance(data, ByteString)
        ):
            if all((isinstance(v, (float, np.floating)) and np.isinf(v)) for v in data):
                return [str(v) for v in data]
            else:
                return [cls.preserve_inf(v) for v in data]
        elif isinstance(data, (float, np.floating)) and np.isinf(data):
            return str(data)
        elif isinstance(data, np.ndarray):
            # noinspection PyTypeChecker
            return data.astype(str).tolist()
        return data


__all__ = ["JsonEncoder", "JsonDecoder", "JsonUtils"]
