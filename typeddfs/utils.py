"""
Tools that could possibly be used outside of typed-dfs.
"""
from __future__ import annotations

import collections
import os
import sys
import typing
from typing import (
    AbstractSet,
    Any,
    Collection,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    Generator,
)

from natsort import ns, ns_enum
import numpy as np
import regex

# noinspection PyProtectedMember
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_complex_dtype,
    is_integer,
    is_float,
    is_bool,
    is_categorical,
    is_complex,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_period_dtype,
    is_interval,
    is_numeric_dtype,
    is_object_dtype,
    is_number,
    is_interval_dtype,
    is_extension_type,
    is_scalar,
)
from pandas.io.common import get_handle

# noinspection PyProtectedMember
from tabulate import DataRow, TableFormat, _table_formats

from typeddfs._utils import _DEFAULT_HASH_ALG, _AUTO_DROPPED_NAMES, _FORBIDDEN_NAMES
from typeddfs.frozen_types import FrozeDict, FrozeList, FrozeSet


_control_chars = regex.compile(r"\p{C}", flags=regex.V1)


class Utils:

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
    def strip_control_chars(cls, s: str) -> str:
        return _control_chars.sub("", s)

    @classmethod
    def freeze(cls, v: Any) -> Any:
        """
        Returns ``v`` or a hashable view of it.
        Note that the returned types must be hashable but might not be ordered.
        You can generally add these values as DataFrame elements, but you might not
        be able to sort on those columns.

        Args:
            v: Any value

        Returns:
            Either ``v`` itself,
            a :class:`typeddfs.utils.FrozeSet` (subclass of :class:`typing.AbstractSet`),
            a :class:`typeddfs.utils.FrozeList` (subclass of :class:`typing.Sequence`),
            or a :class:`typeddfs.utils.FrozeDict` (subclass of :class:`typing.Mapping`).
            int, float, str, np.generic, and tuple are always returned as-is.

        Raises:
            AttributeError: If ``v`` is not hashable and could not converted to
                            a FrozeSet, FrozeList, or FrozeDict, *or* if one of the elements for
                            one of the above types is not hashable.
            TypeError: If ``v`` is an ``Iterator`` or `collections.deque``
        """
        if isinstance(v, (int, float, str, np.generic, tuple, frozenset)):
            return v  # short-circuit
        if isinstance(v, Iterator):  # let's not ruin their iterator by traversing
            raise TypeError("Type is an iterator")
        if isinstance(v, collections.deque):  # the only other major built-in type we won't accept
            raise TypeError("Type is a deque")
        if isinstance(v, Sequence):
            return FrozeList(v)
        if isinstance(v, AbstractSet):
            return FrozeSet(v)
        if isinstance(v, Mapping):
            return FrozeDict(v)
        hash(v)  # let it raise an AttributeError
        return v

    @classmethod
    def default_hash_algorithm(cls) -> str:
        return _DEFAULT_HASH_ALG

    @classmethod
    def insecure_hash_functions(cls) -> Set[str]:
        return {"md5", "sha1"}

    @classmethod
    def property_key_escape(cls, s: str) -> str:
        p = regex.compile(r"([ =:\\])", flags=regex.V1)
        return p.sub(r"\\\1", s)

    @classmethod
    def property_key_unescape(cls, s: str) -> str:
        p = regex.compile(r"\\([ =:\\])", flags=regex.V0)
        return p.sub(r"\1", s)

    @classmethod
    def property_value_escape(cls, s: str) -> str:
        return s.replace("\\", "\\\\")

    @classmethod
    def property_value_unescape(cls, s: str) -> str:
        return s.replace("\\\\", "\\")

    @classmethod
    def banned_names(cls) -> Set[str]:
        """
        Lists strings that cannot be used for column names or index level names.
        """
        return {*_AUTO_DROPPED_NAMES, *_FORBIDDEN_NAMES}

    @classmethod
    def dicts_to_toml_aot(cls, dicts: Sequence[Mapping[str, Any]]):
        """
        Make a tomlkit Document consisting of an array of tables ("AOT").

        Args:
            dicts: A sequence of dictionaries

        Returns:
            A tomlkit AOT
        """
        import tomlkit

        aot = tomlkit.aot()
        for ser in dicts:
            tab = tomlkit.table()
            aot.append(tab)
            for k, v in ser.items():
                tab.add(k, v)
            tab.add(tomlkit.nl())
        return aot

    @classmethod
    def dots_to_dict(cls, items: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Make sub-dictionaries from substrings in ``items`` delimited by ``.``.
        Used for TOML.

        Examples:
            ``Utils.dots_to_dict({"genus.species": "fruit bat"}) == {"genus": {"species": "fruit bat"}}``

        See Also:
            :meth:`dict_to_dots`
        """
        dct = {}
        cls._un_leaf(dct, items)
        return dct

    @classmethod
    def dict_to_dots(cls, items: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Performs the inverse of :meth:`dots_to_dict`.

        Examples:
            ``Utils.dict_to_dots({"genus": {"species": "fruit bat"}}) == {"genus.species": "fruit bat"}``
        """
        return dict(cls._re_leaf("", items))

    @classmethod
    def write(cls, path_or_buff, content, *, mode: str = "w", **kwargs) -> Optional[str]:
        """
        Writes using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        if path_or_buff is None:
            return content
        with get_handle(path_or_buff, mode, **kwargs) as f:
            f.handle.write(content)

    @classmethod
    def read(cls, path_or_buff, *, mode: str = "r", **kwargs) -> str:
        """
        Reads using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        with get_handle(path_or_buff, mode, **kwargs) as f:
            return f.handle.read()

    @classmethod
    def get_encoding(cls, encoding: str = "utf-8") -> str:
        """
        Returns a text encoding from a more flexible string.
        Ignores hyphens and lowercases the string.
        Permits these nonstandard shorthands:

          - "platform": use ``sys.getdefaultencoding()`` on the fly
          - "utf8(bom)": use "utf-8-sig" on Windows; "utf-8" otherwise
          - "utf16(bom)": use "utf-16-sig" on Windows; "utf-16" otherwise
        """
        encoding = encoding.lower().replace("-", "")
        if encoding == "platform":
            encoding = sys.getdefaultencoding()
        if encoding == "utf8(bom)":
            encoding = "utf-8-sig" if os.name == "nt" else "utf-8"
        if encoding == "utf16(bom)":
            encoding = "utf-16-sig" if os.name == "nt" else "utf-16"
        return encoding

    @classmethod
    def all_natsort_flags(cls) -> Mapping[str, int]:
        """
        Simply returns the mapping between natsort flag names and their int values.
        "Combined" flags such as ``ns.REAL`` are included.
        """
        # import enum_fields, enum_combos, enum_aliases
        return dict(ns_enum.enum_fields)

    @classmethod
    def guess_natsort_alg(cls, dtype: Type[Any]) -> Tuple[Set[str], int]:
        """
        Guesses a good natsorted flag for the dtype.

        Here are some specifics:
            - integers       ==> INT and SIGNED
            - floating-point ==> FLOAT and SIGNED
            - strings        ==> COMPATIBILITYNORMALIZE and GROUPLETTERS
            - datetime       ==> GROUPLETTERS (only affects 'Z' vs. 'z'; shouldn't matter)

        Args:
            dtype: Probably from ``pd.Series.dtype``

        Returns:
            A tuple of (set of flags, int) -- see :meth:`exact_natsort_alg`
        """
        st, x = set(), 0
        if is_string_dtype(dtype):
            st.update(["COMPATIBILITYNORMALIZE", "GROUPLETTERS"])
            x |= ns_enum.ns.COMPATIBILITYNORMALIZE | ns_enum.ns.GROUPLETTERS
        elif is_datetime64_any_dtype(dtype):
            st.update(["GROUPLETTERS"])
            x |= ns_enum.ns.GROUPLETTERS
        elif is_categorical_dtype(dtype):
            pass
        elif is_integer_dtype(dtype) or is_bool_dtype(dtype):
            st.update(["INT", "SIGNED"])
            x |= ns_enum.ns.INT | ns_enum.ns.SIGNED
        elif is_float_dtype(dtype):
            st.update(["FLOAT", "SIGNED"])
            x |= ns_enum.ns.FLOAT | ns_enum.ns.SIGNED  # same as ns_enum.ns.REAL
        return st, x

    @classmethod
    def exact_natsort_alg(
        cls, flags: Union[int, Collection[Union[int, str]]]
    ) -> Tuple[Set[str], int]:
        """
        Gets the flag names and combined ``alg=`` argument for natsort.

        Examples:
            - exact_natsort_alg({"REAL"}) == ({"FLOAT", "SIGNED"}, ns.FLOAT | ns.SIGNED)
            - exact_natsort_alg({}) == ({}, 0)
            - exact_natsort_alg(ns.LOWERCASEFIRST) == ({"LOWERCASEFIRST"}, ns.LOWERCASEFIRST)
            - exact_natsort_alg({"localenum", "numafter"})
              == ({"LOCALENUM", "NUMAFTER"}, ns.LOCALENUM | ns.NUMAFTER)

        Args:
            flags: Can be either:
                   - a single integer ``alg`` argument
                   - a set of flag ints and/or names in ``natsort.ns``

        Returns:
            A tuple of the set of flag names, and the corresponding input to ``natsorted``
            Only uses standard flag names, never the "combined" ones.
            (E.g. ``exact_natsort_alg({"REAL"})``
            will return ``({"FLOAT", "SIGNED"}, ns.FLOAT | ns.SIGNED)``.
        """
        if isinstance(flags, str):
            flags = {flags}
        if (
            flags is None
            or isinstance(flags, Collection)
            and len(flags) == 0
            or isinstance(flags, int)
            and flags == 0
        ):
            return set(), 0
        if isinstance(flags, int):
            return cls._ns_info_from_int_flag(flags)
        if isinstance(flags, Collection):
            x = 0
            for f in flags:
                if isinstance(f, str):
                    x |= getattr(ns, f.upper())
                elif isinstance(f, int):
                    x |= f
                else:
                    raise TypeError(f"Unknown type {type(flags)} for {flags}")
            return cls._ns_info_from_int_flag(x)
        raise TypeError(f"Unknown type {type(flags)} for {flags}")

    @classmethod
    def table_formats(cls) -> Sequence[str]:
        """
        Returns the names of styles for :py:mod`tabulate`.
        """
        return _table_formats.keys()

    @classmethod
    def table_format(cls, fmt: str) -> TableFormat:
        """
        Gets a :py:mod`tabulate` style by name.

        Returns:
            A TableFormat, which can be passed as a style
        """
        return _table_formats[fmt]

    @classmethod
    def plain_table_format(cls, sep: str = " ", **kwargs) -> TableFormat:
        """
        Creates a simple :py:mod`tabulate` style using a column-delimiter ``sep``.

        Returns:
            A TableFormat, which can be passed as a style
        """
        defaults = dict(
            lineabove=None,
            linebelowheader=None,
            linebetweenrows=None,
            linebelow=None,
            headerrow=DataRow("", sep, ""),
            datarow=DataRow("", sep, ""),
            padding=0,
            with_header_hide=None,
        )
        kwargs = {**defaults, **kwargs}
        return TableFormat(**kwargs)

    @classmethod
    def _ns_info_from_int_flag(cls, flags: int) -> Tuple[Set[str], int]:
        ignored = {*dict(ns_enum.enum_aliases).keys(), *dict(ns_enum.enum_combos).keys()}
        st = set()
        for f, v in ns_enum.enum_fields.items():
            if f in ns_enum.enum_fields and (v & flags) != 0 and f not in ignored:
                st.add(f)
        return st, flags

    @classmethod
    def _un_leaf(cls, to: typing.MutableMapping[str, Any], items: Mapping[str, Any]) -> None:
        keys = {k.split(".", 1) for k in items.keys()}
        for major_key in keys:
            of_major_key = {k: v for k, v in items.items() if k.split(".", 1) == major_key}
            if len(major_key) > 0:
                to[major_key] = {}
                cls._un_leaf(to[major_key], of_major_key)
            else:
                to[major_key] = of_major_key

    @classmethod
    def _re_leaf(cls, at: str, items: Mapping[str, Any]) -> Generator[Tuple[str, Any], None, None]:
        for k, v in items.items():
            me = at + "." + k
            if hasattr(v, "items") and hasattr(v, "keys") and hasattr(v, "values"):
                yield from cls._re_leaf(me, v)
            else:
                yield me, v


__all__ = ["Utils", "TableFormat"]
