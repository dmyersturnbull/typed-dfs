"""
Tools that could possibly be used outside of typed-dfs.
"""
from __future__ import annotations

import collections
import os
import sys
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Collection,
    Generator,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import regex
from natsort import natsorted, ns, ns_enum
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
from pandas.io.common import get_handle

# noinspection PyProtectedMember
from tabulate import DataRow, TableFormat, _table_formats

from typeddfs._utils import _AUTO_DROPPED_NAMES, _DEFAULT_HASH_ALG, _FORBIDDEN_NAMES
from typeddfs.df_errors import WritePermissionsError
from typeddfs.frozen_types import FrozeDict, FrozeList, FrozeSet

T = TypeVar("T")
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
    def join_to_str(cls, *items: Any, last: str, sep: str = ", ") -> str:
        """
        Joins items to something like "cat, dog, and pigeon" or "cat, dog, or pigeon".

        Args:
            *items: Items to join; ``str(item) for item in items`` will be used
            last: Probably "and", "or", "and/or", or ""
                    Spaces are added/removed as needed if ``suffix`` is alphanumeric
                    or "and/or", after stripping whitespace off the ends.
            sep: Used to separate all words; include spaces as desired

        Examples:
            - ``join_to_str(["cat", "dog", "elephant"], last="and")  # cat, dog, and elephant``
            - ``join_to_str(["cat", "dog"], last="and")  # cat and dog``
            - ``join_to_str(["cat", "dog", "elephant"], last="", sep="/")  # cat/dog/elephant``
        """
        if last.strip().isalpha() or last.strip() == "and/or":
            last = last.strip() + " "
        items = [str(s).strip("'" + '"' + " ") for s in items]
        if len(items) > 2:
            return sep.join(items[:-1]) + sep + last + items[-1]
        else:
            return (" " + last + " ").join(items)

    @classmethod
    def strip_control_chars(cls, s: str) -> str:
        """
        Strips all characters under the Unicode 'Cc' category.
        """
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

    @classmethod
    def verify_can_write_files(cls, *paths: Union[str, Path], missing_ok: bool = False) -> None:
        """
        Checks that all files can be written to, to ensure atomicity before operations.

        Args:
            *paths: The files
            missing_ok: Don't raise an error if a path doesn't exist

        Returns:
            WritePermissionsError: If a path is not a file (modulo existence) or doesn't have 'W' set
        """
        paths = [Path(p) for p in paths]
        for path in paths:
            if path.exists() and not path.is_file():
                raise WritePermissionsError(f"Path {path} is not a file", key=str(path))
            if (not missing_ok or path.exists()) and not os.access(path, os.W_OK):
                raise WritePermissionsError(f"Cannot write to {path}", key=str(path))

    @classmethod
    def verify_can_write_dirs(cls, *paths: Union[str, Path], missing_ok: bool = False) -> None:
        """
        Checks that all directories can be written to, to ensure atomicity before operations.

        Args:
            *paths: The directories
            missing_ok: Don't raise an error if a path doesn't exist

        Returns:
            WritePermissionsError: If a path is not a directory (modulo existence) or doesn't have 'W' set
        """
        paths = [Path(p) for p in paths]
        for path in paths:
            if path.exists() and not path.is_dir():
                raise WritePermissionsError(f"Path {path} is not a dir", key=str(path))
            if missing_ok and not path.exists():
                continue
            if not os.access(path, os.W_OK):
                raise WritePermissionsError(f"{path} lacks write permission", key=str(path))
            if not os.access(path, os.X_OK):
                raise WritePermissionsError(f"{path} lacks access permission", key=str(path))

    @classmethod
    def default_hash_algorithm(cls) -> str:
        return _DEFAULT_HASH_ALG

    @classmethod
    def insecure_hash_functions(cls) -> Set[str]:
        return {"md5", "sha1"}

    @classmethod
    def property_key_escape(cls, s: str) -> str:
        """
        Escapes a key in a .property file.
        """
        p = regex.compile(r"([ =:\\])", flags=regex.V1)
        return p.sub(r"\\\1", s)

    @classmethod
    def property_key_unescape(cls, s: str) -> str:
        """
        Un-escapes a key in a .property file.
        """
        p = regex.compile(r"\\([ =:\\])", flags=regex.V0)
        return p.sub(r"\1", s)

    @classmethod
    def property_value_escape(cls, s: str) -> str:
        """
        Escapes a value in a .property file.
        """
        return s.replace("\\", "\\\\")

    @classmethod
    def property_value_unescape(cls, s: str) -> str:
        """
        Un-escapes a value in a .property file.
        """
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
            A tomlkit`AoT<https://github.com/sdispater/tomlkit/blob/master/tomlkit/items.py>`_
            (i.e. ``[[array]]`)
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

        Example:
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

        Example:
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

          - ``"platform"``: use ``sys.getdefaultencoding()`` on the fly
          - ``"utf8(bom)"``: use ``"utf-8-sig"`` on Windows; ``"utf-8"`` otherwise
          - ``"utf16(bom)"``: use ``"utf-16-sig"`` on Windows; ``"utf-16"`` otherwise
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
    def natsort(
        cls,
        lst: typing.Iterable[T],
        dtype: Type[T],
        *,
        alg: Union[None, int, Set[str]] = None,
        reverse: bool = False,
    ) -> Sequence[T]:
        """
        Perform a natural sort consistent with the type ``dtype``.
        Uses `natsort <https://pypi.org/project/natsort>`_.

        See Also:
            :meth:`guess_natsort_alg`

        Args:
            lst: A sequence of things to sort
            dtype: The type; must be a subclass of each element in ``lst``
            alg: A specific natsort algorithm or set of flags
            reverse: Sort in reverse (e.g. Z to A or 9 to 1)
        """
        if alg is None:
            _, alg = Utils.guess_natsort_alg(dtype)
        else:
            _, alg = Utils.exact_natsort_alg(alg)
        lst = list(lst)
        return natsorted(lst, alg=alg, reverse=reverse)

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
            - integers       ⇒ INT and SIGNED
            - floating-point ⇒ FLOAT and SIGNED
            - strings        ⇒ COMPATIBILITYNORMALIZE and GROUPLETTERS
            - datetime       ⇒ GROUPLETTERS (only affects 'Z' vs. 'z'; shouldn't matter)

        Args:
            dtype: Probably from ``pd.Series.dtype``

        Returns:
            A tuple of (set of flags, int) -- see :meth:`exact_natsort_alg`
        """
        st, x = set(), 0
        if is_string_dtype(dtype):
            st.update(["COMPATIBILITYNORMALIZE", "GROUPLETTERS"])
            x |= ns_enum.ns.COMPATIBILITYNORMALIZE | ns_enum.ns.GROUPLETTERS
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
            - ``exact_natsort_alg({"REAL"}) == ({"FLOAT", "SIGNED"}, ns.FLOAT | ns.SIGNED)``
            - ``exact_natsort_alg({}) == ({}, 0)``
            - ``exact_natsort_alg(ns.LOWERCASEFIRST) == ({"LOWERCASEFIRST"}, ns.LOWERCASEFIRST)``
            - ``exact_natsort_alg({"localenum", "numafter"})``
              ``== ({"LOCALENUM", "NUMAFTER"}, ns.LOCALENUM | ns.NUMAFTER)``

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
        Returns the names of styles for `tabulate <https://pypi.org/project/tabulate/>`_.
        """
        return _table_formats.keys()

    @classmethod
    def table_format(cls, fmt: str) -> TableFormat:
        """
        Gets a tabulate [1]_ style by name.

        Returns:
            A TableFormat, which can be passed as a style

        References:
            [1] `Tabulate <https://pypi.org/project/tabulate>`_
        """
        return _table_formats[fmt]

    @classmethod
    def plain_table_format(cls, sep: str = " ", **kwargs) -> TableFormat:
        """
        Creates a simple tabulate [1]_ style using a column-delimiter ``sep``.

        Returns:
            A tabulate ``TableFormat``, which can be passed as a style

        References:
            [1] `Tabulate <https://pypi.org/project/tabulate>`_
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
