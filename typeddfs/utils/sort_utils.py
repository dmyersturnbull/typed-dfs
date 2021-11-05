"""
Tools for sorting.
"""
from __future__ import annotations

import typing
from typing import (
    AbstractSet,
    Any,
    Collection,
    Mapping,
    NamedTuple,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

from natsort import natsorted, ns
from natsort.ns_enum import ns as ns_enum

# noinspection PyProtectedMember
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)

_empty_frozenset = frozenset()

T = TypeVar("T")


class NatsortFlagsAndValue(NamedTuple):
    flags: AbstractSet[str]
    value: int


class SortUtils:
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
            _, alg = cls.guess_natsort_alg(dtype)
        else:
            _, alg = cls.exact_natsort_alg(alg)
        lst = list(lst)
        return natsorted(lst, alg=alg, reverse=reverse)

    @classmethod
    def all_natsort_flags(cls) -> Mapping[str, int]:
        """
        Returns all flags defined by natsort, including combined and default flags.
        Combined flags are, e.g., ``ns_enum.ns.REAL ns_enum.nsFLOAT | ns_enum.ns.SIGNED.``.
        Default flags are, e.g., ``ns_enum.ns.UNSIGNED``.

        See Also:
            :meth:`std_natsort_flags`

        Returns:
            A mapping from flag name to int value
        """
        return {e.name: e.value for e in ns_enum}

    @classmethod
    def core_natsort_flags(cls) -> Mapping[str, int]:
        """
        Returns natsort flags that are not combinations or defaults.

        See Also:
            :meth:`all_natsort_flags`

        Returns:
            A mapping from flag name to int value
        """
        # exclude 0 values -- they're defaults
        # exclude any that are not a power of 2 -- they're combinations
        # len(ns_enum) is more than the number of core vals, but that's fine
        good_vals = {int(2 ** i) for i in range(len(ns_enum))}
        return {e.name: e.value for e in ns_enum if e.value in good_vals}

    @classmethod
    def guess_natsort_alg(cls, dtype: Type[Any]) -> NatsortFlagsAndValue:
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
            x |= ns_enum.COMPATIBILITYNORMALIZE | ns_enum.GROUPLETTERS
        elif is_categorical_dtype(dtype):
            pass
        elif is_integer_dtype(dtype) or is_bool_dtype(dtype):
            st.update(["INT", "SIGNED"])
            x |= ns_enum.INT | ns_enum.SIGNED
        elif is_float_dtype(dtype):
            st.update(["FLOAT", "SIGNED"])
            x |= ns_enum.FLOAT | ns_enum.SIGNED  # same as ns_enum.REAL
        return NatsortFlagsAndValue(st, x)

    @classmethod
    def exact_natsort_alg(
        cls, flags: Union[None, int, Collection[Union[int, str]]]
    ) -> NatsortFlagsAndValue:
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
            or (isinstance(flags, Collection) and len(flags) == 0)
            or (isinstance(flags, int) and flags == 0)
        ):
            return NatsortFlagsAndValue(_empty_frozenset, 0)
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
    def _ns_info_from_int_flag(cls, val: int) -> NatsortFlagsAndValue:
        good = cls.core_natsort_flags()
        st = {k for k, v in good.items() if v & val != 0}
        return NatsortFlagsAndValue(st, val)


__all__ = ["SortUtils"]
