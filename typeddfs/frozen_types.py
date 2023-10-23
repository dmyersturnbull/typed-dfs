# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
"""
Hashable and ordered collections.
"""
from __future__ import annotations

import functools
from collections.abc import (
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from typing import TypeVar

T_co = TypeVar("T_co", covariant=True)
K_contra = TypeVar("K_contra", contravariant=True)
V_co = TypeVar("V_co", covariant=True)


@functools.total_ordering
class FrozeList(Sequence[T_co], Hashable):
    """
    An immutable list.
    Hashable and ordered.
    """

    EMPTY: FrozeList = NotImplemented  # delayed

    def __init__(self, lst: Sequence[T_co]) -> None:
        self.__lst = lst if isinstance(lst, list) else list(lst)
        try:
            self.__hash = hash(tuple(lst))
        except AttributeError:
            self.__hash = 0

    @property
    def is_empty(self) -> bool:  # pragma: no cover
        return len(self.__lst) == 0

    @property
    def length(self) -> int:  # pragma: no cover
        return len(self.__lst)

    def __iter__(self) -> Iterator[T_co]:  # pragma: no cover
        return iter(self.__lst)

    def __getitem__(self, item: int):  # pragma: no cover
        return self.__lst[item]

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: FrozeList[T_co] | Sequence[T_co]) -> bool:
        return self.__lst == self.__make_other(other)

    def __lt__(self, other: FrozeList[T_co] | Sequence[T_co]):
        return self.__lst < self.__make_other(other)

    def __len__(self) -> int:
        return len(self.__lst)

    def __str__(self) -> str:
        return str(self.__lst)

    def __repr__(self) -> str:
        return repr(self.__lst)

    def to_list(self) -> list[T_co]:
        return list(self.__lst)

    def get(self, item: T_co, default: T_co | None = None) -> T_co | None:
        if item in self.__lst:
            return item
        return default

    def req(self, item: T_co, default: T_co | None = None) -> T_co:
        """
        Returns the requested list item, falling back to a default.
        Short for "require".

        Raise:
            KeyError: If ``item`` is not in this list and ``default`` is ``None``
        """
        if item in self.__lst:
            return item
        if default is None:
            msg = f"Item {item} not found"
            raise KeyError(msg)
        return default

    def __make_other(self, other: FrozeList[T_co] | Sequence[T_co]) -> list[T_co]:
        if isinstance(other, FrozeList):
            other = other.__lst
        if isinstance(other, list):
            return other
        elif isinstance(other, Sequence):
            return list(other)
        msg = f"Cannot compare to {type(other)}"
        raise TypeError(msg)


class FrozeSet(frozenset[T_co], Hashable):
    """
    An immutable set.
    Hashable and ordered.
    This is almost identical to ``typing.FrozenSet``, but it's behavior was made
    equivalent to those of FrozeDict and FrozeList.
    """

    EMPTY: FrozeSet = NotImplemented  # delayed

    def __init__(self, lst: frozenset[T_co]) -> None:
        self.__lst = lst if isinstance(lst, set) else set(lst)
        try:
            self.__hash = hash(tuple(lst))
        except AttributeError:
            # the hashes will collide, making sets slow
            # but at least we'll have a hash and thereby not violate the constraint
            self.__hash = 0

    def get(self, item: T_co, default: T_co | None = None) -> T_co | None:
        if item in self.__lst:
            return item
        return default

    def req(self, item: T_co, default: T_co | None = None) -> T_co:
        """
        Returns ``item`` if it is in this set.
        Short for "require".
        Falls back to ``default`` if ``default`` is not ``None``.

        Raises:
            KeyError: If ``item`` is not in this set and ``default`` is ``None``
        """
        if item in self.__lst:
            return item
        if default is None:
            msg = f"Item {item} not found"
            raise KeyError(msg)
        return default

    def __getitem__(self, item: T_co) -> T_co:
        if item in self.__lst:
            return item
        msg = f"Item {item} not found"
        raise KeyError(msg)

    def __contains__(self, x: T_co) -> bool:  # pragma: no cover
        return x in self.__lst

    def __iter__(self) -> Iterator[T_co]:  # pragma: no cover
        return iter(self.__lst)

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: FrozeSet[T_co]) -> bool:
        return self.__lst == self.__make_other(other)

    def __lt__(self, other: FrozeSet[T_co] | frozenset[T_co]):
        """
        Compares ``self`` and ``other`` for partial ordering.
        Sorts ``self`` and ``other``, then compares the two sorted sets.

        Approximately::
            return list(sorted(self)) < list(sorted(other))
        """
        other = sorted(self.__make_other(other))
        me = sorted(self.__lst)
        return me < other

    @property
    def is_empty(self) -> bool:  # pragma: no cover
        return len(self.__lst) == 0

    @property
    def length(self) -> int:  # pragma: no cover
        return len(self.__lst)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.__lst)

    def __str__(self) -> str:
        return str(self.__lst)

    def __repr__(self) -> str:
        return repr(self.__lst)

    def to_set(self) -> frozenset[T_co]:
        return set(self.__lst)

    def to_frozenset(self) -> frozenset[T_co]:
        return frozenset(self.__lst)

    def __make_other(self, other: FrozeSet[T_co] | frozenset[T_co]) -> set[T_co]:
        if isinstance(other, FrozeSet):
            other = other.__lst
        if isinstance(other, set):
            return other
        elif isinstance(other, frozenset):
            return set(other)
        msg = f"Cannot compare to {type(other)}"
        raise TypeError(msg)


class FrozeDict(Mapping[K_contra, V_co], Hashable):
    """
    An immutable dictionary/mapping.
    Hashable and ordered.
    """

    EMPTY: FrozeDict = NotImplemented  # delayed

    def __init__(self, dct: Mapping[K_contra, V_co]) -> None:
        self.__dct = dct if isinstance(dct, dict) else dict(dct)
        self.__hash = hash(tuple(dct.items()))

    def get(self, key: K_contra, default: V_co | None = None) -> V_co | None:  # pragma: no cover
        return self.__dct.get(key, default)

    def req(self, key: K_contra, default: V_co | None = None) -> V_co:
        """
        Returns the value corresponding to ``key``.
        Short for "require".
        Falls back to ``default`` if ``default`` is not None and ``key`` is not in this dict.

        Raise:
        KeyError: If ``key`` is not in this dict and ``default`` is ``None``
        """
        if default is None:
            return self.__dct[key]
        return self.__dct.get(key, default)

    def items(self) -> frozenset[tuple[K_contra, V_co]]:  # pragma: no cover
        return self.__dct.items()

    def keys(self) -> frozenset[K_contra]:  # pragma: no cover
        return self.__dct.keys()

    def values(self) -> ValuesView[V_co]:  # pragma: no cover
        return self.__dct.values()

    def __iter__(self):  # pragma: no cover
        return iter(self.__dct)

    def __contains__(self, item: K_contra) -> bool:  # pragma: no cover
        return item in self.__dct

    def __getitem__(self, item: K_contra) -> T_co:  # pragma: no cover
        return self.__dct[item]

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: FrozeDict[K_contra, V_co]) -> bool:
        if isinstance(self, FrozeDict):
            return self.__dct == other.__dct
        elif isinstance(self, dict):
            return self == other.__dct
        elif isinstance(self, Mapping):
            return self == dict(other.__dct)
        msg = f"Cannot compare to {type(other)}"
        raise TypeError(msg)

    def __lt__(self, other: Mapping[K_contra, V_co]):
        """
        Compares this dict to another, with partial ordering.

        The algorithm is:
            1. Sort ``self`` and ``other`` by keys
            2. If ``sorted_self < sorted_other``, return ``False``
            3. If the reverse is true (``sorted_other < sorted_self``), return ``True``
            4. (The keys are now known to be the same.)
               For each key, in order: If ``self[key] < other[key]``, return ``True``
            5. Return ``False``
        """
        other = self.__make_other(other)
        me = self.__dct
        o_keys = sorted(other.keys())
        s_keys = sorted(me.keys())
        if o_keys < s_keys:
            return False
        if o_keys > s_keys:
            return True
        # keys are equal
        return any(other[k] > me[k] for k in o_keys)

    @property
    def is_empty(self) -> bool:  # pragma: no cover
        return len(self.__dct) == 0

    @property
    def length(self) -> int:  # pragma: no cover
        return len(self.__dct)

    def __len__(self) -> int:
        return len(self.__dct)

    def __str__(self) -> str:
        return str(self.__dct)

    def __repr__(self) -> str:
        return repr(self.__dct)

    def to_dict(self) -> MutableMapping[K_contra, V_co]:  # pragma: no cover
        return dict(self.__dct)

    def __make_other(
        self,
        other: FrozeDict[K_contra, V_co] | Mapping[K_contra, V_co],
    ) -> dict[K_contra, V_co]:
        if isinstance(other, FrozeDict):
            other = other.__dct
        if isinstance(other, dict):
            return other
        elif isinstance(other, Mapping):
            return dict(other)
        msg = f"Cannot compare to {type(other)}"
        raise TypeError(msg)


# for performance, only make these once:
FrozeList.EMPTY = FrozeList([])
FrozeSet.EMPTY = FrozeSet(set())
FrozeDict.EMPTY = FrozeDict({})


__all__ = ["FrozeList", "FrozeSet", "FrozeDict"]
