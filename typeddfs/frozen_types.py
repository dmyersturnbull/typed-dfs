"""
Hashable and ordered collections.
"""
from __future__ import annotations

import functools
from typing import (
    AbstractSet,
    Dict,
    Hashable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    ValuesView,
)

T = TypeVar("T", covariant=True)
K = TypeVar("K", contravariant=True)
V = TypeVar("V", covariant=True)


@functools.total_ordering
class FrozeList(Sequence[T], Hashable):
    """
    An immutable list.
    Hashable and ordered.
    """

    def __init__(self, lst: Sequence[T]):
        self.__lst = lst if isinstance(lst, list) else list(lst)
        try:
            self.__hash = hash(tuple(lst))
        except AttributeError:
            self.__hash = 0

    def __iter__(self) -> Iterator[T]:  # pragma: no cover
        return iter(self.__lst)

    def __getitem__(self, item: int):  # pragma: no cover
        return self.__lst[item]

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: Union[FrozeList[T], Sequence[T]]) -> bool:
        return self.__lst == self.__make_other(other)

    def __lt__(self, other: Union[FrozeList[T], Sequence[T]]):
        return self.__lst < self.__make_other(other)

    def __len__(self) -> int:
        return len(self.__lst)

    def __str__(self) -> str:
        return str(self.__lst)

    def __repr__(self) -> str:
        return repr(self.__lst)

    def to_list(self) -> List[T]:
        return list(self.__lst)

    def get(self, item: T, default: Optional[T] = None) -> Optional[T]:
        if item in self.__lst:
            return item
        return default

    def req(self, item: T, default: Optional[T] = None) -> T:
        """
        Returns the requested list item, falling back to a default.
        Short for "require".

        Raise:
            KeyError: If ``item`` is not in this list and ``default`` is ``None``
        """
        if item in self.__lst:
            return item
        if default is None:
            raise KeyError(f"Item {item} not found")
        return default

    def __make_other(self, other: Union[FrozeList[T], Sequence[T]]) -> List[T]:
        if isinstance(other, FrozeList):
            other = other.__lst
        if isinstance(other, list):
            return other
        elif isinstance(other, Sequence):
            return list(other)
        raise TypeError(f"Cannot compare to {type(other)}")


class FrozeSet(AbstractSet[T], Hashable):
    """
    An immutable set.
    Hashable and ordered.
    This is almost identical to ``typing.FrozenSet``, but it's behavior was made
    equivalent to those of FrozeDict and FrozeList.
    """

    def __init__(self, lst: AbstractSet[T]):
        self.__lst = lst if isinstance(lst, set) else set(lst)
        try:
            self.__hash = hash(tuple(lst))
        except AttributeError:
            # the hashes will collide, making sets slow
            # but at least we'll have a hash and thereby not violate the constraint
            self.__hash = 0

    def get(self, item: T, default: Optional[T] = None) -> Optional[T]:
        if item in self.__lst:
            return item
        return default

    def req(self, item: T, default: Optional[T] = None) -> T:
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
            raise KeyError(f"Item {item} not found")
        return default

    def __getitem__(self, item: T) -> T:
        if item in self.__lst:
            return item
        raise KeyError(f"Item {item} not found")

    def __contains__(self, x: T) -> bool:  # pragma: no cover
        return x in self.__lst

    def __iter__(self) -> Iterator[T]:  # pragma: no cover
        return iter(self.__lst)

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: FrozeSet[T]) -> bool:
        return self.__lst == self.__make_other(other)

    def __lt__(self, other: Union[FrozeSet[T], AbstractSet[T]]):
        """
        Compares ``self`` and ``other`` for partial ordering.
        Sorts ``self`` and ``other``, then compares the two sorted sets.

        Approximately::
            return list(sorted(self)) < list(sorted(other))
        """
        other = list(sorted(self.__make_other(other)))
        me = list(sorted(self.__lst))
        return me < other

    def __len__(self) -> int:  # pragma: no cover
        return len(self.__lst)

    def __str__(self) -> str:
        return str(self.__lst)

    def __repr__(self) -> str:
        return repr(self.__lst)

    def to_set(self) -> AbstractSet[T]:
        return set(self.__lst)

    def to_frozenset(self) -> AbstractSet[T]:
        return frozenset(self.__lst)

    def __make_other(self, other: Union[FrozeSet[T], AbstractSet[T]]) -> Set[T]:
        if isinstance(other, FrozeSet):
            other = other.__lst
        if isinstance(other, set):
            return other
        elif isinstance(other, AbstractSet):
            return set(other)
        raise TypeError(f"Cannot compare to {type(other)}")


class FrozeDict(Mapping[K, V], Hashable):
    """
    An immutable dictionary/mapping.
    Hashable and ordered.
    """

    def __init__(self, dct: Mapping[K, V]):
        self.__dct = dct if isinstance(dct, dict) else dict(dct)
        self.__hash = hash(tuple(dct.items()))

    def __iter__(self):  # pragma: no cover
        return iter(self.__dct)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:  # pragma: no cover
        return self.__dct.get(key, default)

    def req(self, key: K, default: Optional[V] = None) -> V:
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

    def items(self) -> AbstractSet[tuple[K, V]]:  # pragma: no cover
        return self.__dct.items()

    def keys(self) -> AbstractSet[K]:  # pragma: no cover
        return self.__dct.keys()

    def values(self) -> ValuesView[V]:  # pragma: no cover
        return self.__dct.values()

    def __getitem__(self, item: K) -> T:  # pragma: no cover
        return self.__dct[item]

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: FrozeDict[K, V]) -> bool:
        if isinstance(self, FrozeDict):
            return self.__dct == other.__dct
        elif isinstance(self, dict):
            return self == other.__dct
        elif isinstance(self, Mapping):
            return self == dict(other.__dct)
        raise TypeError(f"Cannot compare to {type(other)}")

    def __lt__(self, other: Mapping[K, V]):
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
        o_keys = list(sorted(other.keys()))
        s_keys = list(sorted(me.keys()))
        if o_keys < s_keys:
            return False
        if o_keys > s_keys:
            return True
        # keys are equal
        return any((other[k] > me[k] for k in o_keys))

    def __len__(self) -> int:
        return len(self.__dct)

    def __str__(self) -> str:
        return str(self.__dct)

    def __repr__(self) -> str:
        return repr(self.__dct)

    def to_dict(self) -> MutableMapping[K, V]:  # pragma: no cover
        return dict(self.__dct)

    def __make_other(self, other: Union[FrozeDict[K, V], Mapping[K, V]]) -> Dict[K, V]:
        if isinstance(other, FrozeDict):
            other = other.__dct
        if isinstance(other, dict):
            return other
        elif isinstance(other, Mapping):
            return dict(other)
        raise TypeError(f"Cannot compare to {type(other)}")


__all__ = ["FrozeList", "FrozeSet", "FrozeDict"]
