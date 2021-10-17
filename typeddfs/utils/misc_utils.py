"""
Misc tools for typed-dfs.
"""
from __future__ import annotations

import collections
from typing import AbstractSet, Any, Iterator, Mapping, Sequence

import numpy as np

# noinspection PyProtectedMember
from tabulate import DataRow, TableFormat, _table_formats

from typeddfs.frozen_types import FrozeDict, FrozeList, FrozeSet


class MiscUtils:
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


__all__ = ["MiscUtils", "TableFormat"]
