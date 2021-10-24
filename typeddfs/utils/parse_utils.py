"""
Misc tools for typed-dfs.
"""
from __future__ import annotations

import typing
from typing import Any, Generator, Mapping, Sequence, Tuple, TypeVar

import regex

T = TypeVar("T")
_control_chars = regex.compile(r"\p{C}", flags=regex.V1)


class ParseUtils:
    @classmethod
    def strip_control_chars(cls, s: str) -> str:
        """
        Strips all characters under the Unicode 'Cc' category.
        """
        return _control_chars.sub("", s)

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
    def dicts_to_toml_aot(cls, dicts: Sequence[Mapping[str, Any]]):
        """
        Make a tomlkit Document consisting of an array of tables ("AOT").

        Args:
            dicts: A sequence of dictionaries

        Returns:
            A tomlkit`AoT<https://github.com/sdispater/tomlkit/blob/master/tomlkit/items.py>`_
            (i.e. ``[[array]]``)
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
    def _un_leaf(cls, to: typing.MutableMapping[str, Any], items: Mapping[str, Any]) -> None:
        for k, v in items.items():
            if "." not in k:
                to[k] = v
            else:
                k0, k1 = k.split(".", 1)
                if k0 not in to:
                    to[k0] = {}
                cls._un_leaf(to[k0], {k1: v})

    @classmethod
    def _re_leaf(cls, at: str, items: Mapping[str, Any]) -> Generator[Tuple[str, Any], None, None]:
        for k, v in items.items():
            me = at + "." + k if len(at) > 0 else k
            if hasattr(v, "items") and hasattr(v, "keys") and hasattr(v, "values"):
                yield from cls._re_leaf(me, v)
            else:
                yield me, v


__all__ = ["ParseUtils"]
