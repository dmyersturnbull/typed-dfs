"""
Mixin for INI, .properties, and TOML.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Set, Union

import pandas as pd

from typeddfs.df_errors import UnsupportedOperationError
from typeddfs.utils import IoUtils, ParseUtils, Utils


class _IniLikeMixin:
    def to_properties(
        self,
        path_or_buff=None,
        mode: str = "w",
        *,
        comment: Union[None, str, Sequence[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        r"""
        Writes a .properties file.
        Backslashes, colons, spaces, and equal signs are escaped in keys.
        Backslashes are escaped in values.
        The separator is always ``=``.

        .. caution::

            This is provided as a preview. It may have issues and may change.

        Args:
            path_or_buff: Path or buffer
            comment: Comment line(s) to add at the top of the document
            mode: Write ('w') or append ('a')
            kwargs: Passed to :meth:`typeddfs.utils.Utils.write`

        Returns:
            The string data if ``path_or_buff`` is a buffer; None if it is a file
        """
        return self._to_properties_like(
            ParseUtils.property_key_escape,
            ParseUtils.property_value_escape,
            "=",
            "#",
            path_or_buff,
            mode,
            comment,
            **kwargs,
        )

    @classmethod
    def read_properties(
        cls,
        path_or_buff,
        strip_quotes: bool = False,
        **kwargs,
    ) -> __qualname__:
        r"""
        Reads a .properties file.
        Backslashes, colons, spaces, and equal signs are escaped in keys and values.

        .. caution::

            This is provided as a preview. It may have issues and may change.
            It currently does not support continued lines (ending with an odd number of backslashes).

        Args:
            path_or_buff: Path or buffer
            strip_quotes: Remove quotation marks ("") surrounding values
            kwargs: Passed to ``read_csv``; avoid setting
        """
        return cls._read_properties_like(
            ParseUtils.property_key_unescape,
            ParseUtils.property_value_unescape,
            {"!", "#"},
            strip_quotes,
            path_or_buff,
            **kwargs,
        )

    @classmethod
    def read_toml(
        cls, path_or_buff, aot: Optional[str] = "row", aot_only: bool = True, **kwargs
    ) -> __qualname__:
        r"""
        Reads a TOML file.

        .. caution::

            This is provided as a preview. It may have issues and may change.

        Args:
            path_or_buff: Path or buffer
            aot: The name of the array of tables (i.e. ``[[ table ]]``)
                 If None, finds the unique outermost TOML key, implying ``aot_only``.
            aot_only: Fail if any outermost keys other than the AOT are found
            kwargs: Passed to ``Utils.read``
        """
        import tomlkit

        txt = IoUtils.read(path_or_buff, **kwargs)
        data = tomlkit.loads(txt)
        if len(data.keys()) == 0:
            return cls.new_df()
        if aot_only and len(data.keys()) > 1 or aot is None:
            raise ValueError(f"Multiple outermost TOML keys: {data.keys()}")
        if aot is None:
            aot = next(iter(data.keys()))
        data = data[aot]
        df = pd.DataFrame([pd.Series(d) for d in data])
        return cls._convert_typed(df)

    def to_toml(
        self,
        path_or_buff=None,
        aot: str = "row",
        comment: Union[None, str, Sequence[str]] = None,
        mode: str = "w",
        **kwargs,
    ) -> __qualname__:
        r"""
        Writes a TOML file.

        .. caution::

            This is provided as a preview. It may have issues and may change.

        Args:
            path_or_buff: Path or buffer
            aot: The name of the array of tables (i.e. ``[[ table ]]``)
            comment: Comment line(s) to add at the top of the document
            mode: 'w' (write) or 'a' (append)
            kwargs: Passed to :meth:`typeddfs.utils.Utils.write`
        """
        import tomlkit
        from tomlkit.toml_document import TOMLDocument

        comment = [] if comment is None else ([comment] if isinstance(comment, str) else comment)
        df = self.vanilla_reset()
        data = [df.iloc[i].to_dict() for i in range(len(df))]
        aot_obj = ParseUtils.dicts_to_toml_aot(data)
        doc: TOMLDocument = tomlkit.document()
        for c in comment:
            doc.add(tomlkit.comment(c))
        doc[aot] = aot_obj
        txt = tomlkit.dumps(doc)
        return IoUtils.write(path_or_buff, txt, mode=mode, **kwargs)

    @classmethod
    def read_ini(
        cls, path_or_buff, hash_sign: bool = False, strip_quotes: bool = False, **kwargs
    ) -> __qualname__:
        r"""
        Reads an INI file.

        .. caution::

            This is provided as a preview. It may have issues and may change.

        Args:
            path_or_buff: Path or buffer
            hash_sign: Allow ``#`` to denote a comment (as well as ``;``)
            strip_quotes: Remove quotation marks ("" or '') surrounding the values
            kwargs: Passed to :meth:`typeddfs.utils.Utils.read`
        """
        return cls._read_properties_like(
            None,
            None,
            {";", "#"} if hash_sign else {";"},
            strip_quotes,
            path_or_buff,
            **kwargs,
        )

    def to_ini(
        self,
        path_or_buff=None,
        comment: Union[None, str, Sequence[str]] = None,
        mode: str = "w",
        **kwargs,
    ) -> __qualname__:
        r"""
        Writes an INI file.

        .. caution::

            This is provided as a preview. It may have issues and may change.

        Args:
            path_or_buff: Path or buffer
            comment: Comment line(s) to add at the top of the document
            mode: 'w' (write) or 'a' (append)
            kwargs: Passed to :meth:`typeddfs.utils.Utils.write`
        """
        return self._to_properties_like(
            None,
            None,
            "=",
            ";",
            path_or_buff,
            mode,
            comment,
            **kwargs,
        )

    @classmethod
    def _read_properties_like(
        cls,
        unescape_keys,
        unescape_values,
        comment_chars: Set[str],
        strip_quotes: bool,
        path_or_buff,
        **kwargs,
    ) -> __qualname__:
        r"""
        Reads a .properties-like file.
        """
        cls._assert_can_write_properties_class()
        if len(cls.get_typing().required_names) == 2:
            key_col, val_col = cls.get_typing().required_names
        else:
            key_col, val_col = "key", "value"
        txt = Utils.read(path_or_buff, **kwargs)
        keys = []
        values = []
        section = ""
        for i, line in enumerate(txt.splitlines()):
            try:
                line = line.strip()
                if any((line.startswith(c) for c in comment_chars)) or len(line.strip()) == 0:
                    continue
                if line.startswith("["):
                    # treat [ ] (with spaces) as the global key
                    section = line.lstrip("[").rstrip("]").strip()
                    continue
                key, value = line.split("=")
                key, value = key.strip(), value.strip()
                if unescape_keys is not None:
                    key = unescape_keys(key)
                if value.endswith("\\"):
                    raise ValueError("Ends with \\; continued lines are not yet supported")
                if unescape_values is not None:
                    value = unescape_values(value)
                if strip_quotes:
                    value = value.strip('"')
                if section != "":
                    key = section + "." + key
                keys.append(key)
                values.append(value)
            except ValueError:
                raise ValueError(f"Malformed line {i}: '{line}'")
        df = pd.DataFrame({key_col: keys, val_col: values})
        return cls.convert(df)

    def _to_properties_like(
        self,
        escape_keys,
        escape_values,
        sep: str,
        comment_char: str,
        path_or_buff=None,
        mode: str = "w",
        comment: Union[None, str, Sequence[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        r"""
        Writes a .properties-like file.
        """
        comment = [] if comment is None else ([comment] if isinstance(comment, str) else comment)
        self.__class__._assert_can_write_properties_class()
        self._assert_can_write_properties_instance()
        df = self.vanilla_reset()
        if len(self.__class__.get_typing().required_names) == 2:
            key_col, val_col = self.__class__.get_typing().required_names
        else:
            key_col, val_col = "key", "value"
        df.columns = [key_col, val_col]
        df = df.sort_values(key_col)  # essential
        lines = [comment_char.lstrip(comment_char).lstrip() + " " + c for c in comment]
        section = ""
        for k, v in zip(df[key_col], df[val_col]):
            if "." in k:
                k, s = str(k).split(".", 1)
                s, k = k.strip(), s.strip()
                if s != section:
                    lines.append(f"[{s}]")
            if escape_keys:
                k = escape_keys(k)
            if escape_values:
                v = escape_values(v)
            lines.append(k + " " + sep + " " + v.strip('"'))
        return Utils.write(path_or_buff, os.linesep.join(lines), mode=mode, **kwargs)

    def _assert_can_write_properties_instance(self) -> None:
        df = self.vanilla_reset()
        cols = df.columns
        if len(cols) != 2:
            raise UnsupportedOperationError(
                f"Cannot write key/value: {len(cols)} columns != 2: {cols}"
            )

    @classmethod
    def _assert_can_write_properties_class(cls) -> None:
        req_names = cls.get_typing().required_names
        if len(req_names) not in [0, 2]:
            raise UnsupportedOperationError(
                f"Cannot write key/value: {len(req_names)} names: {req_names}"
            )

    @classmethod
    def _properties_files_apply(cls) -> bool:
        # Because we don't write a header, applies IF AND ONLY IF
        # we REQUIRE EXACTLY 2 columns
        return (
            len(cls.get_typing().required_names) == 2
            and not cls.get_typing().more_indices_allowed
            and not cls.get_typing().more_columns_allowed
        )


__all__ = ["_IniLikeMixin"]
