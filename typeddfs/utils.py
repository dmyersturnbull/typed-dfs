"""
Tools that could possibly be used outside of typed-dfs.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import os
import re
import sys
from pathlib import Path, PurePath
from typing import (
    Optional,
    Sequence,
    Mapping,
    Set,
    Union,
    Any,
    ValuesView,
    TypeVar,
    List,
    Iterator,
    AbstractSet,
    Dict,
    Type,
    Tuple,
    Collection,
)

# noinspection PyProtectedMember
from pandas.api.types import is_integer_dtype, is_float_dtype, is_bool_dtype, is_string_dtype
from natsort import ns_enum
import numpy as np
from pandas.io.common import get_handle

# noinspection PyProtectedMember
from tabulate import DataRow, TableFormat, _table_formats

from typeddfs._utils import _DEFAULT_HASH_ALG, _AUTO_DROPPED_NAMES, _FORBIDDEN_NAMES
from typeddfs.df_errors import (
    HashFilenameMissingError,
    MultipleHashFilenamesError,
    HashFileMissingError,
    HashContradictsExistingError,
    HashFileExistsError,
    HashAlgorithmMissingError,
    HashDidNotValidateError,
)


_hex_pattern = re.compile(r"[A-Ha-h0-9]+")
_hashsum_file_sep = re.compile(r" [ *]")
T = TypeVar("T", covariant=True)
K = TypeVar("K", contravariant=True)
V = TypeVar("V", covariant=True)


@functools.total_ordering
class FrozeList(Sequence[T]):
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

    def __iter__(self) -> Iterator[T]:
        return iter(self.__lst)

    def __getitem__(self, item: int):
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

    def __make_other(self, other: Union[FrozeList[T], Sequence[T]]) -> List[T]:
        if isinstance(other, FrozeList):
            other = other.__lst
        if isinstance(other, list):
            return other
        elif isinstance(other, Sequence):
            return list(other)
        raise TypeError(f"Cannot compare to {type(other)}")


class FrozeSet(AbstractSet[T]):
    """
    An immutable set.
    Hashable.
    This is almost identical to ``typing.FrozenSet``, but it's behavior was made
    equivalent to those of FrozeDict and FrozeList.
    """

    def __init__(self, lst: AbstractSet[T]):
        self.__lst = lst if isinstance(lst, set) else set(lst)
        try:
            self.__hash = hash(tuple(lst))
        except AttributeError:
            self.__hash = 0

    def get(self, item: T, default: Optional[T]) -> Optional[T]:
        if item in self.__lst:
            return item
        return default

    def __getitem__(self, item: T) -> T:
        if item in self.__lst:
            return item
        raise KeyError(f"Item {item} not found")

    def __contains__(self, x: T) -> bool:
        return x in self.__lst

    def __iter__(self) -> Iterator[T]:
        return iter(self.__lst)

    def __hash__(self) -> int:
        return self.__hash

    def __eq__(self, other: FrozeSet[T]) -> bool:
        return self.__lst == self.__make_other(other)

    def __len__(self) -> int:
        return len(self.__lst)

    def __str__(self) -> str:
        return str(self.__lst)

    def __repr__(self) -> str:
        return repr(self.__lst)

    def to_set(self) -> AbstractSet[T]:
        return set(self.__lst)

    def __make_other(self, other: Union[FrozeSet[T], AbstractSet[T]]) -> Set[T]:
        if isinstance(other, FrozeSet):
            other = other.__lst
        if isinstance(other, set):
            return other
        elif isinstance(other, AbstractSet):
            return set(other)
        raise TypeError(f"Cannot compare to {type(other)}")


class FrozeDict(Mapping[K, V]):
    """
    An immutable dictionary/mapping.
    Hashable.
    """

    def __init__(self, dct: Mapping[K, V]):
        self.__dct = dct if isinstance(dct, dict) else dict(dct)
        self.__hash = hash(tuple(dct.items()))

    def __iter__(self):
        return iter(self.__dct)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self.__dct.get(key)

    def req(self, key: K, default: Optional[V] = None) -> V:
        if default is None:
            return self.__dct[key]
        return self.__dct.get(key, default)

    def items(self) -> AbstractSet[tuple[K, V]]:
        return self.__dct.items()

    def keys(self) -> AbstractSet[K]:
        return self.__dct.keys()

    def values(self) -> ValuesView[V]:
        return self.__dct.values()

    def __getitem__(self, item: K) -> T:
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

    def __len__(self) -> int:
        return len(self.__dct)

    def to_dict(self) -> Mapping[K, V]:
        return dict(self.__dct)

    def __make_other(self, other: Union[FrozeDict[K, V], Mapping[K, V]]) -> Dict[K, V]:
        if isinstance(other, FrozeDict):
            other = other.__dct
        if isinstance(other, dict):
            return other
        elif isinstance(other, Mapping):
            return dict(other)
        raise TypeError(f"Cannot compare to {type(other)}")


class Utils:
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
            a :py.type:`typeddfs.utils.FrozeSet` (subclass of :py.type:`typing.AbstractSet`),
            a :py.type:`typeddfs.utils.FrozeList` (subclass of :py.type:`typing.Sequence`),
            or a :py.type:`typeddfs.utils.FrozeDict` (subclass of :py.type:`typing.Mapping`).
            int, float, str, np.generic, and tuple are always returned as-is.

        Raises:
            AttributeError: If ``v`` is not hashable and could not converted to
                            a FrozeSet, FrozeList, or FrozeDict, *or* if one of the elements for
                            one of the above types is not hashable.
            TypeError: If ``v`` is a :py.type:`iterator.Iterator` or :py.type:`collections.deque`
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
            return frozenset(v)
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
    def banned_names(cls) -> Set[str]:
        """
        Lists strings that cannot be used for column names or index level names.
        """
        return {*_AUTO_DROPPED_NAMES, *_FORBIDDEN_NAMES}

    @classmethod
    def write(cls, path_or_buff, content, mode: str = "w", **kwargs) -> Optional[str]:
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
    def read(cls, path_or_buff, content, mode: str = "w", **kwargs) -> Optional[str]:
        """
        Reads using Pandas's ``get_handle``.
        By default (unless ``compression=`` is set), infers the compression type from the filename suffix
        (e.g. ``.csv.gz``).
        """
        kwargs = {**dict(compression="infer"), **kwargs}
        if path_or_buff is None:
            return content
        with get_handle(path_or_buff, mode, **kwargs) as f:
            f.handle.write(content)

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
            - paths          ==> PATH, COMPATIBILITYNORMALIZE, and GROUPLETTERS

        Args:
            dtype: Probably from ``pd.Series.dtype``

        Returns:
            A tuple of (set of flags, int) -- see :py.meth:`exact_natsort_alg`
        """
        st, x = set(), 0
        if is_string_dtype(dtype):
            st.update(["COMPATIBILITYNORMALIZE", "GROUPLETTERS"])
            x |= ns_enum.ns.COMPATIBILITYNORMALIZE | ns_enum.ns.GROUPLETTERS
        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
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
        if (
            flags is None
            or isinstance(flags, Collection)
            and len(flags) == 0
            or isinstance(flags, int)
            and flags == 0
        ):
            return set(), 0
        if isinstance(flags, int):
            st = set()
            for f, v in ns_enum.enum_fields.items():
                if f in ns_enum.enum_fields and (v & flags) != 0:
                    st.add(f)
            return st, flags
        elif isinstance(flags, Collection):
            a = 0
            for f in flags:
                if isinstance(f, str):
                    f = ns_enum.enum_fields[f.upper()]
                a |= f
            return set(flags), a
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
    def add_any_hashes(
        cls,
        path: Path,
        to_file: bool,
        to_dir: bool,
        algorithm: str = _DEFAULT_HASH_ALG,
        overwrite: bool = True,
    ) -> Optional[str]:
        """
        Adds and/or appends the hex hash of ``path``.
        """
        algorithm = cls.get_algorithm(algorithm)
        hash_file_path = cls.get_hash_file(path, algorithm)
        hash_dir_path = cls.get_hash_dir(path, algorithm)
        if to_file and hash_file_path.exists() and not overwrite:  # check first -- save time
            raise HashFileExistsError(f"Hash file {path} already exists", key=str(path))
        if not to_file and not to_dir:
            return None
        digest = cls.calc_hash(path, algorithm)
        if to_file:
            cls._add_file_hash(path, hash_file_path, digest, overwrite)
        if to_dir:
            cls._append_dir_hash(path, hash_dir_path, digest, overwrite)
        return digest

    @classmethod
    def add_file_hash(
        cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG, overwrite: bool = True
    ) -> str:
        """
        Calculates the hash of ``path`` and adds it to a file ``path+"."+alg``.

        Args:
            path: The path to a file to compute the hash of (in binary mode)
            algorithm: The name of the hashlib algorithm
            overwrite: If False, error if the hash file already exists

        Returns:
            The hex-encoded hash
        """
        algorithm = cls.get_algorithm(algorithm)
        hash_path = path.with_suffix(path.suffix + f".{algorithm}")
        if hash_path.exists() and not overwrite:  # check first -- save time
            raise HashFileExistsError(f"Hash file {path} already exists", key=str(path))
        digest = cls.calc_hash(path)
        cls._add_file_hash(path, hash_path, digest, overwrite)
        return digest

    @classmethod
    def append_dir_hash(
        cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG, overwrite: bool = True
    ) -> str:
        """
        Calculates the hash of ``path`` and appends it to a file ``dir/(dir+"."+alg)``.

        Args:
            path: The path to a file to compute the hash of (in binary mode)
            algorithm: The name of the hashlib algorithm
            overwrite: If False, error if the hash is already listed and differs

        Returns:
            The hex-encoded hash
        """
        path = Path(path)
        hash_path = cls.get_hash_dir(path, algorithm)
        digest = cls.calc_hash(path)
        cls._append_dir_hash(path, hash_path, digest, overwrite)
        return digest

    @classmethod
    def verify_any_hash(
        cls, path: Path, how: Union[bool, str, PurePath], algorithm: str = _DEFAULT_HASH_ALG
    ) -> None:
        """
        See :py.meth:`typeddfs.abs_dfs.AbsDf.write_file`.
        """
        origin_how = str(how)
        f_path = cls.get_hash_file(path, algorithm=algorithm)
        d_path = cls.get_hash_dir(path, algorithm=algorithm)
        if how is True or how == "yes" and f_path.exists():
            how = "file"
        elif how is True or how == "yes" and d_path.exists():
            how = "dir"
        elif how is True or how == "yes":
            how = "file"  # we'll fail later
        elif how is False or how == "false":
            how = "no"
        elif how == "dir" or how == "directory":
            how = "dir"
        elif how == "file":
            how = "file"
        elif isinstance(how, str) and _hex_pattern.fullmatch(how):
            cls.verify_hash_from_hex(path, how, algorithm=algorithm)
            how = "no"
        elif isinstance(how, (str, PurePath)):
            cls.verify_hash_from_file(path, Path(how), algorithm=algorithm)
            how = "no"
        else:
            raise ValueError(f"Option check_hash={origin_how} not understood or impossible")
        if how == "file":
            cls.verify_file_hash(path, algorithm=algorithm)
        if how == "dir":
            cls.verify_dir_hash(path, algorithm=algorithm)

    @classmethod
    def verify_hash_from_hex(
        cls, path: Path, expected: str, algorithm: str = _DEFAULT_HASH_ALG
    ) -> Optional[str]:
        """
        Verifies a hash directly from a hex string.
        """
        algorithm = cls.get_algorithm(algorithm)
        actual = cls.calc_hash(path, algorithm=algorithm)
        if actual != expected:
            raise HashDidNotValidateError(
                f"Hash for {path}: calculated {actual} != expected {expected}"
            )
        return actual

    @classmethod
    def verify_hash_from_file(
        cls, path: Path, hash_path: Path, algorithm: Optional[str] = _DEFAULT_HASH_ALG
    ) -> Optional[str]:
        """
        Verifies a hash directly from a specific hash file.
        The hash file should contain only filename, which is ignored.
        If there are multiple filenames, it will use the one for ``path.name``.
        """
        if algorithm is None:
            algorithm = cls.guess_algorithm(hash_path)
        else:
            algorithm = cls.get_algorithm(algorithm)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found")
        actual = cls.calc_hash(path, algorithm=algorithm)
        cls._verify_file_hash(path, hash_path, actual, use_filename=None)
        return actual

    @classmethod
    def verify_file_hash(
        cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG, use_filename: Optional[bool] = None
    ) -> str:
        """
        Verifies a file against is corresponding hash file.
        The hash file should contain only filename, which is ignored.
        If there are multiple filenames, it will use the one for ``path.name``.

        Args:
            path: The file to calculate the (binary mode) hash of
            algorithm: The algorithm in hashlib
            use_filename: If True, require the filename in the hash file to match ``path.name``;
                          If False, ignore the filename but require exactly 1 filename listed
                          If None, use either the single filename or the one for path.name.
        """
        hash_path = cls.get_hash_file(path, algorithm)
        algorithm = cls.get_algorithm(algorithm)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found")
        actual = cls.calc_hash(path, algorithm=algorithm)
        cls._verify_file_hash(path, hash_path, actual, use_filename)
        return actual

    @classmethod
    def verify_dir_hash(cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG) -> str:
        """
        Verifies a file against is corresponding per-directory hash file.
        The filename ``path.name`` must be listed in the file.

        Args:
            path: The file to calculate the (binary mode) hash of
            algorithm: The algorithm in hashlib
        """
        hash_path = cls.get_hash_dir(path, algorithm)
        algorithm = cls.get_algorithm(algorithm)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found")
        actual = cls.calc_hash(path, algorithm=algorithm)
        cls._verify_file_hash(path, hash_path, actual, True)
        return actual

    @classmethod
    def calc_hash(cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG) -> str:
        """
        Calculates the hash of a file and returns it, hex-encoded.
        """
        algorithm = cls.get_algorithm(algorithm)
        alg = getattr(hashlib, algorithm)()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024), b""):
                alg.update(chunk)
        return alg.hexdigest()

    @classmethod
    def parse_hash_file(cls, path: Path) -> Mapping[Path, str]:
        """
        Reads a hash file.

        Returns:
            A mapping from resolved Paths to their hex hashes
        """
        read = path.read_text(encoding="utf8").splitlines()
        read = [_hashsum_file_sep.split(s, 1) for s in read]
        # obviously this means that / can't appear in a node
        # this is consistent with the commonly accepted spec for shasum
        return {
            Path(path.parent, *r[1].split("/")).resolve(): r[0].strip()
            for r in read
            if len(r[1]) != 0
        }

    @classmethod
    def get_hash_file(cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG) -> Path:
        """
        Returns the path required for the per-file hash of ``path``.

        Example:
            Utils.get_hash_file("my_file.txt.gz")  # Path("my_file.txt.gz.sha256")
        """
        algorithm = cls.get_algorithm(algorithm)
        return path.with_suffix(path.suffix + "." + algorithm)

    @classmethod
    def get_hash_dir(cls, path: Path, algorithm: str = _DEFAULT_HASH_ALG) -> Path:
        """
        Returns the path required for the per-file hash of ``path``.

        Example:
            Utils.get_hash_file(Path("my_dir, my_file.txt.gz"))  # Path("my_dir", "my_dir.sha256")
        """
        algorithm = cls.get_algorithm(algorithm)
        return path.parent / (path.parent.name + "." + algorithm)

    @classmethod
    def guess_algorithm(cls, path: Path) -> str:
        """
        Guesses the hashlib algorithm used from a hash file.

        Example:
            Utils.guess_algorithm("my_file.sha1")  # "sha1"
        """
        algorithm = path.suffix.lstrip(".").lower().replace("-", "")
        try:
            getattr(hashlib, algorithm)
        except AttributeError:
            raise HashAlgorithmMissingError(f"No hashlib algorithm {algorithm}") from None
        return algorithm

    @classmethod
    def get_algorithm(cls, algorithm: str) -> str:
        """
        Finds a hash algorithm by name in :py.mod:`hashlib`.
        Converts to lowercase and removes hyphens.

        Raises:
            HashAlgorithmMissingError: If not found
        """
        algorithm = algorithm.lower().replace("-", "")
        try:
            getattr(hashlib, algorithm)
        except AttributeError:
            raise HashAlgorithmMissingError(f"No hashlib algorithm {algorithm}") from None
        return algorithm

    @classmethod
    def _add_file_hash(cls, path: Path, hash_path: Path, digest: str, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise HashFileExistsError(f"Hash file {path} already exists")
        path = Path(path)
        txt = f"{digest} *{path.name}"
        hash_path.write_text(txt, encoding="utf-8")

    @classmethod
    def _append_dir_hash(cls, path: Path, hash_path: Path, digest: str, overwrite: bool) -> None:
        txt = f"{digest} *{path.name}"
        if hash_path.exists():
            existing = cls.parse_hash_file(hash_path)
            z = existing.get(path.resolve())
            if z is not None and z != digest and not overwrite:
                raise HashContradictsExistingError(
                    f"Hash for {path} already exists in {hash_path} but does not match"
                )
        else:
            with hash_path.open(mode="a", encoding="utf-8") as f:
                f.write(txt)

    @classmethod
    def _verify_file_hash(
        cls, path: Path, hash_path: Path, actual: str, use_filename: Optional[bool]
    ) -> None:
        path = Path(path)
        hashes = cls.parse_hash_file(hash_path)
        resolved = path.resolve()
        if (use_filename is None or use_filename is True) and resolved in hashes:
            expected = hashes[resolved]
        elif not (use_filename is None or use_filename is False) and len(hashes) == 1:
            expected = next(iter(hashes.values()))
        elif len(hashes) > 1:
            raise MultipleHashFilenamesError(f"{hash_path} contains multiple filenames")
        elif len(hashes) == 0:
            raise HashFilenameMissingError(f"{path} not found in {hash_path} (no hashes listed)")
        else:
            raise HashFilenameMissingError(f"{path} not found in {hash_path}")
        if actual != expected:
            raise HashDidNotValidateError(
                f"Hash for {path} from {hash_path}: calculated {actual} != expected {expected}"
            )


__all__ = ["Utils", "TableFormat", "FrozeList", "FrozeDict", "FrozeSet"]
