"""
Models for shasum-like files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import (
    AbstractSet,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    ValuesView,
)

import regex

from typeddfs.df_errors import (
    HashContradictsExistingError,
    HashDidNotValidateError,
    HashExistsError,
    HashFileMissingError,
    HashFilenameMissingError,
    PathNotRelativeError,
)
from typeddfs.utils._utils import PathLike

_hex_pattern = regex.compile(r"[A-Ha-h0-9]+", flags=regex.V1)
_hashsum_file_sep = regex.compile(r" [ *]", flags=regex.V1)


@dataclass(frozen=True, repr=True)
class _ChecksumMapping:
    hash_path: Path
    _dct: Mapping[Path, str]

    def __post_init__(self):
        for p in self._dct.keys():
            # will error if it's not
            try:
                p.relative_to(self.directory)
            except ValueError as e:
                raise PathNotRelativeError(f"{e}: Full contents are {self._dct}")

    def lines(self) -> Sequence[str]:
        """
        Returns the text that would be written for this .shasum-like file.
        Calls :meth:`unresolve` first.
        """
        unresolved = self.unresolve()
        return [f"{v} *{p.name}" for p, v in unresolved._dct.items()]

    def line(self, path: PathLike) -> str:
        """
        Returns the text that would be written for a single path in a .shasum-like file.
        """
        path = Path(path)
        v = self._dct[path]
        return f"{v} *{path.name}"

    @property
    def directory(self) -> Path:
        return self.hash_path.parent

    def resolve(self) -> __qualname__:
        """
        Calls ``pathlib.Path.resolve()`` on all paths.
        This will follow symlinks, etc.
        """
        # noinspection PyArgumentList
        return self.__class__(
            self.hash_path.resolve(), {k.resolve(): v for k, v in self._dct.items()}
        )

    def unresolve(self) -> __qualname__:
        """
        Each path becomes its filename under ``self.directory``.
        This means that the parent nodes of each path are discarded in favor of ``self.directory``.

        Raises:
            ValueError: If two paths are "un-resolved" to the same path

        .. note::
            This will not work correctly with subdirectories
        """
        dct = {}
        for k, v in self._dct.items():
            k = self.directory / k.name
            if k in dct:
                raise ValueError(f"At least 2 paths resolve to {k}")
            dct[k] = v
        # noinspection PyArgumentList
        return self.__class__(self.hash_path, dct)

    @classmethod
    def _parse(
        cls,
        path: Path,
        *,
        lines: Optional[Sequence[str]] = None,
        missing_ok: bool = False,
        subdirs: bool = False,
    ) -> __qualname__:
        path = Path(path)
        if lines is None and path.exists():
            lines = path.read_text(encoding="utf8").splitlines()
        elif missing_ok and lines is None:
            lines = []
        elif lines is None:
            raise HashFileMissingError(f"Hash file {path} not found")

        # ignore spaces -- editors often add an extra line break, and it's probably fine anyway
        read = [_hashsum_file_sep.split(s, 1) for s in lines if len(s) > 0]
        # obviously this means that / can't appear in a node
        # this is consistent with the commonly accepted spec for shasum
        # does not handle root (beginning with /)
        kv = {
            Path(*[n for n in r[1].strip().split("/") if n != "."]): r[0]
            for r in read
            if len(r[0]) != 0
        }
        if not subdirs:
            slashed = {k for k in kv.keys() if len(k.parts) > 1}
            if len(slashed) > 0:
                raise ValueError(f"Subdirectory (containing /): {slashed} in {path}")
        kv = {Path(path.parent, p): v for p, v in kv.items()}
        return cls(path, kv)

    def _get_updated(
        self,
        *,
        path: PathLike,
        new: Optional[str],
        overwrite: Optional[bool],
        missing_ok: bool,
    ) -> Optional[str]:
        path = Path(path)
        z = self._dct.get(path)
        if z is None and not missing_ok:
            raise HashFilenameMissingError(f"{path} not found ({len(self._dct)} are)")
        if z is not None:
            err = None
            if overwrite is None and z != new:
                err = (HashContradictsExistingError, f"Hash for {path} exists but does not match")
            elif overwrite is False:
                err = (
                    HashExistsError,
                    f"Hash for {path} exists ({'matches' if z == new else 'differs'})",
                )
            if err is not None:
                raise err[0](err[1], key=str(path), original=z, new=new)
        return new


@dataclass(frozen=True, repr=True)
class ChecksumFile(_ChecksumMapping):
    def load(self) -> __qualname__:
        """
        Returns a copy of ``self`` read from :attr:`hash_path`.
        """
        return self.__class__.parse(self.hash_path)

    @classmethod
    def parse(
        cls,
        path: Path,
        *,
        lines: Optional[Sequence[str]] = None,
    ) -> __qualname__:
        """
        Reads hash file contents.

        Args:
            path: The path of the checksum file; required to resolve paths relative to its parent
            lines: The lines in the checksum file; reads ``path`` if None

        Returns:
            A ChecksumFile
        """
        return cls._parse(path, lines=lines, missing_ok=False, subdirs=False)

    @classmethod
    def new(
        cls,
        hash_path: PathLike,
        file_path: PathLike,
        hash_value: str,
    ) -> ChecksumFile:
        """
        Use this as a constructor.
        """
        hash_path = Path(hash_path)
        return cls(hash_path, {Path(file_path): hash_value})

    def rename(self, path: Path) -> __qualname__:
        """
        Replaces :attr:`self.file_path` with ``path``.
        This will affect the filename written in a .shasum-like file.
        No OS operations are performed.
        """
        return self.new(self.hash_path, file_path=path, hash_value=self.hash_value)

    def update(self, value: str, overwrite: Optional[bool] = True) -> __qualname__:
        """
        Modifies the hash.

        Args:
            value: The new hex-encoded hash
            overwrite: If ``None``, requires that the value is the same as before
                       (no operation is performed).
                       If ``False``, this method will always raise an error.
        """
        x = self._get_updated(
            path=self.file_path,
            new=value,
            missing_ok=False,
            overwrite=overwrite,
        )
        return self.new(self.hash_path, file_path=self.file_path, hash_value=x)

    def delete(self) -> None:
        """
        Deletes the hash file by calling ``pathlib.Path.unlink(self.hash_path)``.

        Raises:
            OSError: Accordingly
        """
        self.hash_path.unlink(missing_ok=True)

    def write(self) -> None:
        """
        Writes the hash file.

        Raises:
            OsError: Accordingly
        """
        self.directory.mkdir(exist_ok=True, parents=True)
        self.hash_path.write_text("\n".join(self.lines()), encoding="utf8")

    @property
    def file_path(self) -> Path:
        if len(self._dct) != 1:
            raise AssertionError(f"{self.hash_path} contains {len(self._dct)} (!= 1) items")
        return next(iter(self._dct.keys()))

    @property
    def hash_value(self) -> str:
        if len(self._dct) != 1:
            raise AssertionError(f"{self.hash_path} contains {len(self._dct)} (!= 1) items")
        return next(iter(self._dct.values()))

    def verify(self, computed: str) -> None:
        """
        Verifies the checksum.

        Args:
            computed: A pre-computed hex-encoded hash

        Raises:
            HashDidNotValidateError: If the hashes are not equal
        """
        if computed != self.hash_value:
            raise HashDidNotValidateError(
                f"Hash for {self.file_path}: calculated {computed} != expected {self.hash_value}",
                actual=computed,
                expected=self.hash_value,
            )


@dataclass(frozen=True, repr=True)
class ChecksumMapping(_ChecksumMapping):
    def load(self, missing_ok: bool = False) -> __qualname__:
        """
        Replaces this map with one read from the hash file.

        Args:
            missing_ok: If the hash path does not exist, treat it has having no items
        """
        return self.__class__.parse(self.hash_path, missing_ok=missing_ok)

    @classmethod
    def parse(
        cls,
        path: Path,
        *,
        lines: Optional[Sequence[str]] = None,
        missing_ok: bool = False,
        subdirs: bool = False,
    ) -> __qualname__:
        """
        Reads hash file contents.

        Args:
            path: The path of the checksum file; required to resolve paths relative to its parent
            lines: The lines in the checksum file; reads ``path`` if None
            missing_ok: If ``path`` does not exist, assume it contains no items
            subdirs: Permit files within subdirectories specified with ``/``
                     Most tools do not support these.

        Returns:
            A mapping from raw string filenames to their hex hashes.
            Any node called ``./`` in the path is stripped.
        """
        return cls._parse(path, lines=lines, missing_ok=missing_ok, subdirs=subdirs)

    @classmethod
    def new(
        cls,
        hash_path: PathLike,
        dct: Mapping[PathLike, str],
    ) -> ChecksumMapping:
        """
        Use this as the constructor.
        """
        hash_path = Path(hash_path)
        return cls(hash_path, {Path(k): v for k, v in dct.items()})

    def write(
        self,
        *,
        sort: Union[bool, Callable[[Sequence[Path]], Sequence[Path]]] = False,
        rm_if_empty: bool = False,
    ) -> None:
        """
        Writes to the hash (.shasum-like) file.

        Args:
            sort: Sort with this function, or ``sorted`` if True
            rm_if_empty: Delete with ``pathlib.Path.unlink`` if this contains no items

        Raises:
            OSError: Accordingly
        """
        if sort is True:
            sort = sorted
        if rm_if_empty and len(self._dct) == 0:
            self.hash_path.unlink(missing_ok=True)
        else:
            lines = self.lines()
            if callable(sort):
                lines = sort(lines)
            self.directory.mkdir(exist_ok=True, parents=True)
            self.hash_path.write_text("\n".join(lines), encoding="utf8")

    @property
    def entries(self) -> Mapping[Path, str]:
        return dict(self._dct)

    def keys(self) -> AbstractSet[Path]:
        return self._dct.keys()

    def values(self) -> ValuesView[str]:
        return self._dct.values()

    def items(self) -> AbstractSet[Tuple[Path, str]]:
        return self._dct.items()

    def get(self, key: Path, default: Optional[str] = None) -> Optional[str]:
        return self._dct.get(key, default)

    def __contains__(self, path: Path) -> bool:
        return path in self._dct

    def __getitem__(self, path: Path) -> str:
        return self._dct[path]

    def __len__(self) -> int:
        return len(self._dct)

    def __add__(
        self, other: Union[ChecksumMapping, Mapping[PathLike, str], __qualname__]
    ) -> __qualname__:
        """
        Performs a symmetric addition.

        Raises:
            ValueError: If ``other`` intersects (shares keys) with ``self``

        See Also:
            :meth:`append`
        """
        if isinstance(other, ChecksumMapping):
            other = other._dct
        other = {Path(k): v for k, v in other.items()}
        intersection = set(self._dct).intersection(other)
        if len(intersection) > 0:
            raise ValueError(f"Cannot merge with intersection: {intersection}")
        return ChecksumMapping(self.hash_path, {**self, **other})

    def __sub__(self, other: Union[PathLike, Iterable[PathLike], ChecksumMapping]) -> __qualname__:
        """
        Removes entries.

        See Also:
            :meth:`remove`
        """
        if isinstance(other, ChecksumMapping):
            other = other._dct
        if isinstance(other, (PurePath, str)):
            other = {other}
        other = {Path(p) for p in other}
        return self.new(self.hash_path, {k: v for k, v in self.items() if k not in other})

    def remove(
        self, remove: Union[PathLike, Iterable[PathLike]], *, missing_ok: bool = False
    ) -> __qualname__:
        """
        Strips paths from this hash collection.
        Like :meth:`update` but less flexible and only for removing paths.

        Raises:
            :class:`typeddfs.df_errors.PathNotRelativeError`: To avoid, try calling ``resolve`` first
        """
        if isinstance(remove, (str, PurePath)):
            remove = [remove]
        return self.update({p: None for p in remove}, missing_ok=missing_ok, overwrite=True)

    def append(
        self, append: Mapping[PathLike, str], *, overwrite: Optional[bool] = False
    ) -> __qualname__:
        """
        Append paths to a dir hash file.
        Like :meth:`update` but less flexible and only for adding paths.
        """
        return self.update(append, missing_ok=True, overwrite=overwrite)

    def update(
        self,
        update: Union[Callable[[Path], Optional[PathLike]], Mapping[PathLike, Optional[PathLike]]],
        *,
        missing_ok: bool = True,
        overwrite: Optional[bool] = True,
    ) -> __qualname__:
        """
        Returns updated hashes from a dir hash file.

        Args:
            update: Values to overwrite.
                    May be a function or a dictionary from paths to values.
                    If ``None`` is returned, the entry will be removed;
                    otherwise, updates with the returned hex hash.
            missing_ok: Require that the path is already listed
            overwrite: Allow overwriting an existing value.
                       If ``None``, only allow if the hash is the same.
        """
        fixed = {}
        # update existing items:
        for p, v in self.items():
            v_new = update(p) if callable(update) else update.get(p, v)
            if v == v_new:
                # avoid an error about overwriting if we're not changing values
                fixed[p] = v
            else:
                fixed[p] = self._get_updated(
                    path=p,
                    new=v_new,
                    missing_ok=missing_ok,
                    overwrite=overwrite,
                )
        # add new items:
        if not callable(update):
            for p, v in update.items():
                p = Path(p)
                fixed[p] = self._get_updated(
                    path=p,
                    new=v,
                    missing_ok=missing_ok,
                    overwrite=overwrite,
                )
        fixed = {k: v for k, v in fixed.items() if v is not None}
        return self.new(self.hash_path, fixed)

    def verify(
        self, path: PathLike, computed: str, *, resolve: bool = False, exist: bool = False
    ) -> None:
        """
        Verifies a checksum.
        The file ``path`` must be listed.

        Args:
            path: The file to look for
            computed: A pre-computed hex-encoded hash; if set, do not calculate from ``path``
            resolve: Resolve paths before comparison
            exist: Require that ``path`` exists

        Raises:
            FileNotFoundError: If ``path`` does not exist
            HashFileMissingError: If the hash file does not exist
            HashDidNotValidateError: If the hashes are not equal
            HashVerificationError`: Superclass of ``HashDidNotValidateError`` if
                                    the filename is not listed, etc.
        """
        path = Path(path)
        if resolve:
            path = path.resolve()
        elif not path.is_absolute():
            path = self.directory / path
        if exist and not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        found = self.get(path)
        if found is None:
            raise FileNotFoundError(f"Path {path} not listed in {self.hash_path}")
        if computed != found:
            raise HashDidNotValidateError(
                f"Hash for {path}: calculated {computed} != expected {found}",
                actual=computed,
                expected=found,
            )


__all__ = ["ChecksumFile", "ChecksumMapping"]
