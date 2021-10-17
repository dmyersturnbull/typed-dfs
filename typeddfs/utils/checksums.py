"""
Tools for shasum-like files.
"""
from __future__ import annotations

import hashlib
import os
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import regex

from typeddfs.df_errors import (
    HashAlgorithmMissingError,
    HashContradictsExistingError,
    HashDidNotValidateError,
    HashExistsError,
    HashFileExistsError,
    HashFileMissingError,
    HashFilenameMissingError,
    MultipleHashFilenamesError,
)
from typeddfs.utils._utils import _DEFAULT_HASH_ALG, PathLike

_hex_pattern = regex.compile(r"[A-Ha-h0-9]+", flags=regex.V1)
_hashsum_file_sep = regex.compile(r" [ *]", flags=regex.V1)


class ChecksumMapping(UserDict[Path, str]):
    def __getitem__(self, path: Path) -> str:
        path = path.resolve()
        return super().__getitem__(path)

    @property
    def lines(self) -> Sequence[str]:
        return [self.line(p) for p in self.keys()]

    def line(self, path: PathLike) -> str:
        path = Path(path)
        v = self[path]
        return f"{v} *{path.name}"


class ChecksumMappingOpt(UserDict[Path, Optional[str]]):
    def __getitem__(self, path: Path) -> Optional[str]:
        path = path.resolve()
        return super().__getitem__(path)

    @property
    def lines(self) -> Sequence[str]:
        return [self.line(p) for p in self.keys() if self[p] is not None]

    def line(self, path: PathLike) -> Optional[str]:
        path = Path(path)
        v = self.get(path)
        if v is None:
            return None
        return f"{v} *{path.name}"


@dataclass(frozen=True, repr=True, order=True)
class Checksums:
    alg: str = _DEFAULT_HASH_ALG

    def default_algorithm(self) -> str:
        return _DEFAULT_HASH_ALG

    def add_any_hashes(
        self,
        path: PathLike,
        *,
        to_file: bool,
        to_dir: bool,
        overwrite: Optional[bool] = True,
    ) -> Optional[str]:
        """
        Adds and/or appends the hex hash of ``path``.

        Args:
            path: Path to the file to hash
            to_file: Whether to save a per-file hash
            to_dir: Whether to save a per-dir hash
            overwrite: If True, overwrite the file hash and any entry in the dir hash.
                       If False, never overwrite either.
                       If None, never overwrite, but ignore if equal to any existing entries.
        """
        path = Path(path)
        hash_file_path = self.get_hash_file(path)
        hash_dir_path = self.get_hash_dir(path)
        if to_file and hash_file_path.exists() and overwrite is False:  # check first -- save time
            raise HashFileExistsError(f"Hash file {path} already exists", key=str(path))
        if not to_file and not to_dir:
            return None
        digest = self.calc_hash(path)
        if to_file:
            self._add_file_hash(path, hash_file_path, digest, overwrite=overwrite, dry_run=False)
        if to_dir:
            self.append_dir_hashes(hash_dir_path, {path: digest}, overwrite=overwrite)
        return digest

    def add_file_hash(self, path: PathLike, *, overwrite: bool = True) -> str:
        """
        Calculates the hash of ``path`` and adds it to a file ``path+"."+alg``.

        Args:
            path: The path to a file to compute the hash of (in binary mode)
            overwrite: If False, error if the hash file already exists

        Returns:
            The hex-encoded hash
        """
        path = Path(path)
        hash_path = path.with_suffix(path.suffix + f".{self.alg}")
        if hash_path.exists() and not overwrite:  # check first -- save time
            raise HashFileExistsError(f"Hash file {path} already exists", key=str(path))
        digest = self.calc_hash(path)
        self._add_file_hash(path, hash_path, digest, overwrite=overwrite, dry_run=False)
        return digest

    def append_dir_hash(self, path: PathLike, *, overwrite: Optional[bool] = True) -> str:
        """
        Calculates the hash of ``path`` and appends it to a file ``dir/(dir+"."+alg)``.

        Args:
            path: The path to a file to compute the hash of (in binary mode)
            overwrite: If True, overwrite the file hash and any entry in the dir hash.
                       If False, never overwrite either.
                       If None, never overwrite, but ignore if equal to any existing entries.

        Returns:
            The hex-encoded hash
        """
        path = Path(path)
        hash_path = self.get_hash_dir(path)
        digest = self.calc_hash(path)
        self.append_dir_hashes(hash_path, {path: digest}, overwrite=overwrite)
        return digest

    def verify_hash_from_hex(self, path: PathLike, expected: str) -> Optional[str]:
        """
        Verifies a hash directly from a hex string.
        """
        path = Path(path)
        actual = self.calc_hash(path)
        if actual != expected:
            raise HashDidNotValidateError(
                f"Hash for {path}: calculated {actual} != expected {expected}",
                actual=actual,
                expected=expected,
            )
        return actual

    def verify_any(
        self,
        path: PathLike,
        *,
        file_hash: bool,
        dir_hash: bool,
        computed: Optional[str],
    ) -> Optional[str]:
        path = Path(path)
        if computed is not None:
            self.verify_hash_from_hex(path, computed)
        if file_hash or dir_hash:
            computed = self.calc_hash(path)
        if file_hash:
            self.verify_file_hash(path, computed=computed)
        if dir_hash:
            self.verify_dir_hash(path, computed=computed)
        return computed

    def verify_hash_from_file(
        self,
        path: PathLike,
        hash_path: Path,
        *,
        computed: Optional[str] = None,
    ) -> Optional[str]:
        """
        Verifies a hash directly from a specific hash file.
        The hash file should contain only filename, which is ignored.
        If there are multiple filenames, it will use the one for ``path.name``.

        Args:
            path: The file to calculate the (binary mode) hash of
            hash_path: The path to the hash file
            computed: A pre-computed hex-encoded hash; if set, do not calculate from ``path``
        """
        path = Path(path)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found", key=str(hash_path))
        if computed is None:
            computed = self.calc_hash(path)
        self._verify_file_hash(path, hash_path, computed, use_filename=False)
        return computed

    def verify_file_hash(
        self,
        path: PathLike,
        *,
        use_filename: Optional[bool] = False,
        computed: Optional[str] = None,
    ) -> str:
        """
        Verifies a file against is corresponding hash file.
        The hash file should contain only filename, which is ignored.
        If there are multiple filenames, it will use the one for ``path.name``.

        Args:
            path: The file to calculate the (binary mode) hash of
            use_filename: If True, require the filename in the hash file to match ``path.name``;
                          If False, ignore the filename but require exactly 1 filename listed
                          If None, use either the single filename or the one for path.name.
            computed: A pre-computed hex-encoded hash; if set, do not calculate from ``path``

        Returns:
            The hex-encoded hash

        Raises:
            FileNotFoundError: If ``path`` does not exist
            HashFileMissingError: If the hash file does not exist
            HashDidNotValidateError: If the hashes are not equal
        """
        path = Path(path)
        hash_path = self.get_hash_file(path)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found", key=str(hash_path))
        if computed is None:
            computed = self.calc_hash(path)
        self._verify_file_hash(path, hash_path, computed, use_filename)
        return computed

    def verify_dir_hash(self, path: PathLike, *, computed: Optional[str] = None) -> str:
        """
        Verifies a file against is corresponding per-directory hash file.
        The filename ``path.name`` must be listed in the file.

        Args:
            path: The file to calculate the (binary mode) hash of
            computed: A pre-computed hex-encoded hash; if set, do not calculate from ``path``

        Returns:
            The hex-encoded hash

        Raises:
            FileNotFoundError: If ``path`` does not exist
            HashFileMissingError: If the hash file does not exist
            HashDidNotValidateError: If the hashes are not equal
            HashVerificationError`: Superclass of ``HashDidNotValidateError`` if
                                    the filename is not listed, etc.
        """
        path = Path(path)
        hash_path = self.get_hash_dir(path)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found", key=str(hash_path))
        if not path.exists():
            raise FileNotFoundError(f"Path {path} not found")
        if computed is None:
            computed = self.calc_hash(path)
        self._verify_file_hash(path, hash_path, computed, True)
        return computed

    def calc_hash(self, path: PathLike) -> str:
        """
        Calculates the hash of a file and returns it, hex-encoded.
        """
        path = Path(path)
        alg = getattr(hashlib, self.alg)()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024), b""):
                alg.update(chunk)
        return alg.hexdigest()

    def parse_hash_file_resolved(self, path: PathLike) -> ChecksumMapping:
        """
        Reads a hash file, resolving each read path.

        See Also: :meth:`parse_hash_file_generic`.

        Args:
            path The hash file (e.g. my-dir.sha1)

        Returns:
            A mapping from resolved ``Path`` instances to their hex hashes
        """
        path = Path(path)
        return ChecksumMapping(
            {
                Path(path.parent, *k.split("/")).resolve(): v
                for k, v in self.parse_hash_file_generic(path).items()
            }
        )

    def parse_hash_file_generic(
        self, path: PathLike, *, forbid_slash: bool = False
    ) -> Mapping[str, str]:
        """
        Reads a hash file.

        See Also: :meth:`parse_hash_file_resolved`.

        Args:
            path: The path to read
            forbid_slash: Raise a ``ValueError`` if a a path contains a slash
                          In other words, do not allow specifying subdirectories.
                          In general, most tools do not support these,
                          nor does typeddfs.

        Returns:
            A mapping from raw string filenames to their hex hashes.
            Any node called ``./`` in the path is stripped.
        """
        path = Path(path)
        read = path.read_text(encoding="utf8").splitlines()
        # ignore spaces -- editors often add an extra line break, and it's probably fine anyway
        read = [_hashsum_file_sep.split(s, 1) for s in read if len(s) > 0]
        # obviously this means that / can't appear in a node
        # this is consistent with the commonly accepted spec for shasum
        kv = {
            r[1]: "/".join([n for n in r[0].strip().split() if n != "./"])
            for r in read
            if len(r[1]) != 0
        }
        if forbid_slash:
            slashed = {k for k in kv.keys() if "/" in k}
            if len(slashed) > 0:
                raise ValueError(f"Subdirectory (containing /): {slashed} in {path}")
        return kv

    def get_dir_hash(self, path: PathLike) -> Optional[str]:
        """
        Reads the dir hash corresponding to a file, or None if it does not exist.
        The path is not resolved (symlinks are not followed, and ``path`` is not normalized.)
        """
        path = Path(path)
        dir_hash = self.get_hash_dir(path)
        if dir_hash.exists():
            return self.parse_hash_file_generic(dir_hash).get(path.name)
        return None

    def get_file_hash(self, path: PathLike) -> Optional[str]:
        """
        Reads the hash file corresponding to a file, or None if it does not exist.
        The hash file may only contain 1 entry, and that filename need not match.

        Raises:
            MultipleHashFilenamesError: If multiple files are listed in the hash file
        """
        path = Path(path)
        file_hash = self.get_hash_file(path)
        if file_hash.exists():
            hashes = self.parse_hash_file_generic(file_hash, forbid_slash=True)
            try:
                return self._find_entry(path, file_hash, hashes, use_filename=False)
            except HashFilenameMissingError:
                return None
        return None

    def get_hash_file(self, path: PathLike) -> Path:
        """
        Returns the path required for the per-file hash of ``path``.

        Example:
            ``Utils.get_hash_file("my_file.txt.gz")  # Path("my_file.txt.gz.sha256")``
        """
        path = Path(path)
        return path.with_suffix(path.suffix + "." + self.alg)

    def get_hash_dir(self, path: PathLike) -> Path:
        """
        Returns the path required for the per-directory hash of ``path``.

        Example:
            ``Utils.get_hash_file(Path("my_dir, my_file.txt.gz"))  # Path("my_dir", "my_dir.sha256")``
        """
        path = Path(path)
        return path.parent / (path.parent.name + "." + self.alg)

    @classmethod
    def guess_algorithm(cls, path: PathLike) -> str:
        """
        Guesses the hashlib algorithm used from a hash file.

        Args:
            path The hash file (e.g. my-file.sha256)

        Example:
            ``Utils.guess_algorithm("my_file.sha1")  # "sha1"``
        """
        path = Path(path)
        alg = path.suffix.lstrip(".").lower().replace("-", "")
        try:
            getattr(hashlib, alg)
        except AttributeError:
            raise HashAlgorithmMissingError(f"No hashlib algorithm {alg}", key=alg) from None
        return alg

    @classmethod
    def get_algorithm(cls, alg: str) -> str:
        """
        Finds a hash algorithm by name in :mod:`hashlib`.
        Converts to lowercase and removes hyphens.

        Raises:
            HashAlgorithmMissingError: If not found
        """
        alg = alg.lower().replace("-", "")
        try:
            getattr(hashlib, alg)
        except AttributeError:
            raise HashAlgorithmMissingError(f"No hashlib algorithm {alg}", key=alg) from None
        return alg

    def delete_dir_hash(self, path: PathLike, *, missing_ok: bool = False) -> None:
        """
        Strips a single path from a dir hash file.
        Less flexible than :meth:`delete_dir_hashes` and :meth:`update_dir_hashes`.
        """
        self.delete_dir_hashes(self.get_hash_dir(path), [path], missing_ok=missing_ok)

    def delete_file_hash(self, path: PathLike, *, missing_ok: bool = False) -> None:
        """
        Deletes the hash file of a file.
        """
        self.get_hash_file(path).unlink(missing_ok=missing_ok)

    def delete_dir_hashes(
        self,
        hash_path: PathLike,
        delete: Iterable[PathLike],
        *,
        missing_ok: bool = False,
        rm: bool = False,
    ) -> None:
        """
        Strips paths from a dir hash file.
        Like :meth:`update_dir_hashes` but less flexible and only for removing paths.
        """
        self.update_dir_hashes(
            hash_path, {p: None for p in delete}, missing_ok=missing_ok, overwrite=True, rm=rm
        )

    def append_dir_hashes(
        self,
        hash_path: PathLike,
        append: Mapping[PathLike, str],
        *,
        overwrite: Optional[bool] = False,
    ):
        """
        Append paths to a dir hash file.
        Like :meth:`update_dir_hashes` but less flexible and only for adding paths.
        """
        self.update_dir_hashes(hash_path, append, missing_ok=True, overwrite=overwrite)

    def update_dir_hashes(
        self,
        hash_path: PathLike,
        update: Union[Callable[[Path], Optional[PathLike]], Mapping[PathLike, Optional[PathLike]]],
        *,
        missing_ok: bool = True,
        overwrite: Optional[bool] = True,
        sort: Union[bool, Callable[[Mapping[Path, str]], Mapping[Path, str]]] = False,
        rm: bool = False,
    ) -> None:
        """
        Reads a dir hash file and writes back with new values.
        Can add, update, and delete.

        Args:
            hash_path: The path of the checksum file (e.g. "dir.sha256")
            update: Values to overwrite.
                    May be a function or a dictionary from paths to values.
                    If ``None`` is returned, the entry will be removed;
                    otherwise, updates with the returned hex hash.
            missing_ok: Require that the path is already listed
            overwrite: Allow overwriting an existing value.
                       If ``None``, only allow if the hash is the same.
            sort: Apply a sorting algorithm afterward.
                  If ``True``, uses ``sorted``, sorting only on the keys.
            rm: Delete the file if it contains no more hashes.
        """
        hash_path = Path(hash_path)
        existing = self.parse_hash_file_resolved(hash_path) if hash_path.exists() else {}
        fixed = self.get_updated_hashes(
            existing, update, missing_ok=missing_ok, overwrite=overwrite, sort=sort
        )
        if len(fixed) == 0 and rm:
            hash_path.unlink()
        else:
            hash_path.write_text(os.linesep.join(fixed.lines), encoding="utf8")

    def get_updated_hashes(
        self,
        existing: Mapping[PathLike, str],
        update: Union[Callable[[Path], Optional[PathLike]], Mapping[PathLike, Optional[PathLike]]],
        *,
        missing_ok: bool = True,
        overwrite: Optional[bool] = True,
        sort: Union[bool, Callable[[Sequence[Path]], Sequence[Path]]] = False,
    ) -> ChecksumMappingOpt:
        """
        Returns updated hashes from a dir hash file.
        See :meth:`update_dir_hashes`; this just returns values instead of reading and writing.

        Returns:
            A Mapping from resolved Paths; to: hex-encoded digests or ``None`` to indicate removal.
            Has a method :meth:`typedfs.checksums.ChecksumMappingOpt.lines`.
        """
        existing = ChecksumMapping({Path(p).resolve(): h for p, h in existing.items()})
        fixed = {}
        for p, v in existing.items():
            v_new = update(p) if callable(update) else update.get(p, v)
            fixed[p] = self._get_updated_hash(
                path=p,
                new=v_new,
                existing=existing,
                missing_ok=missing_ok,
                overwrite=overwrite,
            )
        if not callable(update):
            for p, v in update.items():
                p = Path(p).resolve()
                fixed[p] = self._get_updated_hash(
                    path=p,
                    new=v,
                    existing=existing,
                    missing_ok=missing_ok,
                    overwrite=overwrite,
                )
        if sort is True:
            fixed = sorted(fixed)
        elif callable(sort):
            fixed = sort(fixed)
        return ChecksumMappingOpt(fixed)

    def _add_file_hash(
        self, path: Path, hash_path: Path, digest: str, *, overwrite: Optional[bool], dry_run: bool
    ) -> None:
        path = Path(path)
        if path.exists() and overwrite is False:
            raise HashFileExistsError(f"Hash file {path} already exists")
        if hash_path.exists() and overwrite is None:
            self.verify_file_hash(path, computed=digest)
            # it's ok -- they're the same
        txt = f"{digest} *{path.name}"
        if not dry_run:
            hash_path.write_text(txt, encoding="utf-8")

    def _get_updated_hash(
        self,
        *,
        path: Path,
        new: Optional[str],
        existing: Mapping[Path, str],
        overwrite: Optional[bool],
        missing_ok: bool,
    ) -> Optional[str]:
        path = path.resolve()
        z = existing.get(path)
        if z is None and not missing_ok:
            raise HashFilenameMissingError(f"{path} not found ({len(existing)} are)")
        if z is not None:
            err = None
            if overwrite is False and z == new:
                err = (
                    HashExistsError,
                    f"Hash for {path} already exists (and it matches)",
                )
            if overwrite is False and z != new:
                err = (
                    HashExistsError,
                    f"Hash for {path} already exists (and it does not match)",
                )
            if overwrite is None and z != new:
                err = (
                    HashContradictsExistingError,
                    f"Hash for {path} already exists but does not match",
                )
            if err is not None:
                raise err[0](err[1], key=str(path), original=z, new=new)
        return new

    def _verify_file_hash(
        self, path: Path, hash_path: Path, actual: str, use_filename: Optional[bool]
    ) -> None:
        path = Path(path)
        hashes = self.parse_hash_file_generic(hash_path)
        expected = self._find_entry(path, hash_path, hashes, use_filename=use_filename)
        if actual != expected:
            raise HashDidNotValidateError(
                f"Hash for {path} from {hash_path}: calculated {actual} != expected {expected}",
                actual=actual,
                expected=expected,
            )

    def _find_entry(self, path: Path, hash_path: Path, hashes, *, use_filename: Optional[bool]):
        name = path.name
        if (use_filename is None or use_filename is True) and name in hashes:
            return hashes[path.name]
        elif (use_filename is None or use_filename is False) and len(hashes) == 1:
            return next(iter(hashes.values()))
        elif len(hashes) > 1 and use_filename is False:
            raise MultipleHashFilenamesError(f"{hash_path} contains multiple filenames", key=name)
        else:
            raise HashFilenameMissingError(
                f"{name} not found in {hash_path}" + f" [listed: {', '.join(hashes.keys())}]",
                key=name,
            )


__all__ = ["Checksums", "ChecksumMapping"]
