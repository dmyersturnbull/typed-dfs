"""
Tools for shasum-like files.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from typeddfs.df_errors import (
    HashAlgorithmMissingError,
    HashDidNotValidateError,
    HashFileExistsError,
    HashFileMissingError,
    HashFilenameMissingError,
    MultipleHashFilenamesError,
    PathNotRelativeError,
)
from typeddfs.utils._utils import _DEFAULT_HASH_ALG, PathLike
from typeddfs.utils.checksum_models import ChecksumFile, ChecksumMapping


@dataclass(frozen=True, repr=True, order=True)
class Checksums:
    alg: str = _DEFAULT_HASH_ALG

    @classmethod
    def default_algorithm(cls) -> str:
        return _DEFAULT_HASH_ALG

    def write_any(
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
        if not to_file and not to_dir:
            return None
        fh, dh = None, None
        x, y = None, None
        path = Path(path)
        hash_file_path = self.get_filesum_of_file(path)
        if to_file:
            if hash_file_path.exists():
                fh = self.load_filesum_exact(hash_file_path)
            else:
                fh = ChecksumFile.new(hash_file_path, file_path=path, hash_value="")
            y = fh.hash_value
            if y != "" and overwrite is False:  # check first -- save time
                raise HashFileExistsError(f"Hash file of {path} already exists", key=str(path))
        hash_dir_path = self.get_dirsum_of_file(path)
        if to_dir:
            dh = self.load_dirsum_exact(hash_dir_path)
            x = dh.get(path)
            if x is not None and overwrite is False:
                raise MultipleHashFilenamesError(
                    f"Path {path} listed in {hash_dir_path}", key=str(path)
                )
        digest = self.calc_hash(path)
        if overwrite is None:
            if x is not None and x != digest:
                raise MultipleHashFilenamesError(
                    f"Path {path} listed in {hash_dir_path}", key=str(path)
                )
            if y is not None and y != digest:
                raise MultipleHashFilenamesError(
                    f"Path {path} listed in {hash_dir_path}", key=str(path)
                )
        if to_file:
            fh = fh.update(digest, overwrite=overwrite)
        if to_dir:
            dh = dh.update({path: digest})
        # write only at the very end:
        if to_file:
            fh.write()
        if to_dir:
            dh.write()
        return digest

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
            self.verify_hex(path, computed)
        hash_file_path = self.get_filesum_of_file(path)
        hash_dir_path = self.get_dirsum_of_file(path)
        # check first to save time:
        if file_hash and not hash_file_path.exists():
            raise HashFileMissingError(f"File hash of {path} not found", key=str(path))
        if dir_hash and not hash_dir_path.exists():
            raise HashFilenameMissingError(f"Hash of {path} not in {hash_dir_path}", key=str(path))
        # now calculate the actual hash for comparison
        if file_hash or dir_hash:
            computed = self.calc_hash(path)
        # check it:
        if file_hash:
            fh = self.load_filesum_exact(hash_file_path)
            fh.verify(computed)
        if dir_hash:
            dh = self.load_dirsum_exact(hash_dir_path)
            dh.verify(path, computed)
        return computed

    def delete_any(self, path: PathLike, *, rm_if_empty: bool = False) -> None:
        """
        Deletes the filesum and removes ``path`` from the dirsum.
        Ignores missing files.
        """
        path = Path(path)
        self.get_filesum_of_file(path).unlink(missing_ok=True)
        try:
            ds = self.load_dirsum_of_file(path, missing_ok=True)
            ds.remove(path, missing_ok=True).write(rm_if_empty=rm_if_empty)
        except PathNotRelativeError:
            pass

    def verify_hex(self, path: PathLike, expected: str) -> Optional[str]:
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

    def generate_dirsum(self, directory: PathLike, glob: str = "*") -> ChecksumMapping:
        """
        Generates a new hash mapping, calculating hashes for extant files.

        Args:
            directory: Base directory
            glob: Glob pattern under ``directory`` (cannot be recursive)

        Returns:
            A ChecksumMapping; use ``.write`` to write it
        """
        directory = Path(directory)
        path = self.get_dirsum_of_dir(directory)
        sums = {p: self.calc_hash(p) for p in directory.glob(glob)}
        return ChecksumMapping(path, sums)

    def load_filesum_of_file(self, path: PathLike) -> ChecksumFile:
        hash_file = self.get_filesum_of_file(path)
        return ChecksumFile.parse(hash_file)

    def load_dirsum_of_file(self, path: PathLike, *, missing_ok: bool = True) -> ChecksumMapping:
        hash_dir = self.get_dirsum_of_file(path)
        return ChecksumMapping.parse(hash_dir, missing_ok=missing_ok)

    def load_dirsum_of_dir(self, path: PathLike, *, missing_ok: bool = True) -> ChecksumMapping:
        hash_dir = self.get_dirsum_of_dir(path)
        return ChecksumMapping.parse(hash_dir, missing_ok=missing_ok)

    def load_dirsum_exact(self, path: PathLike, *, missing_ok: bool = True) -> ChecksumMapping:
        return ChecksumMapping.parse(Path(path), missing_ok=missing_ok)

    def load_filesum_exact(self, path: PathLike) -> ChecksumFile:
        return ChecksumFile.parse(Path(path))

    def get_filesum_of_file(self, path: PathLike) -> Path:
        """
        Returns the path required for the per-file hash of ``path``.

        Example:
            ``Utils.get_hash_file("my_file.txt.gz")  # Path("my_file.txt.gz.sha256")``
        """
        path = Path(path)
        return path.with_suffix(path.suffix + "." + self.alg)

    def get_dirsum_of_file(self, path: PathLike) -> Path:
        """
        Returns the path required for the per-directory hash of ``path``.

        Example:
            ``Utils.get_hash_file(Path("my_dir, my_file.txt.gz"))  # Path("my_dir", "my_dir.sha256")``
        """
        path = Path(path)
        return path.parent / (path.parent.name + "." + self.alg)

    def get_dirsum_of_dir(self, path: PathLike) -> Path:
        """
        Returns the path required for the per-directory hash of ``path``.

        Example:
            ``Utils.get_hash_file("my_dir")  # Path("my_dir", "my_dir.sha256")``
        """
        path = Path(path)
        return path / (path.name + "." + self.alg)

    @classmethod
    def guess_algorithm(cls, path: PathLike) -> str:
        """
        Guesses the hashlib algorithm used from a hash file.

        Args:
            path: The hash file (e.g. my-file.sha256)

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
    def resolve_algorithm(cls, alg: str) -> str:
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


__all__ = ["Checksums"]
