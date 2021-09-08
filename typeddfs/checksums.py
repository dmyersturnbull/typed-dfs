"""
Tools for shasum-like files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Mapping

import regex

from typeddfs._utils import _DEFAULT_HASH_ALG
from typeddfs.df_errors import (
    HashFilenameMissingError,
    MultipleHashFilenamesError,
    HashFileMissingError,
    HashContradictsExistingError,
    HashFileExistsError,
    HashAlgorithmMissingError,
    HashDidNotValidateError,
)


_hex_pattern = regex.compile(r"[A-Ha-h0-9]+", flags=regex.V1)
_hashsum_file_sep = regex.compile(r" [ *]", flags=regex.V1)


class Checksums:
    @classmethod
    def default_algorithm(cls) -> str:
        return _DEFAULT_HASH_ALG

    @classmethod
    def add_any_hashes(
        cls,
        path: Path,
        to_file: bool,
        to_dir: bool,
        *,
        algorithm: str = _DEFAULT_HASH_ALG,
        overwrite: bool = True,
    ) -> Optional[str]:
        """
        Adds and/or appends the hex hash of ``path``.
        """
        algorithm = cls.get_algorithm(algorithm)
        hash_file_path = cls.get_hash_file(path, algorithm=algorithm)
        hash_dir_path = cls.get_hash_dir(path, algorithm=algorithm)
        if to_file and hash_file_path.exists() and not overwrite:  # check first -- save time
            raise HashFileExistsError(f"Hash file {path} already exists", key=str(path))
        if not to_file and not to_dir:
            return None
        digest = cls.calc_hash(path, algorithm=algorithm)
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
        hash_path = cls.get_hash_dir(path, algorithm=algorithm)
        digest = cls.calc_hash(path)
        cls._append_dir_hash(path, hash_path, digest, overwrite)
        return digest

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
    def verify_any(
        cls,
        path: Path,
        file_hash: bool,
        dir_hash: bool,
        computed: Optional[str],
        *,
        algorithm: str = _DEFAULT_HASH_ALG,
    ) -> Optional[str]:
        if computed is not None:
            cls.verify_hash_from_hex(path, computed)
        if file_hash or dir_hash:
            computed = cls.calc_hash(path, algorithm=algorithm)
            if file_hash:
                cls.verify_file_hash(path, algorithm=algorithm, computed=computed)
            if dir_hash:
                cls.verify_dir_hash(path, algorithm=algorithm, computed=computed)
        return computed

    @classmethod
    def verify_hash_from_file(
        cls,
        path: Path,
        hash_path: Path,
        *,
        algorithm: Optional[str] = _DEFAULT_HASH_ALG,
        computed: Optional[str] = None,
    ) -> Optional[str]:
        """
        Verifies a hash directly from a specific hash file.
        The hash file should contain only filename, which is ignored.
        If there are multiple filenames, it will use the one for ``path.name``.

        Args:
            path: The file to calculate the (binary mode) hash of
            hash_path: The path to the hash file
            algorithm: The algorithm in hashlib (ignored if ``computed`` is passed)
            computed: A pre-computed hex-encoded hash; if set, do not calculate from ``path``
        """
        if algorithm is None:
            algorithm = cls.guess_algorithm(hash_path)
        else:
            algorithm = cls.get_algorithm(algorithm)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found")
        if computed is None:
            computed = cls.calc_hash(path, algorithm=algorithm)
        cls._verify_file_hash(path, hash_path, computed, use_filename=None)
        return computed

    @classmethod
    def verify_file_hash(
        cls,
        path: Path,
        *,
        algorithm: str = _DEFAULT_HASH_ALG,
        use_filename: Optional[bool] = None,
        computed: Optional[str] = None,
    ) -> str:
        """
        Verifies a file against is corresponding hash file.
        The hash file should contain only filename, which is ignored.
        If there are multiple filenames, it will use the one for ``path.name``.

        Args:
            path: The file to calculate the (binary mode) hash of
            algorithm: The algorithm in hashlib (ignored if ``computed`` is passed)
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
        hash_path = cls.get_hash_file(path, algorithm=algorithm)
        algorithm = cls.get_algorithm(algorithm)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found")
        if computed is None:
            computed = cls.calc_hash(path, algorithm=algorithm)
        cls._verify_file_hash(path, hash_path, computed, use_filename)
        return computed

    @classmethod
    def verify_dir_hash(
        cls, path: Path, *, algorithm: str = _DEFAULT_HASH_ALG, computed: Optional[str] = None
    ) -> str:
        """
        Verifies a file against is corresponding per-directory hash file.
        The filename ``path.name`` must be listed in the file.

        Args:
            path: The file to calculate the (binary mode) hash of
            algorithm: The algorithm in hashlib (ignored if ``computed`` is passed)
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
        hash_path = cls.get_hash_dir(path, algorithm=algorithm)
        algorithm = cls.get_algorithm(algorithm)
        if not hash_path.exists():
            raise HashFileMissingError(f"No hash file {hash_path} found")
        if not path.exists():
            raise FileNotFoundError(f"Path {path} not found")
        if computed is None:
            computed = cls.calc_hash(path, algorithm=algorithm)
        cls._verify_file_hash(path, hash_path, computed, True)
        return computed

    @classmethod
    def calc_hash(cls, path: Path, *, algorithm: str = _DEFAULT_HASH_ALG) -> str:
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
    def parse_hash_file_resolved(cls, path: Path) -> Mapping[Path, str]:
        """
        Reads a hash file.

        See Also: :py.meth:`parse_hash_file_generic`.

        Returns:
            A mapping from resolved ``Path`` instances to their hex hashes
        """
        return {
            Path(path.parent, *k.split("/")).resolve(): v
            for k, v in cls.parse_hash_file_generic(path).items()
        }

    @classmethod
    def parse_hash_file_generic(
        cls, path: Path, *, forbid_slash: bool = False
    ) -> Mapping[str, str]:
        """
        Reads a hash file.

        See Also: :py.meth:`parse_hash_file_resolved`.

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
        read = path.read_text(encoding="utf8").splitlines()
        read = [_hashsum_file_sep.split(s, 1) for s in read]
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

    @classmethod
    def get_hash_file(cls, path: Path, *, algorithm: str = _DEFAULT_HASH_ALG) -> Path:
        """
        Returns the path required for the per-file hash of ``path``.

        Example:
            Utils.get_hash_file("my_file.txt.gz")  # Path("my_file.txt.gz.sha256")
        """
        algorithm = cls.get_algorithm(algorithm)
        return path.with_suffix(path.suffix + "." + algorithm)

    @classmethod
    def get_hash_dir(cls, path: Path, *, algorithm: str = _DEFAULT_HASH_ALG) -> Path:
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
            existing = cls.parse_hash_file_resolved(hash_path)
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
        hashes = cls.parse_hash_file_resolved(hash_path)
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


__all__ = ["Checksums"]
