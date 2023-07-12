# SPDX-License-Identifier Apache-2.0
# Source: https://github.com/dmyersturnbull/typed-dfs
#
"""
Exceptions used by typeddfs.
"""
from collections.abc import Sequence
from pathlib import PurePath
from typing import AbstractSet, Optional, Union


class UnsupportedOperationError(Exception):
    """
    Something could not be performed, in general.
    """


class FilenameSuffixError(UnsupportedOperationError):
    """
    A filename extension was not recognized.

    Attributes:
        key: The unrecognized suffix
        filename: The bad filename
    """

    def __init__(self, *args, key: str | None = None, filename: str | None = None):
        super().__init__(*args)
        self.key = key
        self.filename = filename


class FormatInsecureError(UnsupportedOperationError):
    """
    A requested format is less secure than required or requested.

    Attributes:
        key: The problematic format name
    """

    def __init__(self, *args, key: str | None = None):
        super().__init__(*args)
        self.key = key


class FormatDiscouragedError(UnsupportedOperationError):
    """
    A requested format is not recommended.

    Attributes:
        key: The problematic format name
    """

    def __init__(self, *args, key: str | None = None):
        super().__init__(*args)
        self.key = key


class NonStrColumnError(UnsupportedOperationError):
    """
    A column name is not a string.
    """


class NotSingleColumnError(UnsupportedOperationError):
    """
    A DataFrame needs to contain exactly 1 column.
    """


class DfTypeConstructionError(Exception):
    """
    An inconsistency prevents creating the DataFrame type.
    """


class ClashError(DfTypeConstructionError):
    """
    Duplicate columns or other keys were added.

    Attributes:
        keys: The clashing name(s)
    """

    def __init__(self, *args, keys: AbstractSet[str] | None = None):
        super().__init__(*args)
        self.keys = keys


class InvalidDfError(ValueError):
    """
    A general typing failure of typeddfs.
    """


class VerificationFailedError(InvalidDfError):
    """
    A custom typing verification failed.

    Attributes:
        key: The key name of the verification that failed
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class MissingColumnError(InvalidDfError):
    """
    A required column is missing.

    Attributes:
        key: The name of the missing column
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class RowColumnMismatchError(InvalidDfError):
    """
    The row and column names differ.

    Attributes:
        rows: The row names, in order
        columns: The column names, in order
    """

    def __init__(
        self, *args, rows: Sequence[str] | None = None, columns: Sequence[str] | None = None
    ):  # pragma: no cover
        super().__init__(*args)
        self.rows = rows
        self.columns = columns


class UnexpectedColumnError(InvalidDfError):
    """
    An extra/unrecognized column is present.

    Attributes:
        key: The name of the unexpected column
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class UnexpectedIndexNameError(InvalidDfError):
    """
    An extra/unrecognized index level is present.

    Attributes:
        key: The name of the unexpected index level
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class ValueNotUniqueError(ValueError):
    """
    There is more than 1 unique value.

    Attributes:
        key: The key used for lookup
        values: The set of values
    """

    def __init__(
        self, *args, key: str | None = None, values: AbstractSet[str] | None = None
    ):  # pragma: no cover
        super().__init__(*args)
        self.key = key
        self.values = values


class NoValueError(ValueError):
    """
    No value because the collection is empty.

    Attributes:
        key: The key used for lookup
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class LengthMismatchError(ValueError):
    """
    The lengths of at least two lists do not match.

    Attributes:
        key: The key used for lookup
        lengths: The lengths
    """

    def __init__(
        self, *args, key: str | None = None, lengths: AbstractSet[int]
    ):  # pragma: no cover
        super().__init__(*args)
        self.key = key
        self.lengths = lengths


class ReadPermissionsError(OSError):
    """
    Couldn't read from a file.
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class WritePermissionsError(OSError):
    """
    Couldn't write to a file.
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class HashError(OSError):
    """
    Something went wrong with hash file writing or reading.
    """


class HashWriteError(HashError):
    """
    Something went wrong when writing a hash file.
    """


class HashExistsError(HashWriteError, ValueError):
    """
    A hash for the filename already exists in the directory hash list.

    Attributes:
        key: The filename (excluding parents)
        original: Hex hash found listed for the file
        new: Hex hash that was to be written
        filename: The filename of the listed file
    """

    def __init__(
        self,
        *args,
        key: str | None = None,
        original: str | None = None,
        new: str | None = None,
    ):  # pragma: no cover
        super().__init__(*args)
        self.key = key
        self.original = original
        self.new = new


class HashContradictsExistingError(HashExistsError):
    """
    A hash for the filename already exists in the directory hash list, but they differ.

    Attributes:
        key: The filename (excluding parents)
        original: Hex hash found listed for the file
        new: Hex hash that was to be written
        filename: The filename of the listed file
    """


class HashAlgorithmMissingError(HashWriteError, LookupError):
    """
    The hash algorithm was not found in :mod:`hashlib`.

    Attributes:
        key: The missing hash algorithm
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class HashVerificationError(HashError):
    """
    Something went wrong when validating a hash.
    """


class HashDidNotValidateError(HashVerificationError):
    """
    The hashes did not validate (expected != actual).

    Attributes:
        actual: The actual hex-encoded hash
        expected: The expected hex-encoded hash
    """

    def __init__(
        self, *args, actual: str | None = None, expected: str | None = None
    ):  # pragma: no cover
        super().__init__(*args)
        self.actual = actual
        self.expected = expected


class HashFileInvalidError(HashVerificationError, ValueError):
    """
    The hash file could not be parsed.

    Attributes:
        key: The path to the hash file
    """

    def __init__(self, *args, key: None | PurePath | str = None):  # pragma: no cover
        super().__init__(*args)
        if isinstance(key, PurePath):
            key = str(key)
        self.key = key


class HashFileMissingError(HashVerificationError, FileNotFoundError):
    """
    The hash file does not exist.

    Attributes:
        key: The path or filename of the file corresponding to the expected hash file(s)
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class HashFilenameMissingError(HashVerificationError, LookupError):
    """
    The filename was not found listed in the hash file.

    Attributes:
        key: The filename
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class MultipleHashFilenamesError(HashVerificationError, ValueError):
    """
    There are multiple filenames listed in the hash file where only 1 was expected.

    Attributes:
        key: The filename with duplicate entries
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class HashFileExistsError(HashVerificationError, FileExistsError):
    """
    The hash file already exists and cannot be overwritten.

    Attributes:
        key: The existing hash file path or filename
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class HashEntryExistsError(HashVerificationError, FileExistsError):
    """
    The file is already listed in the hash dir, and it cannot be overwritten.

    Attributes:
        key: The existing hash dir path
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key


class PathNotRelativeError(ValueError):
    """
    The filename is not relative to the hash dir.

    Attributes:
        key: The filename
    """

    def __init__(self, *args, key: str | None = None):  # pragma: no cover
        super().__init__(*args)
        self.key = key
